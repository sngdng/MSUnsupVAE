import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Union, Set, Tuple, Dict

import wandb
import torch
import numpy as np
from datasets import load_metric, Metric
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    default_data_collator,
    PreTrainedTokenizer,
    PreTrainedModel,
    T5ForConditionalGeneration,
)

from d2t.data.datasets import Seq2seqDataset
from d2t.data.formatting import Example, DataFormat, construct_all_prefixes
from d2t.eval.utils import get_precision_recall_f1, compute_meteor_score
from d2t.eval.metrics.sembleu_score import NgramInstance, corpus_bleu
from d2t.utils import MyLogger, Mode
from nltk.tokenize import word_tokenize


def distinct(seqs):
    """
    Taken from https://github.com/PaddlePaddle/models/blob/release/1.6/PaddleNLP/Research/Dialogue-PLATO/plato/metrics/metrics.py
    Calculate intra/inter distinct 1/2. 
    """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


class VAEvaluator:
    # Evaluator for VAE 
    # text -> generate a certain number of texts for BLEU
    # data -> compute precision, recall and f1 score with data parsing
    # -- d2t metrics using hugging metrics for consistency
    # # use sacreBLEU (a standardized version of BLEU, using a standard tokenizer) and rouge
    # use METEORv1.5 from original java file
    # -- t2d metrics
    # # use adapted semBLEU for graphs and prec, rec, f1 scores for pred entites/relations

    def __init__(
        self,
        mode: Mode,
        datasets: Dict[str, Seq2seqDataset],
        tokenizer: PreTrainedTokenizer,
        model,
        device,
        batch_size: int,
        num_beams_t2d: int,
        num_beams_d2t: int,
        log_path: Path,
        num_sx: int,
        dim_sx: int,
        tensorboard_writer: SummaryWriter = None,
    ):
        self.mode = mode
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.prefixes = construct_all_prefixes(tokenizer)
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_beams_t2d = num_beams_t2d
        self.num_beams_d2t = num_beams_d2t
        self.log_path = log_path
        self.num_sx = num_sx
        self.dim_sx = dim_sx
        metrics_path = Path(__file__).resolve().parents[0]
        self.d2t_metrics = {"bleu": load_metric(str((metrics_path / 'metrics' / 'sacrebleu.py'))),
                            "rouge": load_metric(str((metrics_path / 'metrics' / 'rouge.py'))),
                            "meteor": 0.0}
        self.logger = MyLogger(
            tensorboard_writer=tensorboard_writer,
            log_every_n_steps=1,
            use_loggers=True,
            accelerator=None
        )
        self.limit_samples = None

    def evaluate_and_log(self):
        """
        Evaluate model on val or test dataset.
        Log metrics to tensorboard & wandb, log artifacts (input/predictions/labels)
        to wandb as txt files
        """

        # get model ready for inference
        self.model.eval()  # .no_grad() already called by model.generate(), but not .eval()

        for split in self.datasets.keys():
            # run the evaluation loop: do inference and compute metrics
            logging.info(f"Evaluating on {split}...")
            dataset = self.datasets[split]
            logs_t2d, logs_d2t = "", ""
            if self.mode == Mode.t2d:
                metrics, logs_t2d = self.run_evaluation_t2d(dataset, split)
                # print and save eval metrics
                logging.info(f"Eval results: {metrics}")
                self.logger.log_metrics(metrics)
            elif self.mode == Mode.d2t:
                metrics, logs_d2t = self.run_evaluation_d2t(dataset, split)
                # print and save eval metrics
                logging.info(f"Eval results: {metrics}")
                self.logger.log_metrics(metrics)
            elif self.mode == Mode.both_sup or self.mode == Mode.both_unsup:
                metrics_t2d, logs_t2d = self.run_evaluation_t2d(dataset, split)
                # print and save eval metrics
                logging.info(f"Eval results: {metrics_t2d}")
                self.logger.log_metrics(metrics_t2d)
                metrics_d2t, logs_d2t = self.run_evaluation_d2t(dataset, split)
                # print and save eval metrics
                logging.info(f"Eval results: {metrics_d2t}")
                self.logger.log_metrics(metrics_d2t)
            else:
                raise ValueError

            # save predictions logs to wandb
            for mode, logs in {"d2t": logs_d2t, "t2d": logs_t2d}.items():
                logs_path = self.log_path / f"{mode}_out/{mode}_{split}.txt"
                self.logger.log_text(
                    text=logs, file_path=logs_path, folder_name=f"{mode}_out"
                )

    def evaluate_d2t_batch(
        self, batch, example_ids, dataset: Seq2seqDataset
    ):
        batch_data = batch["data_ids"].to(self.device)
        if len(dataset.id2format) > 1:
            syn_text_predictions_ids = self.model.generate_with_target(
                batch_data,
                target="text",
                prefixes=self.prefixes,
                max_seq_length=dataset.max_seq_length,
                method="top_k",
                vae_latent_s=torch.zeros((batch_data.size(0), self.dim_sx)).to(self.device)
            )
            # encode the style posterior
            style_outputs = self.model.encoder(
                input_ids=syn_text_predictions_ids,
                source="text",
                return_dict=True,
            )

        batch_text_predictions = []
        batch_references = []
        list_of_references = [dataset.get_example(example_id).references for example_id in example_ids]

        for i in range(self.num_sx):
            kwargs = {}
            if len(dataset.id2format) > 1:
                vae_s = style_outputs.q_phi_s.sample()
                kwargs['vae_latent_s'] = vae_s
            # get raw batch predictions
            text_predictions_ids = self.model.generate_with_target(
                batch_data,
                target="text",
                prefixes=self.prefixes,
                max_seq_length=dataset.max_seq_length,
                method="beam_search",
                num_beams=self.num_beams_d2t,
                **kwargs
            )
            # transform predictions and references to plain text (detokenized)
            text_predictions = self.tokenizer.batch_decode(
                text_predictions_ids, skip_special_tokens=True
            )
            #references = self.tokenizer.batch_decode(
            #    batch["text_ids"], skip_special_tokens=True
            #)
            batch_text_predictions.append(text_predictions)

        logs = ""
        for i in range(len(batch_data)):
            current_id = example_ids[i]
            example = dataset.get_example(current_id)
            logs += (
                    f"[{current_id}] input data / refs / outputs \n"
                    f"{example.data}\n"
                )
            # Writing refs
            for j in range(len(list_of_references[i])):
                logs += f"{list_of_references[i][j]}\n"
            
            for k in range(self.num_sx):
                # Writing outputs
                logs += f"{batch_text_predictions[k][i]}\n"
        
        return batch_text_predictions, list_of_references, logs
    
    def run_evaluation_d2t(self, dataset, split: str):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        text_predictions = {k: [] for k in range(self.num_sx)}
        refs = []
        logs = ""
        num_samples = 0
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.limit_samples and (i + 1) * self.batch_size > self.limit_samples:
                # to speed up validation (for a quick test), do not consider all samples
                break
            example_ids = range(num_samples, num_samples + len(batch["data_ids"]))
            batch_text_predictions, batch_refs, batch_logs = self.evaluate_d2t_batch(batch, example_ids, dataset)
            for j in range(self.num_sx):
                text_predictions[j] += batch_text_predictions[j]
            refs += batch_refs
            logs += batch_logs
            num_samples += len(batch["data_ids"])

        metrics = self.compute_d2t_metrics(text_predictions, refs)
        diversity_metrics = self.compute_d2t_diversity_metrics(text_predictions)
        metrics.update(diversity_metrics)
        meteor_results = []
        # For METEORv1.5 need to specify a fixed number of references for each prediction
        # # we crop the references to the min of length
        ref_min_len = min([len(ref) for ref in refs])
        cropped_refs = [references[:ref_min_len] for references in refs]
        for i in range(self.num_sx):
            meteor_results.append(compute_meteor_score(cropped_refs, text_predictions[i], ref_min_len))
        metrics['meteor_min'] = np.min(meteor_results)
        metrics['meteor_mean'] = np.mean(meteor_results)
        metrics['meteor_max'] = np.max(meteor_results)
        metrics = {f"{split}/{k}": v for k, v in metrics.items()}

        return metrics, logs
    
    def compute_d2t_diversity_metrics(self, preds):
        # preds (num_sx, B, T), refs (B, T)
        
        self_bleu_results = []
        distinct_1_results = []
        distinct_2_results = []
        n = len(preds[0])
        for i in range(n):
            inter_bleu = 0
            tok_preds = []
            for j in range(self.num_sx):
                tok_preds.append(word_tokenize(preds[j][i]))
                other_preds = [preds[k][i] for k in range(self.num_sx) if k != j]
                # Compute normalized selfBLEU
                inter_bleu += self.d2t_metrics['bleu'].compute(predictions=[preds[j][i]], references=[other_preds])["score"]
            self_bleu_results.append(inter_bleu / self.num_sx)
            # Compute Distinct-N
            _, _, inter_dist1, inter_dist2 = distinct(tok_preds)
            distinct_1_results.append(inter_dist1)
            distinct_2_results.append(inter_dist2)

        return {
            "distinct_1_mean": np.mean(distinct_1_results),
            "distinct_1_std": np.std(distinct_1_results),
            "distinct_2_mean": np.mean(distinct_2_results),
            "distinct_2_std": np.std(distinct_2_results),
            "self_bleu_mean": np.mean(self_bleu_results),
            "self_bleu_std": np.std(self_bleu_results)
        }

    
    def compute_d2t_metrics(self, preds, refs):
        bleu_results = []
        rouge_results = []
        bleu_sys_len = []
        bleu_ref_len = []
        n = len(preds)
        # since hf sacreBLEU only support references with same length, we have to pad them into the same length
        ref_max_len = max([len(ref) for ref in refs])
        # see https://github.com/mjpost/sacrebleu/pull/132
        padded_refs = [references + [None] * (ref_max_len - len(references)) for references in refs]

        for i in range(self.num_sx):
            bleu = self.d2t_metrics['bleu']
            bleu_result = bleu.compute(predictions=preds[i], references=padded_refs)
            bleu_results.append(bleu_result["score"])
            bleu_sys_len.append(bleu_result["sys_len"])
            bleu_ref_len.append(bleu_result["ref_len"])

            rouge_result = self.d2t_metrics['rouge'].compute(predictions=preds[i], references=refs)
            rouge_results.append(rouge_result["rougeL"].mid.fmeasure)
            #bertscore_results = self.d2t_metrics["bertscore"].compute(lang='en')
            # to match cyclegt evaluation, compute mean F1 score of rougeL metric
            # (mid is the mean, see https://github.com/google-research/google-research/blob/master/rouge/scoring.py#L141)
        
        return {
            "bleu_min": np.min(bleu_results),
            "bleu_mean": np.mean(bleu_results),
            "bleu_max": np.max(bleu_results),
            "bleu_std": np.std(bleu_results),
            "rouge_l_min": np.min(rouge_results),
            "rouge_l_mean": np.mean(rouge_results),
            "rouge_l_max": np.max(rouge_results),
            "rouge_l_std": np.std(rouge_results),
            #"bertscore_precision": sum(bertscore_results["precision"]) / n,
            #"bertscore_recall": sum(bertscore_results["recall"]) / n,
            #"bertscore_f1": sum(bertscore_results["f1"]) / n,
            # sys/ref_len is the length (in space separated tokens, i.e. words) of predicted/label sentences
            # https://github.com/mjpost/sacrebleu/blob/5dfcaa3cee00039bcad7a12147b6d6976cb46f42/sacrebleu/metrics/bleu.py#L248
            "avg_predicted_len": np.mean(bleu_sys_len) / n,
            "avg_correct_len": np.mean(bleu_ref_len) / n,
        }
    
    def evaluate_t2d_example(
        self,
        predicted_entities: Set[str],
        predicted_relations: Set[Tuple[str]],
        label_entities: Set[str],
        label_relations: Set[Tuple[str]],
    ):
        # filter correct entities and relations
        correct_entities = predicted_entities & label_entities
        correct_relations = predicted_relations & label_relations

        # sanity check
        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(label_entities)
        assert len(correct_relations) <= len(predicted_relations)
        assert len(correct_relations) <= len(label_relations)

        res = Counter(
            {
                "num_sentences": 1,
                "gt_entities": len(label_entities),
                "predicted_entities": len(predicted_entities),
                "correct_entities": len(correct_entities),
                "gt_relations": len(label_relations),
                "predicted_relations": len(predicted_relations),
                "correct_relations": len(correct_relations),
            }
        )

        error_log = None
        if not len(correct_relations) == len(predicted_relations) == len(label_relations):
            error_log = "error\n" f"{predicted_relations}\n" f"{label_relations}\n"

        return res, error_log
    
    def evaluate_t2d_batch(self, batch, example_ids, dataset):
        batch_results = Counter()
        logs = ""
        format_data = batch["format_data"][0].item()
        kwargs = {}
        if len(dataset.id2format) > 1:
            kwargs['vae_latent_s'] = torch.nn.functional.one_hot(batch["format_data"], num_classes=len(dataset.id2format)).to(self.device)        # get raw batch predictions
        # shape: (N, max_seq_len), with max_seq_len depending on the batch (dynamically padded)
        data_prediction_ids = self.model.generate_with_target(
            batch["text_ids"].to(self.device),
            target="data",
            prefixes=self.prefixes,
            max_seq_length=dataset.max_seq_length,
            method="beam_search",
            num_beams=self.num_beams_t2d,
            **kwargs
        )
        batch_hypotheses = []
        batch_refs = []
        data_labels = [dataset.get_example(example_id).data for example_id in example_ids]

        for text_ids, data_label_sentence, data_pred_ids in zip(
            batch["text_ids"].to(self.device),
            data_labels,
            data_prediction_ids,
        ):
            # decode the token ids (of prediction, label and input)
            data_out_sentence = self.tokenizer.decode(
                data_pred_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text_in_sentence = self.tokenizer.decode(text_ids, skip_special_tokens=True)
            #data_label_sentence = self.tokenizer.decode(
            #    data_label_ids, skip_special_tokens=True
            #)

            # parse sentence with predicted data, obtain sets of predicted entities and relations
            if dataset.id2format[format_data] == 'knowledge_graph':
                parsed_data_pred = DataFormat.extract_raw_graph(data_out_sentence)
                parsed_data_label = DataFormat.extract_raw_graph(data_label_sentence)
            elif dataset.id2format[format_data] == 'tripleset':
                parsed_data_pred = DataFormat.extract_raw_tripleset(data_out_sentence)
                parsed_data_label = DataFormat.extract_raw_tripleset(data_label_sentence)
            elif dataset.id2format[format_data] == 'totto_table':
                parsed_data_pred = DataFormat.extract_raw_totto_table(data_out_sentence)
                parsed_data_label = DataFormat.extract_raw_totto_table(data_label_sentence)
            elif dataset.id2format[format_data] == 'mr':
                parsed_data_pred = DataFormat.extract_raw_mr(data_out_sentence)
                parsed_data_label = DataFormat.extract_raw_mr(data_label_sentence)
            else:
                raise ValueError
            predicted_entities, predicted_relations, wrong_format = parsed_data_pred
            label_entities, label_relations, parsed_label_format = parsed_data_label

            ## convert to graph structure for semBLEU metric computation
            pred_graph = DataFormat.convert_to_graph_structure(predicted_entities, predicted_relations)
            label_graph = DataFormat.convert_to_graph_structure(label_entities, label_relations)
            pred_n_grams = DataFormat.extract_n_grams_from_graph(pred_graph, 3)
            label_n_grams = DataFormat.extract_n_grams_from_graph(label_graph, 3)
            pred_n_grams_string = DataFormat.convert_n_grams_to_string(pred_graph, pred_n_grams)
            label_n_grams_string = DataFormat.convert_n_grams_to_string(label_graph, label_n_grams)
            batch_hypotheses.append(NgramInstance(ngram=pred_n_grams_string, length=len(pred_graph.nodes) + len(pred_graph.edges)))
            batch_refs.append(NgramInstance(ngram=label_n_grams_string, length=len(label_graph.nodes) + len(label_graph.edges)))

            # compare predictions to ground truth
            new_results, error_log = self.evaluate_t2d_example(
                predicted_entities, predicted_relations,
                label_entities, label_relations
            )
            if wrong_format:
                new_results["wrong_format"] = 1

            # log example
            logs += (
                f"[input / output / label (+pred/gt rel)\n"
                f"{text_in_sentence}\n"
                f"{data_out_sentence}\n"
                f"{data_label_sentence}\n"
            )
            if error_log:
                # predicted data did not match exactly target data
                new_results["data_errors"] += 1
                # log predicted vs ground truth sets of relations
                logs += error_log

            # update statistics: number of correct/pred/gt relations, etc
            batch_results += new_results

        return batch_results, batch_hypotheses, batch_refs, logs

    def compute_t2d_metrics(self, t2d_results):
        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=t2d_results["correct_entities"],
            num_predicted=t2d_results["predicted_entities"],
            num_gt=t2d_results["gt_entities"],
        )
        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=t2d_results["correct_relations"],
            num_predicted=t2d_results["predicted_relations"],
            num_gt=t2d_results["gt_relations"],
        )
        n = t2d_results["num_sentences"]
        metrics = {
            f"entity_f1": entity_f1,
            f"entity_precision": entity_precision,
            f"entity_recall": entity_recall,
            f"relation_f1": relation_f1,
            f"relation_precision": relation_precision,
            f"relation_recall": relation_recall,
            # % errors when parsing model output
            f"format_error": t2d_results["wrong_format"] / n,
            f"data_acc": (n - t2d_results["data_errors"]) / n,
        }
        return metrics

    def run_evaluation_t2d(self, dataset, split: str):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # if using shuffle and dataset.get_example(id), make sure that id is correct
            collate_fn=default_data_collator,
        )

        t2d_results = Counter()
        logs = ""
        hypotheses = []
        refs = []
        num_samples = 0
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.limit_samples and (i + 1) * self.batch_size > self.limit_samples:
                # to speed up validation (for a quick test), do not consider all samples
                break
            example_ids = range(num_samples, num_samples + len(batch["text_ids"]))
            batch_results, batch_hyp, batch_refs, batch_logs = self.evaluate_t2d_batch(batch, example_ids, dataset)
            t2d_results += batch_results
            logs += batch_logs
            hypotheses += batch_hyp
            refs += batch_refs
            num_samples += len(batch["text_ids"])
        
        references = [[r] for r in refs]
        metrics = self.compute_t2d_metrics(t2d_results)
        # compute semBLEU using official adapted script
        metrics['semBLEU'] = corpus_bleu(references, hypotheses)
        metrics = {f"{split}/{k}": v for k, v in metrics.items()}

        return metrics, logs


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate models in d2t and t2d')
    parser.add_argument('--name',
                        help='Name of the model')
    parser.add_argument('--data_path',
                        help='Path to the test datasets')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    model_name, data_path, batch_size = args.name, args.data_path, args.batch_size
    
