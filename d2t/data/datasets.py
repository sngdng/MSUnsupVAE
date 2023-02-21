import json
import logging
import os
import unicodedata
import random
from copy import deepcopy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from glob import glob

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from d2t.data.formatting import (
    Triple,
    Entity,
    RelationType,
    Example,
    DataFormat,
)
from d2t.utils import (
    Mode,
    camel_case_to_natural_text,
    filter_string,
    parse_meaning_representation
)


def get_dataset_class(dataset_name):
    if dataset_name == 'multiformat':
        return MultiFormatDataset
    elif dataset_name == 'multiformat_dart':
        return MultiFormatDataset_Dart
    elif dataset_name == 'multiformat_totto':
        return MultiFormatDataset_Totto
    elif dataset_name == 'multiformat_webnlg':
        return MultiFormatDataset_WebNLG
    elif dataset_name == 'multiformat_e2e':
        return MultiFormatDataset_E2E
    else:
        raise ValueError


def upsample(data: list, weight: float):
    # Upsample lower size dataset to match the temperature mixing used in multitask
    n_data = len(data)
    assert weight >= 1

    integral = list(range(n_data)) * int(np.floor(weight))
    residual = list(range(n_data))
    random.shuffle(residual)
    residual = residual[:int(n_data * (weight - int(np.floor(weight))))]
    return [deepcopy(data[idx]) for idx in integral + residual]


def unsup_shuffle(features: List[dict]):
    shuffled_features = []
    n = len(features)
    text_idx = np.random.permutation(n)
    data_idx = np.random.permutation(n)

    for (text_id, data_id) in zip(text_idx, data_idx):
        # keep data example_id
        shuffled_features.append(
            {
                "example_id": features[data_id]["example_id"],
                "text_ids": features[text_id]["text_ids"],
                "att_mask_text": features[text_id]["att_mask_text"],
                "data_ids": features[data_id]["data_ids"],
                "att_mask_data": features[data_id]["att_mask_data"],
                "format_data": features[data_id]["format_data"]
            }
        )
    return shuffled_features


class Seq2seqDataset(Dataset, ABC):
    splits: List[str]
    max_seq_length = 256
    dataset_name: str  # name of the folder after processing

    def __init__(
        self,
        data_dir: Path,
        split: str,
        tokenizer: PreTrainedTokenizer = None,
        accelerator: Accelerator = None,
    ):
        """
        Base class for both data-to-text and text-to-data tasks, in a seq2seq setting

        Args:
            data_dir:
            name: name of the dataset to save the cached files
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        is_main_process = accelerator is None or accelerator.is_main_process

        if is_main_process and not os.path.isfile(
            data_dir / f"processed/{self.dataset_name}/train.pth"
        ):
            # if not already done, preprocess raw data and save it to disk, for training and eval
            self.build_dataset()
        if accelerator:
            # wait for the main process to build the dataset
            accelerator.wait_for_everyone()

        self.examples, self.features, self.unique_data_ids = torch.load(
            data_dir / f"processed/{self.dataset_name}/{split}.pth"
        )

    def build_dataset(self):
        """
        Load raw data, process it (build the list of examples and of features), and save it to the disk.
        This is called at init, if the cached files are not found.
        """

        logging.info(
            f"[{self.dataset_name}] Processed data not found. "
            f"Loading and processing raw data..."
        )
        os.makedirs(self.data_dir / f"processed/{self.dataset_name}", exist_ok=True)

        for split in self.splits:
            # load raw data of the split
            dataset = self.load_raw_dataset(split)

            examples, unique_data_ids = self.construct_examples(dataset, split)
            features = self.compute_features(examples)
            torch.save(
                (examples, features, unique_data_ids),
                self.data_dir / f"processed/{self.dataset_name}/{split}.pth",
            )

            # for evaluation, compute ref text files
            if "train" not in split:
                self.build_references(dataset, split=split)

    @abstractmethod
    def load_raw_dataset(self, split: str):
        """
        Load raw data, either from disk, or from huggingface datasets library.

        Returns
            An object that will be passed to construct_examples()
        """
        pass

    @abstractmethod
    def construct_examples(self, raw_dataset, split: str) -> List[Example]:
        """
        Construct the list of Examples from the raw data
        """
        pass

    def compute_features(self, examples: List[Example]):
        """

        Args:
            examples:

        Returns:
            features: list of (input_ids, att_mask, label_ids) to be used by the seq2seq model

        Examples
            input_ids: tokenized version of
                'text to data: Abilene , Texas is served by the Abilene Regional Airport .'
            label_ids: tokenized version of
                '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

        """
        logging.info("Computing features (format and tokenize data/text sequences)...")
        # format text and data into sequences
        text_sentences = []
        data_sentences = []
        for example in tqdm(examples):
            text_sentences.append(example.text)
            data_sentences.append(DataFormat.serialize_graph(example.data))

        text_tok = self.tokenizer(
            text_sentences,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        self._warn_max_sequence_length(self.max_seq_length, text_sentences, "input")

        data_tok = self.tokenizer(
            data_sentences,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        self._warn_max_sequence_length(self.max_seq_length, data_sentences, "output")

        assert (
            text_tok.input_ids.size(0) == data_tok.input_ids.size(0) == len(examples)
        )
        # todo: have a better look at data_ids -> why is there no space before the TAIL/TYPE tokens?
        features = []
        for i, (text_ids, att_mask_text, data_ids, att_mask_data, example) in enumerate(
            zip(
                text_tok.input_ids,
                text_tok.attention_mask,
                data_tok.input_ids,
                data_tok.attention_mask,
                examples
            )
        ):
            features.append(
                {
                    "example_id": torch.as_tensor(i),
                    "text_ids": torch.as_tensor(text_ids.tolist()),
                    "att_mask_text": torch.as_tensor(att_mask_text.tolist()),
                    "data_ids": torch.as_tensor(data_ids.tolist()),
                    "att_mask_data": torch.as_tensor(att_mask_data.tolist()),
                    "format_data": torch.as_tensor(example.format_data)
                }
            )

        return features

    def build_references(self, dataset, split: str):
        """
        Construct and save files with reference texts, for d2t evaluation.
        For each split, we make as many files as the max number of lexicalizations
        """
        pass

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def get_example(self, index):
        return self.examples[index]

    def _warn_max_sequence_length(
        self, max_sequence_length: int, sentences: List[str], name: str
    ):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f"Max sequence length is {max_sequence_length} but the longest {name} sequence is "
                f"{max_length_needed} long"
            )


class MultiFormatDataset(Seq2seqDataset):
    """
    A hand-crafted dataset which consists of Dart, WebNLG, the cleaned version of E2E
    augmented by a sample of Genwiki and ToTTo datasets
    """
    splits = [
        "train",
        "validation_dart",
        "validation_web_nlg",
        "validation_e2e",
        "validation_totto",
        "test_dart",
        "test_web_nlg",
        "test_e2e",
        "test_totto",
    ]
    id2format = {
        0: "knowledge_graph",
        1: "tripleset",
        2: "mr",
        3: "totto_table",
    }
    dataset_name = "multiformat"

    def __init__(
        self,
        data_dir: Path,
        split: str,
        mode: Mode = Mode('both_sup'),
        tokenizer: PreTrainedTokenizer = None,
        accelerator: Accelerator = None,
        temp_mixing: float = 2
    ):
        """
        Base class for both data-to-text and text-to-data tasks, in a seq2seq setting

        Args:
            data_dir:
            name: name of the dataset to save the cached files
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        is_main_process = accelerator is None or accelerator.is_main_process

        if is_main_process and not os.path.isfile(
            data_dir / f"processed/{self.dataset_name}/train.pth"
        ):
            # if not already done, preprocess raw data and save it to disk, for training and eval
            self.build_dataset()
        if accelerator:
            # wait for the main process to build the dataset
            accelerator.wait_for_everyone()

        self.examples, self.features, self.unique_data_ids = torch.load(
            data_dir / f"processed/{self.dataset_name}/{split}.pth"
        )

        if split == 'train':
            self._upsample(temp_mixing)
            # if unsup -> shuffle each dataset to make them non-parallel
            if mode == Mode('both_unsup') or mode == Mode('both_unsup_mf'):
                self.features = {k: unsup_shuffle(v) for k, v in self.features.items()}
            concat_features = []
            # Concatenate the dataset
            for l in self.features.values():
                concat_features += l
            self.features = concat_features
    
    def _upsample(self, temp_mixing: float):
        # Dataset statistics.
        dataset2size = {}
        for dataset_name, data in self.features.items():
            dataset2size[dataset_name] = len(data)

        # Compute resampling weights.
        upsample_weights = {}
        sum_tau_size = sum([np.exp(np.log(size) / temp_mixing) for size in dataset2size.values()])
        sum_size = sum(dataset2size.values())
        for dataset_name, size in dataset2size.items():
            tau_size = np.exp(np.log(size) / temp_mixing)
            upsample_weights[dataset_name] = tau_size / sum_tau_size * sum_size / size

        # Compute upsampling weights.
        largest_dataset, _ = max(dataset2size.items(), key=lambda x: x[1])
        norm_coef = upsample_weights[largest_dataset]
        for dataset_name in upsample_weights.keys():
            upsample_weights[dataset_name] = upsample_weights[dataset_name] / norm_coef
        
        # Upsample.
        for dataset_name in sorted(self.features.keys()):
            self.features[dataset_name] = upsample(self.features[dataset_name], upsample_weights[dataset_name])
        print('Before upsampling:', dataset2size)
        print('Upsampling weights:', upsample_weights)
        print('After upsampling:', {dataset_name: len(data) for dataset_name, data in self.features.items()})

    def load_raw_dataset(self):
        # DART dataset
        dart_dataset = load_dataset(
            "GEM/dart",
        )
        dart_dataset['name'] = 'dart'
        # Cleaned E2E dataset
        cleaned_e2e_dataset = load_dataset(
            "GEM/e2e_nlg",
        )
        cleaned_e2e_dataset['name'] = 'e2e'
        # WebNLG dataset
        web_nlg_dataset = load_dataset(
            "GEM/web_nlg",
            name='en'
        )
        web_nlg_dataset['name'] = 'web_nlg'
        # ToTTo dataset
        totto_dataset = load_dataset(
            "GEM/totto",
        )
        totto_dataset['name'] = 'totto'
        totto_dataset['test'] = totto_dataset['validation']

        full_dataset = [dart_dataset,
                        cleaned_e2e_dataset,
                        web_nlg_dataset,
                        totto_dataset
                        ]

        return full_dataset
    
    def build_dataset(self):
        logging.info(
            f"[{self.dataset_name}] Processed data not found. "
            f"Loading and processing raw data..."
        )
        os.makedirs(self.data_dir / f"processed/{self.dataset_name}", exist_ok=True)

        raw_full_dataset = self.load_raw_dataset()

        for split in ['train', 'validation', 'test']:
            if split == 'train':
                dict_examples, dict_unique_data_ids, dict_features = {}, {}, {}
                for raw_dataset in raw_full_dataset:
                    # load raw data of the split
                    examples, unique_data_ids = self.construct_examples(raw_dataset, split)
                    features = self.compute_features(examples)
                    dict_examples[raw_dataset['name']] = examples
                    dict_unique_data_ids[raw_dataset['name']] = unique_data_ids
                    dict_features[raw_dataset['name']] = features
                    logging.info(f"[TRAIN] There are {len(examples)} examples from {raw_dataset['name']}")

                torch.save(
                    (dict_examples, dict_features, dict_unique_data_ids),
                    self.data_dir / f"processed/{self.dataset_name}/{split}.pth",
                )
            else:
                for raw_dataset in raw_full_dataset:
                    list_examples, list_unique_data_ids, list_features = [], [], []
                    # load raw data of the split
                    examples, unique_data_ids = self.construct_examples(raw_dataset, split)
                    features = self.compute_features(examples)
                    list_examples += examples
                    list_unique_data_ids += unique_data_ids
                    list_features += features
                    torch.save(
                        (list_examples, list_features, list_unique_data_ids),
                        self.data_dir / f"processed/{self.dataset_name}/{split}_{raw_dataset['name']}.pth",
                    )

    def construct_examples(self, raw_dataset, split: str):
        logging.info(f"[{split}] Constructing examples for {raw_dataset['name']}")

        examples = []
        unique_data_ids = []

        for entry in tqdm(raw_dataset[split]):
            refs = None
            if split.startswith('test') or split.startswith('validation'):
                refs = entry['references']

            if raw_dataset["name"] == 'web_nlg':
                # graph y
                graph = []
                raw_triples = entry['input']
                for raw_triple in raw_triples:
                    e1, rel, e2 = raw_triple.split(' | ')
                    e1 = e1.replace("_", " ")
                    e2 = e2.replace("_", " ")
                    rel_natural = camel_case_to_natural_text(rel)
                    graph += [
                        Triple(Entity(e1), RelationType(rel, rel_natural), Entity(e2))
                    ]
                data = DataFormat.serialize_graph(graph) # Linearize the processed triples
                # id of the first example with this graph
                unique_data_ids.append(entry['gem_id'])
                text = entry['target']
                examples.append(Example(text=text, data=data, format_data=0, references=refs))

            elif raw_dataset["name"] == 'dart':
                tripleset = []
                raw_triples = entry['tripleset']
                data = ''
                for i, raw_triple in enumerate(raw_triples):
                    e1, rel, e2 = raw_triple
                    e1 = e1.replace("_", " ")
                    e2 = e2.replace("_", " ")
                    rel_natural = rel.lower()
                    if i > 0:
                        data += ' | '
                    data += f'{e1} : {rel_natural} : {e2}'
                unique_data_ids.append(entry['gem_id'])
                text = entry['target']
                examples.append(Example(text=text, data=data, format_data=1, references=refs))
            
            elif raw_dataset["name"] == 'totto':
                data = entry['linearized_input']
                unique_data_ids.append(entry['gem_id'])
                text = entry['target']
                examples.append(Example(text=text, data=data, format_data=3, references=refs))

            elif raw_dataset["name"] == 'e2e':
                data = entry['meaning_representation']
                unique_data_ids.append(entry['gem_id'])
                text = entry['target']
                examples.append(Example(text=text, data=data, format_data=2, references=refs))

        logging.info(f"[{split}] unique data: {len(unique_data_ids)}")
        return examples, unique_data_ids
    
    def compute_features(self, examples: List[Example]):
        """

        Args:
            examples:

        Returns:
            features: list of (input_ids, att_mask, label_ids) to be used by the seq2seq model

        Examples
            input_ids: tokenized version of
                'text to graph: Abilene , Texas is served by the Abilene Regional Airport .'
            label_ids: tokenized version of
                '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

        """
        logging.info("Computing features (format and tokenize graph/text sequences)...")
        # format text and data into sequences
        text_sentences = []
        data_sentences = []
        for example in tqdm(examples):
            text_sentences.append(example.text)
            data_sentences.append(example.data)

        text_tok = self.tokenizer(
            text_sentences,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        self._warn_max_sequence_length(self.max_seq_length, text_sentences, "input")

        data_tok = self.tokenizer(
            data_sentences,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        self._warn_max_sequence_length(self.max_seq_length, data_sentences, "output")

        assert (
            text_tok.input_ids.size(0) == data_tok.input_ids.size(0) == len(examples)
        )
        # todo: have a better look at data_ids -> why is there no space before the TAIL/TYPE tokens?
        features = []
        for i, (text_ids, att_mask_text, data_ids, att_mask_data, example) in enumerate(
            zip(
                text_tok.input_ids,
                text_tok.attention_mask,
                data_tok.input_ids,
                data_tok.attention_mask,
                examples
            )
        ):
            features.append(
                {
                    "example_id": torch.as_tensor(i),
                    "text_ids": torch.as_tensor(text_ids.tolist()),
                    "att_mask_text": torch.as_tensor(att_mask_text.tolist()),
                    "data_ids": torch.as_tensor(data_ids.tolist()),
                    "att_mask_data": torch.as_tensor(att_mask_data.tolist()),
                    "format_data": torch.as_tensor(example.format_data)
                }
            )

        return features


class MultiFormatDataset_Dart(MultiFormatDataset):
    """
    A hand-crafted dataset which consists of Dart, WebNLG, the cleaned version of E2E
    augmented by a sample of Genwiki and ToTTo datasets
    """
    splits = [
        "train",
        "test_dart",
    ]
    id2format = {
        1: "tripleset"
    }
    dataset_name = "multiformat"

    def __init__(
        self,
        data_dir: Path,
        split: str,
        mode: Mode = Mode('both_sup'),
        tokenizer: PreTrainedTokenizer = None,
        accelerator: Accelerator = None,
        temp_mixing: float = 2
    ):
        """
        Base class for both data-to-text and text-to-data tasks, in a seq2seq setting

        Args:
            data_dir:
            name: name of the dataset to save the cached files
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        is_main_process = accelerator is None or accelerator.is_main_process

        if is_main_process and not os.path.isfile(
            data_dir / f"processed/{self.dataset_name}/train.pth"
        ):
            # if not already done, preprocess raw data and save it to disk, for training and eval
            self.build_dataset()
        if accelerator:
            # wait for the main process to build the dataset
            accelerator.wait_for_everyone()

        self.examples, self.features, self.unique_data_ids = torch.load(
            data_dir / f"processed/{self.dataset_name}/{split}.pth"
        )

        if split == 'train':
            features = deepcopy(self.features['dart'])
            del self.features
            # if unsup -> shuffle each dataset to make them non-parallel
            if mode == Mode('both_unsup'):
                self.features = unsup_shuffle(features)
            else:
                self.features = features


class MultiFormatDataset_Totto(MultiFormatDataset):
    """
    A hand-crafted dataset which consists of Dart, WebNLG, the cleaned version of E2E
    augmented by a sample of Genwiki and ToTTo datasets
    """
    splits = [
        "train",
        "test_totto",
    ]
    id2format = {
        3: "totto_table"
    }
    dataset_name = "multiformat_totto"

    def __init__(
        self,
        data_dir: Path,
        split: str,
        mode: Mode = Mode('both_sup'),
        tokenizer: PreTrainedTokenizer = None,
        accelerator: Accelerator = None,
        temp_mixing: float = 2
    ):
        """
        Base class for both data-to-text and text-to-data tasks, in a seq2seq setting

        Args:
            data_dir:
            name: name of the dataset to save the cached files
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        is_main_process = accelerator is None or accelerator.is_main_process

        if is_main_process and not os.path.isfile(
            data_dir / f"processed/{self.dataset_name}/train.pth"
        ):
            # if not already done, preprocess raw data and save it to disk, for training and eval
            self.build_dataset()
        if accelerator:
            # wait for the main process to build the dataset
            accelerator.wait_for_everyone()

        self.examples, self.features, self.unique_data_ids = torch.load(
            data_dir / f"processed/{self.dataset_name}/{split}.pth"
        )

        if split == 'train':
            # if unsup -> shuffle each dataset to make them non-parallel
            if mode == Mode('both_unsup'):
                self.features = unsup_shuffle(self.features['totto'])
            else:
                self.features = self.features['totto']


class MultiFormatDataset_WebNLG(MultiFormatDataset):
    """
    A hand-crafted dataset which consists of Dart, WebNLG, the cleaned version of E2E
    augmented by a sample of Genwiki and ToTTo datasets
    """
    splits = [
        "train",
        "test_web_nlg",
    ]
    id2format = {
        0: "knowledge_graph"
    }
    dataset_name = "multiformat"

    def __init__(
        self,
        data_dir: Path,
        split: str,
        mode: Mode = Mode('both_sup'),
        tokenizer: PreTrainedTokenizer = None,
        accelerator: Accelerator = None,
        temp_mixing: float = 2
    ):
        """
        Base class for both data-to-text and text-to-data tasks, in a seq2seq setting

        Args:
            data_dir:
            name: name of the dataset to save the cached files
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        is_main_process = accelerator is None or accelerator.is_main_process

        if is_main_process and not os.path.isfile(
            data_dir / f"processed/{self.dataset_name}/train.pth"
        ):
            # if not already done, preprocess raw data and save it to disk, for training and eval
            self.build_dataset()
        if accelerator:
            # wait for the main process to build the dataset
            accelerator.wait_for_everyone()

        self.examples, self.features, self.unique_data_ids = torch.load(
            data_dir / f"processed/{self.dataset_name}/{split}.pth"
        )

        if split == 'train':
            # if unsup -> shuffle each dataset to make them non-parallel
            if mode == Mode('both_unsup'):
                self.features = unsup_shuffle(self.features['web_nlg'])
            else:
                self.features = self.features['web_nlg']


class MultiFormatDataset_E2E(MultiFormatDataset):
    """
    A hand-crafted dataset which consists of Dart, WebNLG, the cleaned version of E2E
    augmented by a sample of Genwiki and ToTTo datasets
    """
    splits = [
        "train",
        "test_e2e",
    ]
    id2format = {
        2: "mr"
    }
    dataset_name = "multiformat"

    def __init__(
        self,
        data_dir: Path,
        split: str,
        mode: Mode = Mode('both_sup'),
        tokenizer: PreTrainedTokenizer = None,
        accelerator: Accelerator = None,
        temp_mixing: float = 2
    ):
        """
        Base class for both data-to-text and text-to-data tasks, in a seq2seq setting

        Args:
            data_dir:
            name: name of the dataset to save the cached files
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        is_main_process = accelerator is None or accelerator.is_main_process

        if is_main_process and not os.path.isfile(
            data_dir / f"processed/{self.dataset_name}/train.pth"
        ):
            # if not already done, preprocess raw data and save it to disk, for training and eval
            self.build_dataset()
        if accelerator:
            # wait for the main process to build the dataset
            accelerator.wait_for_everyone()

        self.examples, self.features, self.unique_data_ids = torch.load(
            data_dir / f"processed/{self.dataset_name}/{split}.pth"
        )

        if split == 'train':
            # if unsup -> shuffle each dataset to make them non-parallel
            if mode == Mode('both_unsup'):
                self.features = unsup_shuffle(self.features['amr'])
            else:
                self.features = self.features['amr']