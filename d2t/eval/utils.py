import os
import codecs
import logging
import subprocess

from pathlib import Path

metrics_dir = Path(__file__).resolve().parent
METEOR_PATH = str(metrics_dir / "metrics/meteor-1.5/meteor-1.5.jar")


def get_precision_recall_f1(num_correct: int, num_predicted: int, num_gt: int):
    """
    For t2g evaluation
    """
    assert 0 <= num_correct <= num_predicted
    assert 0 <= num_correct <= num_gt

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_gt if num_gt > 0 else 0.0
    f1 = 2.0 / (1.0 / precision + 1.0 / recall) if num_correct > 0 else 0.0

    return precision, recall, f1


def compute_meteor_score(references, hypothesis, num_refs, lng="en"):
    """
    Compute METEOR score using the original Java Meteor-1.5 file
    """
    logging.info("STARTING TO COMPUTE METEOR...")
    hyps_tmp, refs_tmp = "hypothesis_meteor", "reference_meteor"

    with codecs.open(hyps_tmp, "w", "utf-8") as f:
        f.write("\n".join(hypothesis))

    linear_references = []
    for refs in references:
        for i in range(num_refs):
            linear_references.append(refs[i])

    with codecs.open(refs_tmp, "w", "utf-8") as f:
        f.write("\n".join(linear_references))

    try:
        command = "java -Xmx2G -jar {0} ".format(METEOR_PATH)
        command += "{0} {1} -l {2} -norm -r {3}".format(
            hyps_tmp, refs_tmp, lng, num_refs
        )
        result = subprocess.check_output(command, shell=True)
        meteor = result.split(b"\n")[-2].split()[-1]
    except:
        logging.error(
            "ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE."
        )
        meteor = -1

    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info("FINISHING TO COMPUTE METEOR...")
    return float(meteor)


def compute_bleurt(
    references, hypothesis, num_refs, checkpoint="metrics/bleurt-base-128"
):
    from bleurt import score

    refs, cands = [], []
    for i, hyp in enumerate(hypothesis):
        for ref in references[i][:num_refs]:
            cands.append(hyp)
            refs.append(ref)

    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=refs, candidates=cands)
    scores = [max(scores[i : i + num_refs]) for i in range(0, len(scores), num_refs)]
    return round(sum(scores) / len(scores), 2)
