import random
import re
import nltk
import contractions
from nltk.corpus import stopwords
import numpy as np

from d2t.data.formatting import DataFormat, Triple, Entity, RelationType

blank_symbol = DataFormat.BLANK_TOKEN

SOURCE_FORMATS = ['knowledge_graph', 'tripleset', 'mr', 'table']

nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))
stop_words.remove("is")
stop_words.remove("this")
stop_words.remove("that")


def swap_words(seq, K=3):
    tokens = seq.split()

    noise = np.arange(len(tokens)) + np.random.uniform(0, K, size=(len(tokens),))
    permutation = noise.argsort()
    shuffled_tokens = np.array(tokens)[permutation]

    return " ".join(shuffled_tokens)


def swap_facts(seq, source_format):
    #   -> also swap e1 and e2?
    if source_format == 'knowledge_graph':
        facts = DataFormat.split_triples(seq)
        random.shuffle(facts)
        return ' '.join(facts)
    elif source_format == 'tripleset':
        facts = DataFormat.split_triplesets(seq)
        random.shuffle(facts)
        return ' | '.join(facts)
    elif source_format == 'mr':
        facts = DataFormat.split_mr(seq)
        noisy_seq = facts[0]
        facts = facts[1:]
        random.shuffle(facts)
        return noisy_seq + ', ' + ', '.join(facts)
    elif source_format == 'totto_table':
        facts = [f.strip() for f in re.split('<cell>|<table>|</table>', seq) if f != '' and f != ' ']
        for i in range(len(facts)):
            if i > 0:
                facts[i] = '<cell> ' + facts[i]
        noisy_seq = facts[0]
        facts = facts[1:]
        random.shuffle(facts)
        return noisy_seq + ' <table> ' + ' '.join(facts) + ' </table>'
    else:
        return seq

def swap(seq, source_format, **kwargs):
    if source_format == 'text':
        return swap_words(seq), 'text'
    else:
        return swap_facts(seq, source_format), source_format


def drop_fact(seq, source_format, drop_prob=0.1):
    if source_format == 'knowledge_graph':
        facts = DataFormat.split_triples(seq)
        kept_facts = [f for f in facts if random.random() > drop_prob]
        # we need to keep at least 1 fact
        if kept_facts:
            # return fact_delim.join(kept_facts)
            return " ".join(kept_facts)
        else:
            return facts[0]
    elif source_format == 'tripleset':
        facts = DataFormat.split_triplesets(seq)
        kept_facts = [f for f in facts if random.random() > drop_prob]
        # we need to keep at least 1 fact
        if kept_facts:
            # return fact_delim.join(kept_facts)
            return " | ".join(kept_facts)
        else:
            return facts[0]
    elif source_format == 'mr':
        facts = DataFormat.split_mr(seq)
        noisy_seq = facts[0]
        facts = facts[1:]
        kept_facts = [f for f in facts if random.random() > drop_prob]
        if kept_facts:
            return noisy_seq + ', ' + ', '.join(kept_facts)
        else:
            return noisy_seq
    elif source_format == 'totto_table':
        facts = [f.strip() for f in re.split('<cell>|<table>|</table>', seq) if f != '' and f != ' ']
        for i in range(len(facts)):
            if i > 0:
                facts[i] = '<cell> ' + facts[i]
        noisy_seq = facts[0]
        facts = facts[1:]
        kept_facts = [f for f in facts if random.random() > drop_prob]
        if kept_facts:
            return noisy_seq + ' <table> ' + ' '.join(kept_facts) + ' </table>'
        return noisy_seq
    else:
        return seq


def drop_word(seq, drop_prob=0.1):
    tokens = seq.split()
    kept_tokens = [t for t in tokens if random.random() > drop_prob]

    # we need to keep at least 1 word
    if kept_tokens:
        return " ".join(kept_tokens)
    else:
        return " ".join(tokens[: 1 + random.randrange(len(tokens))])


def drop(seq, source_format, **kwargs):
    if source_format == 'text':
        return drop_word(seq), source_format
    else:
        return drop_fact(seq, source_format), source_format


def blank_word(seq, blank_prob=0.2):
    tokens = seq.split()
    blanked_tokens = [
        t if random.random() > blank_prob else blank_symbol for t in tokens
    ]
    return " ".join(blanked_tokens)


def blank_fact(seq, source_format, blank_prob=0.2):
    if source_format == 'knowledge_graph':
        facts = DataFormat.split_triples(seq)
        blanked_facts = [
            fact if random.random() > blank_prob else blank_symbol for fact in facts
        ]
        return " ".join(blanked_facts)
    elif source_format == 'tripleset':
        facts = DataFormat.split_triplesets(seq)
        blanked_facts = [
        fact if random.random() > blank_prob else blank_symbol for fact in facts
        ]
        return " | ".join(blanked_facts)
    elif source_format == 'mr':
        facts = DataFormat.split_mr(seq)
        noisy_seq = facts[0]
        facts = facts[1:]
        blanked_facts = [
            fact if random.random() > blank_prob else blank_symbol for fact in facts
        ]
        return noisy_seq + ', ' + ', '.join(blanked_facts)
    elif source_format == 'totto_table':
        facts = [f.strip() for f in re.split('<cell>|<table>|</table>', seq) if f != '' and f != ' ']
        for i in range(len(facts)):
            if i > 0:
                facts[i] = '<cell> ' + facts[i]
        noisy_seq = facts[0]
        facts = facts[1:]
        blanked_facts = [
            fact if random.random() > blank_prob else blank_symbol for fact in facts
        ]
        return noisy_seq + ' <table> ' + ' '.join(blanked_facts) + ' </table>'
    else:
        return seq


def blank(seq, source_format, **kwargs):
    if source_format == 'text':
        return blank_word(seq), source_format
    else:
        return blank_fact(seq, source_format), source_format


def data2text(seq, source_format):
    buf = []
    if source_format == 'totto_table':
        facts = DataFormat.split_totto_tables(seq)
        return " and ".join(facts)
    else:
        if source_format == "knowledge_graph":
            _, facts, _ = DataFormat.extract_raw_graph(seq)
        elif source_format == "tripleset":
            _, facts, _ = DataFormat.extract_raw_tripleset(seq)
        elif source_format == "mr":
            _, facts, _ = DataFormat.extract_raw_mr(seq)
        else:
            return seq

        for e1, rel, e2 in facts:
            buf.append(f"{e1} {rel} {e2}")
        return " and ".join(buf)


def text2data(seq, target_format=None, tokenizer=None):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    stop_words.remove("is")
    stop_words.remove("this")
    stop_words.remove("that")
    fixed_seq = contractions.fix(seq)
    tokens = nltk.word_tokenize(fixed_seq)
    stripped = [t for t in tokens if t != ',']
    tagged_tokens = nltk.pos_tag(stripped)
    tagged_content_tokens = [e for e in tagged_tokens if e[0] not in stop_words]
    verb_positions = [
        i
        for i, e in enumerate(tagged_content_tokens)
        if e[1].startswith("V") and i >= 1
    ]
    adj_positions = [
        i for i, e in enumerate(tagged_content_tokens) if e[1].startswith("JJ")
    ]

    if target_format is None:
        # Random pick a format to be converted to
        target_format = random.choice(SOURCE_FORMATS)

    facts = []
    if target_format == 'mr':
        # instantiate a dictionary keeping track of subjects and its attributes
        facts = {}

    if target_format == 'totto_table':
        facts = []
        lin_seq = "<page_title> </page_title> <section_title> </section_title> "
        if len(verb_positions) > 0 or len(adj_positions) > 0:
            lin_seq += "<table> <col_header> Entities </col_header> "
        # Add entities
        for verb_pos in verb_positions:
            try:
                sbj = tagged_content_tokens[verb_pos - 1][0]
                facts.append(sbj)
                lin_seq += f"<col_header> {sbj} </col_header> "
            except IndexError:
                continue
        for adj_pos in adj_positions:
            try:
                sbj_candidates = [
                    e[0]
                    for i, e in enumerate(tagged_content_tokens)
                    if i > adj_pos and e[1].startswith("N")
                ]
                sbj = (
                    sbj_candidates[0]
                    if sbj_candidates
                    else tagged_content_tokens[adj_pos + 1][0]
                )
                facts.append(sbj)
                lin_seq += f"<col_header> {sbj} </col_header> "
            except IndexError:
                continue

    for verb_pos in verb_positions:
        pred = tagged_content_tokens[verb_pos][0]

        sbj = tagged_content_tokens[verb_pos - 1][0]
        try:
            obj = tagged_content_tokens[verb_pos + 1][0]
            if target_format == 'knowledge_graph':
                triplet_elements = [
                    DataFormat.HEAD_TOKEN,
                    sbj,
                    DataFormat.TYPE_TOKEN,
                    pred,
                    DataFormat.TAIL_TOKEN,
                    obj,
                ]
                facts.append(" ".join(triplet_elements))
            elif target_format == 'tripleset':
                facts.append(f"{sbj} : {pred} : {obj}")
            elif target_format == 'mr':
                if sbj not in facts.keys():
                    facts[sbj] = f"name[{sbj}], {pred}[{obj}]"
                else:
                    facts[sbj] += f", {pred}[{obj}]"
            elif target_format == 'totto_table':
                lin_seq += f"<cell> {obj} <col_header> {pred} </col_header> <row_header> {sbj} </row_header> </cell> "

        except IndexError:
            continue

    for adj_pos in adj_positions:
        pred = "has_attribute"
        obj = tagged_content_tokens[adj_pos][0]
        sbj_candidates = [
            e[0]
            for i, e in enumerate(tagged_content_tokens)
            if i > adj_pos and e[1].startswith("N")
        ]
        try:
            sbj = (
                sbj_candidates[0]
                if sbj_candidates
                else tagged_content_tokens[adj_pos + 1][0]
            )
            if target_format == 'knowledge_graph':
                triplet_elements = [
                    DataFormat.HEAD_TOKEN,
                    sbj,
                    DataFormat.TYPE_TOKEN,
                    pred,
                    DataFormat.TAIL_TOKEN,
                    obj,
                ]
                facts.append(" ".join(triplet_elements))
            elif target_format == 'tripleset':
                facts.append(f"{sbj} : {pred} : {obj}")
            elif target_format == 'mr':
                if sbj not in facts.keys():
                    facts[sbj] = f"name[{sbj}], {pred}[{obj}]"
                else:
                    facts[sbj] += f" | {pred} : {obj}"
            elif target_format == 'totto_table':
                lin_seq += f"<cell> {obj} <col_header> {pred} </col_header> <row_header> {sbj} </row_header> </cell> "
        except IndexError:
            continue

    if facts:
        if target_format == 'knowledge_graph':
            return " ".join(facts), 'knowledge_graph'
        elif target_format == 'tripleset':
            return " | ".join(facts), 'tripleset'
        elif target_format == 'mr':
            return ", ".join(facts.values()), 'mr'
        elif target_format == 'totto_table':
            lin_seq += "</table>"
            return lin_seq, "totto_table"
    else:
        # drop is fallback when rules don't find facts in text
        return blank_word(seq), "text"

def rule(seq, source_format, target_format=None):
    if source_format == 'text':
        data, target_format = text2data(seq, target_format=target_format)
        return data, target_format
    else:
        return data2text(seq, source_format), "text"


def repeat_fact(seq, source_format, repeat_prob=0.2):
    if source_format == 'knowledge_graph':
        facts = DataFormat.split_triples(seq)
        n = len(facts)
        for _ in range(n):
            if random.random() > repeat_prob:
                continue

            fact = random.choice(facts)
            pos = random.randint(0, len(facts))
            facts.insert(pos, fact)

        return " ".join(facts)
    elif source_format == 'tripleset':
        facts = DataFormat.split_triplesets(seq)
        n = len(facts)
        for _ in range(n):
            if random.random() > repeat_prob:
                continue

            fact = random.choice(facts)
            pos = random.randint(0, len(facts))
            facts.insert(pos, fact)

        return " | ".join(facts)
    elif source_format == 'mr':
        facts = DataFormat.split_mr(seq)
        noisy_seq = facts[0]
        facts = facts[1:]
        n = len(facts)
        for _ in range(n):
            if random.random() > repeat_prob:
                continue

            fact = random.choice(facts)
            pos = random.randint(0, len(facts))
            facts.insert(pos, fact)
        return noisy_seq + ', ' + ', '.join(facts)
    elif source_format == 'totto_table':
        facts = [f.strip() for f in re.split('<cell>|<table>|</table>', seq) if f != '' and f != ' ']
        for i in range(len(facts)):
            if i > 0:
                facts[i] = '<cell> ' + facts[i]
        noisy_seq = facts[0]
        facts = facts[1:]
        n = len(facts)
        for _ in range(n):
            if random.random() > repeat_prob:
                continue

            fact = random.choice(facts)
            pos = random.randint(0, len(facts))
            facts.insert(pos, fact)
        return noisy_seq + ' <table> ' + ' '.join(facts) + ' </table>'
    else:
        return seq


def repeat_text(seq, repeat_prob=0.1):
    tokens = seq.split()
    new_tokens = []
    for t in tokens:
        new_tokens.append(t)
        if random.random() <= repeat_prob:
            new_tokens.append(t)
    return " ".join(new_tokens)


def repeat(seq, source_format, **kwargs):
    if source_format == 'text':
        return repeat_text(seq), 'text'
    else:
        return repeat_fact(seq, source_format), source_format


existing_noise_functions = {
    "swap": swap,
    "drop": drop,
    "blank": blank,
    "repeat": repeat,
    "rule": rule
}
