from typing import List, Tuple, Union, Dict

import networkx as nx
import re
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

GENERATE_TEXT_TOKEN = "[GENERATE_TEXT]"
GENERATE_DATA_TOKEN = "[GENERATE_DATA]"

STYLE_TOKEN = "[STYLE]"


@dataclass
class Entity:
    text: str


@dataclass
class RelationType:
    short: str
    natural: str  # string to use in input/output sequences, more readable


@dataclass
class Triple:
    head: Entity
    rel: RelationType
    tail: Entity

    def __repr__(self):
        return f"({self.head.text} -> {self.rel.short} -> {self.tail.text})"

    def to_tuple(self):
        return self.head.text, self.rel.natural, self.tail.text


@dataclass
class Example:
    text: str  # plain text sentence
    data: str # data in plain text
    format_data: int = 0 # format of data
    references: list = None


class DataFormat:
    HEAD_TOKEN = "[HEAD]"
    TYPE_TOKEN = "[TYPE]"
    TAIL_TOKEN = "[TAIL]"
    BLANK_TOKEN = "[BLANK]"
    DART_TOKENS = ["[TITLE]", "[TABLECONTEXT]"]
    TOTTO_TOKENS = ["<page_title>", "</page_title>",
                    "<section_title>", "</section_title>",
                    "<table>", "</table>",
                    "<cell>", "</cell>",
                    "<col_header>", "</col_header>",
                    "<row_header>", "</row_header>"]

    @staticmethod
    def serialize_graph(graph: List[Triple]) -> str:
        """
        Format graph (list of relations) into a string

        Examples
            for a graph with only one relation:
            '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

        """
        seralized_graphs = []
        for triple in graph:
            if triple == DataFormat.BLANK_TOKEN:
                # make it possible to replace entire triplets with a [BLANK] token
                seralized_graphs.append(triple)
            else:
                seralized_graphs.append(
                    " ".join(
                        [
                            DataFormat.HEAD_TOKEN,
                            triple.head.text,
                            DataFormat.TYPE_TOKEN,
                            triple.rel.natural,
                            DataFormat.TAIL_TOKEN,
                            triple.tail.text,
                        ]
                    )
                )
        return " ".join(seralized_graphs)
    
    @staticmethod
    def convert_to_graph_structure(entities: set, relations: set) -> nx.Graph:
        graph = nx.DiGraph()
        graph.add_nodes_from(entities)
        for rel in relations:
            graph.add_edge(rel[0], rel[2], rel=rel[1])
        
        return graph

    @staticmethod
    def extract_n_grams_from_graph(graph: nx.Graph, n_gram: int):
        assert n_gram > 1
        n_grams_dict = {k: [] for k in range(1, n_gram + 1)}
        # get roots from graph
        roots = [n for n, d in graph.in_degree() if d == 0]
        # Nodes and edges
        n_grams_dict[1] = [node for node in graph.nodes]
        # perform bfs starting from each root to extract all n_grams
        for k in range(2, n_gram + 1):
            for root in roots:
                # setting max rec call = nb of nodes to handle cycles if there are
                k_grams_from_root = DataFormat.extract_n_grams_from_node(graph, root, k, 2*len(graph.nodes))
                # filter n_gram of longer length
                filtered_k_grams = [kg[:k] for kg in k_grams_from_root if len(kg) >= k]
                n_grams_dict[k] += filtered_k_grams
        return n_grams_dict
    
    @staticmethod
    def convert_n_grams_to_string(graph: nx.Graph, dict_n_grams: dict):
        n_grams_dict = {k: [] for k in range(1, max(dict_n_grams.keys()) + 1)}
        if dict_n_grams[2] == []:
            return dict_n_grams
        n_grams_dict[1] = dict_n_grams[1]
        for k in range(2, 4):
            for kgram in dict_n_grams[k]:
                string = kgram[0]
                for i in range(len(kgram)-1):
                    edge_label = graph.get_edge_data(kgram[i], kgram[i+1])
                    relation = edge_label['rel']
                    string += f' {relation} {kgram[i+1]}'
                n_grams_dict[k].append(string)

        return n_grams_dict

    @staticmethod
    def extract_n_grams_from_node(graph: nx.Graph, node: str, k_gram: int, max_rec: int):
        if list(graph.successors(node)) == [] or k_gram == 1 or max_rec == 0:
            return [[node]]
        else:
            list_k_grams = []
            for succ in graph.successors(node):
                # k_gram containing node
                l1 = DataFormat.extract_n_grams_from_node(graph, succ, k_gram-1, max_rec-1)
                # k_gram not containing node
                l2 = DataFormat.extract_n_grams_from_node(graph, succ, k_gram, max_rec-1)
                list_k_grams += [[node, *l] for l in l1 if l[0] in graph.successors(node)]
                list_k_grams += l2
                
            return list_k_grams

    @staticmethod
    def extract_raw_graph(output_sentence: str) -> Tuple[set, set, bool]:
        """
        Parse raw output sentence, extract entities and relations

        Returns:
            Raw set of entities (str) and triples (tuple of str), and whether
            there was a parsing error at some point

        Examples:
            output_sentence:
                '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

            predicted_entities:
                {"Abilene , Texas", "Abilene Regional Airport"}
            predicted_relations:
                {("Abilene , Texas", "city served", "Abilene Regional Airport"),}
            format_error:
                False

        """
        format_error = False
        predicted_entities = set()
        predicted_relations = set()

        # parse_output_sentence
        for relation in output_sentence.split(DataFormat.HEAD_TOKEN):
            if len(relation) == 0:
                # if '[HEAD]' is at the beginning of the sentence we can obtain an empty str
                continue

            # try splitting head from type and tail
            split_type = relation.split(DataFormat.TYPE_TOKEN)
            if len(split_type) != 2:
                format_error = True
                continue
            head, type_and_tail = split_type

            # try splitting type and tail
            split_tail = type_and_tail.split(DataFormat.TAIL_TOKEN)
            if len(split_tail) != 2:
                format_error = True
                continue
            type, tail = split_tail

            e1 = head.strip()
            rel = type.strip()
            e2 = tail.strip()
            predicted_entities.update([e1, e2])
            predicted_relations.add((e1, rel, e2))

        return predicted_entities, predicted_relations, format_error
    
    @staticmethod
    def extract_raw_tripleset(output_sentence: str) -> Tuple[set, set, bool]:
        """
        Parse raw output sentence, extract entities and relations

        Returns:
            Raw set of entities (str) and triples (tuple of str), and whether
            there was a parsing error at some point

        Examples:
            output_sentence:
                'Abilene , Texas : city served : Abilene Regional Airport'

            predicted_entities:
                {"Abilene , Texas", "Abilene Regional Airport"}
            predicted_relations:
                {("Abilene , Texas", "city served", "Abilene Regional Airport")}
            format_error:
                False

        """
        format_error = False
        predicted_entities = set()
        predicted_relations = set()

        # parse_output_sentence
        for fact in output_sentence.split(" | "):
            if len(fact) == 0:
                continue

            # try splitting ent1 rel ent2
            split_fact = fact.split(" : ")
            if len(split_fact) != 3:
                format_error = True
                continue
            ent1, rel, ent2 = split_fact

            e1 = ent1.strip()
            rel = rel.strip()
            e2 = ent2.strip()
            predicted_entities.update([e1, e2])
            predicted_relations.add((e1, rel, e2))

        return predicted_entities, predicted_relations, format_error

    @staticmethod
    def extract_raw_mr(output_sentence: str) -> Tuple[set, set, bool]:
        """
        Parse raw output sentence, extract entities and relations

        Returns:
            Raw set of entities (str) and triples (tuple of str), and whether
            there was a parsing error at some point

        Examples:
            output_sentence:
                'name[Abilene , Texas], city served[Abilene Regional Airport]'

            predicted_entities:
                {"Abilene , Texas", "Abilene Regional Airport"}
            predicted_relations:
                {("Abilene , Texas", "city served", "Abilene Regional Airport")}
            format_error:
                False

        """
        format_error = False
        predicted_entities = set()
        predicted_relations = set()

        list_facts = output_sentence.split(', ')
        entity1 = ''
        for i, fact in enumerate(list_facts):
            mrc_format = "(.*)\[(.*)\]"
            try:
                attr, value = re.compile(mrc_format).match(fact).groups()
                if i == 0:
                    if attr == 'name':
                        entity1 = value
                    else:
                        format_error = True
            except:
                format_error = True
                continue
            
            e1 = entity1.strip()
            rel = attr.strip()
            e2 = value.strip()
            predicted_entities.update([e1, e2])
            predicted_relations.add((e1, rel, e2))

        return predicted_entities, predicted_relations, format_error
    
    @staticmethod
    def extract_raw_totto_table(output_sentence: str) -> Tuple[set, set, bool]:
        """
        Parse raw output sentence, extract entities and relations

        Returns:
            Raw set of entities (str) and triples (tuple of str), and whether
            there was a parsing error at some point

        """
        format_error = False
        predicted_entities = set()
        predicted_relations = set()

        facts = []
        for fact in re.split('<page_title>|</page_title>|<section_title>|</section_title>|<table>|</table>|<cell>|</cell>', output_sentence):
            if fact != '' and fact != ' ':
                facts.append(fact.strip())

        for fact in facts:
            if '<row_header>' in fact:
                list_row = [e for e in re.split('<row_header>|</row_header>', fact) if e != '' and e != ' ']
                if list_row == []:
                    format_error = True
                    continue
                e2 = list_row[-1].strip()
                cell = [e for e in re.split('<col_header>|</col_header>', fact) if e != '' and e != ' ']
                if len(cell) < 2:
                    format_error = True
                    continue 
                rel = cell[1].strip()
                e1 = cell[0].strip()
                predicted_entities.update([e1, e2])
                predicted_relations.add((e1, rel, e2))
            elif '<col_header>' in fact:
                e1 = facts[0]
                cell = [e for e in re.split('<col_header>|</col_header>', fact) if e != '' and e != ' ']
                if len(cell) < 2:
                    format_error = True
                    continue
                rel = cell[1].strip()
                del(cell[1])
                for c in cell:
                    e2 = c.strip()
                    predicted_entities.update([e1, e2])
                    predicted_relations.add((e1, rel, e2))
            else:
                continue
        
        return predicted_entities, predicted_relations, format_error


    @staticmethod
    def split_totto_tables(output_sentence: str):
        """
        Parse raw output sentence, extract entities and relations

        Returns:
            Raw set of entities (str) and triples (tuple of str), and whether
            there was a parsing error at some point

        """
        FILLED_TOKENS = ["<page_title>",
                         "<section_title>",
                         "<table>",
                         "<cell>",
                         "<col_header>",
                         "<row_header>"]
        SEP_TOTTO_TOKENS = ["</page_title>",
                            "</section_title>",
                            "</table>",
                            "</cell>",
                            "</col_header>",
                            "</row_header>"]

        for token in FILLED_TOKENS:
            output_sentence = output_sentence.replace(token, '')
        for token in SEP_TOTTO_TOKENS:
            output_sentence = output_sentence.replace(token, '|')
        processed_sentence = output_sentence.strip()
        facts = []
        processed_facts = processed_sentence.split('|')
        for fact in processed_facts:
            fact = fact.strip()
            fact = re.sub('\\s+', ' ', fact)
            if fact != ' ' and fact != '':
                facts.append(fact)
        
        return facts

    @staticmethod
    def split_triples(sequence) -> List[str]:
        """
        Split an input sequence into a list of triples (still as strings)

        Examples
            >>> sequence = "[HEAD] 20 Fenchurch Street [TYPE] location [TAIL] London [HEAD] London [TYPE] leader title [TAIL] European Parliament"
            >>> DataFormat.split_triples(sequence)
            ["[HEAD] 20 Fenchurch Street [TYPE] location [TAIL] London",
             "[HEAD] London [TYPE] leader title [TAIL] European Parliament"]
        """
        triples = sequence.split(DataFormat.HEAD_TOKEN)
        # don't consider the empty string at the beginning, given by split()
        assert len(triples[0]) == 0
        triples = triples[1:]
        # add the HEAD token back at the beginning of the triple str
        triples = [f"{DataFormat.HEAD_TOKEN} {t.strip()}" for t in triples]
        return triples
    
    @staticmethod
    def split_triplesets(sequence) -> List[str]:
        """
        Split an input sequence into a list of triples (still as strings)

        Examples
            >>> sequence = "20 Fenchurch Street : location : London | London : leader title : European Parliament"
            >>> DataFormat.split_triplesets(sequence)
            ["20 Fenchurch Street : location : London",
             "London : leader title : European Parliament"]
        """
        triples = sequence.split(' | ')
        return triples

    @staticmethod
    def split_mr(sequence) -> List[str]:
        """
        Split an input sequence into a list of triples (still as strings)

        Examples
            >>> sequence = "name : The Eagle | eatType : coffee shop"
            >>> DataFormat.split_mr(sequence)
            ["name[The Eagle]",
             "eatType[coffee shop]"]
        """
        triples = sequence.split(', ')
        return triples


def construct_all_prefixes(tokenizer: PreTrainedTokenizer):
    prefixes = {}
    prefixes['text'] = tokenizer("Generate text: ", return_tensors='pt').input_ids[:,:-1]
    prefixes['data'] = tokenizer("Generate data: ", return_tensors='pt').input_ids[:,:-1]
    prefixes['knowledge_graph'] = tokenizer("Generate graph: ", return_tensors='pt').input_ids[:,:-1]
    prefixes['totto_table'] = tokenizer("Generate table: ", return_tensors='pt').input_ids[:,:-1]
    prefixes['tripleset'] = tokenizer("Generate triple: ", return_tensors='pt').input_ids[:,:-1]
    prefixes['mr'] = tokenizer("Generate meaning representation: ", return_tensors='pt').input_ids[:,:-1]

    return prefixes


def add_target_prefix(
    input_ids: torch.Tensor,
    target: Union[list, str],
    prefixes: Dict[str, torch.Tensor]
):
    # this avoids using tokenizer
    # this effectively adds a few tokens at the beginning of the sequence, but keeps a length
    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)

    if isinstance(target, str):
        prefix = prefixes[target]
        length_prefix = int(prefix.size(1))
        shifted_input_ids[..., length_prefix:] = input_ids[..., :-length_prefix].clone()
        shifted_input_ids[..., :length_prefix] = prefix
    elif isinstance(target, list):
        # loop over target list and append
        for i in range(len(target)):
            prefix = prefixes[target[i]]
            length_prefix = int(prefix.size(1))
            shifted_input_ids[i, length_prefix:] = input_ids[..., :-length_prefix].clone()
            shifted_input_ids[i, :length_prefix] = prefix
    else:
        raise ValueError

    return shifted_input_ids


def add_style_prefix(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """
    Take input_ids, shift it to the right and add the [STYLE] special token at the beginning.
    Inspired from T5 `_shift_right` method
    """
    style_token_id = tokenizer.convert_tokens_to_ids(STYLE_TOKEN)
    # make sure STYLE_TOKEN was correctly converted to id (e.g. verify it's known '<unk>')
    assert tokenizer.convert_ids_to_tokens(style_token_id) == STYLE_TOKEN

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = style_token_id
    return shifted_input_ids
