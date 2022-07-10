import math
import os
import json
from typing import List, Set, Union, Optional

import numpy as np
import torch
from parlai.core.dict import DictionaryAgent
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def cefr_to_int(cefr: str) -> int:
    mapping = {
        "A1": 0,
        "A2": 1,
        "B1": 2,
        "B2": 3,
        "C1": 4,
        "C2": 5,
    }
    clean_cefr = cefr.upper().strip()
    assert clean_cefr in mapping, f"CEFR must be one of {list(mapping.keys())}, not {cefr}"

    return mapping[clean_cefr]


def load_wordlist(path: str) -> List[str]:
    """
    Load a list of words from a text file containing one word per line
    """
    vocab = []

    if not path:
        return vocab

    assert os.path.isfile(path)

    with open(path, 'r', encoding="utf-8") as vocab_file:
        for row in vocab_file:
            token = row.strip()
            vocab.append(token)

    return vocab


class Wordlist():
    def __init__(self, allowed_words: List[str], dict_agent: DictionaryAgent):
        self.dict_agent = dict_agent

        # Identify IDs that represent a word boundary and those that don't
        self.boundary_ids = []
        self.non_boundary_ids = []

        for idx, subtoken in dict_agent.ind2tok.items():
            if subtoken[0] == "\u0120" or not subtoken.isalpha():
                self.boundary_ids.append(idx)
            else:
                self.non_boundary_ids.append(idx)

        # Identify token ID sequences that are allowed words
        # Identify allowed continuations of sequences
        self.allowed_sequences = []
        self.allowed_continuations = {}
        for word in allowed_words:
            for word_variant in self._get_word_variants(word):
                token_ids = dict_agent.txt2vec(word_variant)
                self.allowed_sequences.append(repr(token_ids))

                for i, idx in enumerate(token_ids[1:]):
                    prefix = repr(token_ids[:i + 1])      # List represented as string for lookup
                    if prefix not in self.allowed_continuations:
                        self.allowed_continuations[prefix] = []
                    self.allowed_continuations[prefix].append(idx)

        self.allowed_sequences = set(self.allowed_sequences)


    def get_allowed_ids(self, token_ids: List[int]) -> List[int]:
        last_word = self._get_last_word(token_ids)
        continuation_ids = self._get_continuation_ids(last_word)

        return continuation_ids


    def _is_word(self, token_ids: List[int]) -> bool:
        """
        For a given sequence of token IDs, determine whether that sequence is a complete word
        """
        return (token_ids == [] or repr(token_ids) in self.allowed_sequences)


    def _get_continuation_ids(self, token_ids: List[int]) -> List[int]:
        """
        For a given sequence of last word token IDs, determine which token IDs the word can continue with
        """
        continuation_ids = []
        if repr(token_ids) in self.allowed_continuations:
            continuation_ids.extend(self.allowed_continuations[repr(token_ids)])

        if self._is_word(token_ids) or token_ids == []:
            continuation_ids.extend(self.boundary_ids)

        return continuation_ids


    def _get_last_word(self, token_ids: List[int]) -> List[int]:
        """
        Get the sequence of token IDs after the last word boundary.
        Assumes that a word boundary is denoted by punctuation or whitespace (Ġ).
        """
        for i in range(-1, -len(token_ids), -1):
            last_word = token_ids[i:]
            check_token = self.dict_agent[last_word[0]]

            if not check_token.isalpha():
                return last_word[1:]

            if check_token[0] == "Ġ":
                return last_word

        raise ValueError("Boundary token not found")


    def _get_word_variants(self, word: str) -> Set[str]:
        return {word, word.lower(), word.capitalize()}



class Reranker():
    def __init__(self,
                 cefr: int,
                 model: str,
                 tokenizer: str = "distilroberta-base",
                 device: Optional[str] = "cuda",
                 text_truncate: int = 128,
                 exempt_tokens: Union[str, List[int]] = "all",
                 penalty_stddev: int = 2,
                 vocab_size: int = 8008,
                 word_filter: Optional[List[str]] = None):

        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        self.model = RobertaForSequenceClassification.from_pretrained(model)
        self.model.to(device)
        self.device = device

        self.target_cefr = cefr
        self.text_truncate = text_truncate
        self.word_filter = word_filter

        cefr_filepath = os.path.join(os.path.dirname(__file__), 'tokens_by_cefr.json')
        with open(cefr_filepath, 'r') as cefr_file:
            token_cefrs = json.load(cefr_file)

        if exempt_tokens == "all" or penalty_stddev < 0:      # No penalties
            self.token_penalties = torch.tensor([[1] * vocab_size])
        else:
            # calculate penalties per CEFR level difference (0 = same CEFR)
            normal_dist = torch.distributions.normal.Normal(0, penalty_stddev)
            cefr_penalties = [math.exp(normal_dist.log_prob(torch.tensor(i))) for i in range(6)]

            token_penalties = []
            for i in range(vocab_size):
                if i in exempt_tokens:
                    token_penalties.append(cefr_penalties[0])

                elif str(i) in token_cefrs:
                    token_str, token_cefr = token_cefrs[str(i)]
                    penalty = cefr_penalties[int(token_cefr - self.target_cefr)]

                    if token_cefr <= self.target_cefr or not token_str.isalpha():         # ignore lower CEFR levels and punctuation/special tokens
                        penalty = cefr_penalties[0]

                    token_penalties.append(penalty)

                else:       # Assume highest CEFR level if we don't have an assigned CEFR level
                    token_penalties.append(cefr_penalties[int(5 - self.target_cefr)])

            self.token_penalties = torch.tensor([token_penalties])

    def get_complexity_scores(self, hyps: List[str]) -> np.ndarray:
        model_inputs = self.tokenizer(hyps,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.text_truncate,
                                      return_tensors='pt',
                                      return_token_type_ids=True,
                                      return_attention_mask=True)

        model_output = self.model(input_ids=model_inputs["input_ids"].to(self.device),
                                  attention_mask=model_inputs["attention_mask"].to(self.device),
                                  token_type_ids=model_inputs["token_type_ids"].to(self.device))

        complexity_scores = model_output.logits.cpu().numpy().flatten()
        complexity_diffs = 5 - np.absolute(complexity_scores - self.target_cefr)      # reversed so that higher score = better

        return complexity_diffs

