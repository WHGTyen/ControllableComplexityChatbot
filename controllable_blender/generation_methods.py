import math
from operator import attrgetter
from typing import Callable

import numpy as np
import regex
from scipy.stats import rankdata

import torch
from parlai.core.torch_generator_agent import TopKSampling, TreeSearch, _HypothesisTail, _PathSelection
from parlai.utils.torch import neginf

from .generation_utils import Reranker, Wordlist

class VocabTopKSampling(TopKSampling):

    def __init__(self,
                 k: int,
                 wordlist: Wordlist,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.wordlist = wordlist

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        """
        Select the next vocabulary item in these beams.
        """
        if len(self.all_scores) > 1:
            for hypid in range(self.beam_size):
                allowed_ids = self.wordlist.get_allowed_ids(self.partial_hyps[hypid])

                neginf_assign = torch.ones(logprobs.shape[1], dtype=bool)
                neginf_assign[allowed_ids] = False

                logprobs[hypid, neginf_assign] = neginf(logprobs.dtype)

        return super().select_paths(logprobs, prior_scores, current_length)


class RerankedTopKSampling(TreeSearch):
    def __init__(self,
                 k: int,
                 reranker: Reranker,
                 tokenids_to_text: Callable,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.reranker = reranker
        self.tokenids_to_text = tokenids_to_text

    def select_paths(self, logprobs, prior_scores, current_length) -> _PathSelection:
        """
        Select the next vocabulary item in these beams.
        Adapted from top-k sampling https://github.com/facebookresearch/ParlAI/blob/054a0fff8183e357727dc7a91682496734badb7f/parlai/core/torch_generator_agent.py
        """
        values, indices = logprobs.topk(self.k, dim=-1)
        probs = torch.softmax(values, dim=-1)

        all_penalties = self.reranker.token_penalties.repeat(self.beam_size, 1).to(probs.device)
        penalties = torch.gather(all_penalties, -1, indices)
        penalised_probs = torch.mul(probs, penalties)

        choices = torch.multinomial(penalised_probs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = indices[hyp_ids, choices]
        scores = values[hyp_ids, choices]
        best_scores = prior_scores.expand_as(scores) + scores

        token_details: Optional[List[_PathSelectionTokenDetails]] = None
        if self.verbose:
            tok_logprobs = probs[hyp_ids, choices].log().view(-1).cpu().numpy()
            tok_ranks = choices.view(-1).cpu().numpy()
            token_details = []

            for tok_logprob, tok_rank in zip(tok_logprobs, tok_ranks):
                token_details.append(
                    {"token_logprob": tok_logprob, "token_rank": int(tok_rank)}
                )

        return _PathSelection(
            hypothesis_ids=hyp_ids,
            token_ids=tok_ids,
            scores=best_scores,
            token_details=token_details,
        )


    def get_rescored_finished(self, n_best=None):
        """
        Adapted version of code taken from https://github.com/facebookresearch/ParlAI/blob/054a0fff8183e357727dc7a91682496734badb7f/parlai/core/torch_generator_agent.py
        Adds complexity scoring and reranking.

        Original description:
        Return finished hypotheses according to adjusted scores.
        Score adjustment is done according to the Google NMT paper, which
        penalizes long utterances.
        :param n_best:
            number of finalized hypotheses to return
        :return:
            list of (tokens, score, token_metadata) 3-tuples, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
              - token_metadata dictionary:
                    token_logprobs -> a tensor of conditional log probabilities of tokens
                    token_ranks -> a tensor of ranks of tokens in vocabulator, by probability, when sampled
        """
        # if we never actually finished, force one
        if not self.finished:
            self.outputs[-1][0] = self.eos
            self.finished.append(
                _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=0,
                    score=self.all_scores[-1][0],
                    tokenid=self.outputs[-1][0],
                    token_details=self.token_details[0][-1]
                    if self.token_details is not None
                    else None,
                )
            )

        # Calculate scores
        hyps_str = []
        length_penalties = []
        for finished_item in self.finished:
            token_ids = self._get_pretty_hypothesis(self._get_hyp_from_finished(finished_item))
            hyps_str.append(self.tokenids_to_text(token_ids))
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, self.length_penalty)
            length_penalties.append(length_penalty)

        original_scores = []
        for i, finished_item in enumerate(self.finished):
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, self.length_penalty)
            original_scores.append(finished_item.score.cpu() / length_penalty)

        complexity_scores = self.reranker.get_complexity_scores(hyps_str)
        complexity_ranks = rankdata(complexity_scores)
        original_ranks = rankdata(original_scores)

        combined_ranks = complexity_ranks + original_ranks


        rescored_finished = []
        for i, finished_item in enumerate(self.finished):
            score = combined_ranks[i]
            if "u/" in hyps_str[i] or "r/" in hyps_str[i]:      # Fix for Reddit language, see paper appendix
                score = np.array(-1, dtype=combined_ranks.dtype)

            if self.reranker.word_filter:
                for word in regex.findall("(?<=[^\p{L}])\p{Ll}+", hyps_str[i]): # Find all non-capitalised words
                    if word not in self.reranker.word_filter:
                        score = np.array(-1, dtype=combined_ranks.dtype)
                        break

            rescored_finished.append(
                _HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                    token_details=finished_item.token_details,
                )
            )

        # Note: beam size is almost always pretty small, so sorting is cheap enough
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        n_best_list = []
        for hyp in srted:
            hyp_data = self._get_hyp_from_finished(hyp)
            token_ids = self._get_pretty_hypothesis(hyp_data)
            token_metadata = (
                [tok.token_details for tok in reversed(hyp_data)]
                if self.verbose
                else None
            )
            n_best_list.append((token_ids, hyp.score, token_metadata))

        # check that there is at least one finished candidate
        # and assert that each of them contains only one EOS
        assert (
            len(n_best_list) >= 1
        ), f'TreeSearch returned {len(n_best_list)} candidates, must be >= 1'
        for (pred, score, _) in n_best_list:
            assert (pred == self.eos).sum() == 1, (
                f'TreeSearch returned a finalized hypo with multiple end tokens '
                f'with score {score.item():.2f}'
            )

        return n_best_list


