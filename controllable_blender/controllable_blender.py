from parlai.core.opt import Opt
from parlai.utils.typing import TShared
from parlai.agents.transformer.transformer import TransformerGeneratorAgent

from .generation_methods import VocabTopKSampling, RerankedTopKSampling
from .generation_utils import Wordlist, Reranker, load_wordlist, cefr_to_int

class ControllableBlender(TransformerGeneratorAgent):
    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__(opt, shared)

        if opt.get("inference", None) == "vocab":
            wordlist_path = opt.get("wordlist_path", None)
            assert wordlist_path, "Please provide path to vocab list, in order to use inference method 'vocab'"

            allowed_words = load_wordlist(wordlist_path)
            self.wordlist = Wordlist(allowed_words, self.dict)

        elif opt.get("inference", None) == "rerank":
            cefr = opt.get("rerank_cefr", None)
            assert cefr, "Please provide CEFR level, in order to use inference method 'rerank'"

            rerank_tokenizer = opt.get("rerank_tokenizer", None)
            rerank_model = opt.get("rerank_model", None)
            assert rerank_model, "Please provide path to directory containing model weights, in order to use inference method 'rerank'"

            device = opt.get("complexity_model_device", None)
            penalty_stddev = opt.get("penalty_stddev", None)
            text_truncate = opt.get("text_truncate", None)

            word_filter = None
            filter_path = opt.get("filter_path", "")
            if filter_path:
                word_filter = load_wordlist(filter_path)

            exempt_tokens = [self.dict.tok2ind.get(self.dict.null_token),
                             self.dict.tok2ind.get(self.dict.start_token),
                             self.dict.tok2ind.get(self.dict.end_token),
                             self.dict.tok2ind.get(self.dict.unk_token)]

            if penalty_stddev < 0:
                exempt_tokens = "all"

            self.reranker = Reranker(cefr=cefr_to_int(cefr),
                                     model=rerank_model,
                                     tokenizer=rerank_tokenizer,
                                     device=device,
                                     text_truncate=text_truncate,
                                     exempt_tokens=exempt_tokens,
                                     penalty_stddev=penalty_stddev,
                                     vocab_size=len(self.dict),
                                     word_filter=word_filter)

        else:
            raise ValueError(f"Inference method {opt.get('inference', None)} does not exist. "
                             f"Please use 'vocab' or 'rerank'.")


    def _treesearch_factory(self, device, verbose=False):
        method = self.opt.get('inference', 'greedy')
        beam_size = self.opt.get('beam_size', 1)
        if method == 'vocab':
            return VocabTopKSampling(
                k=self.opt.get('topk', 40),
                wordlist=self.wordlist,
                beam_size=beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
            )
        elif method == "rerank":
            return RerankedTopKSampling(
                k=self.opt.get('topk', 40),
                reranker=self.reranker,
                tokenids_to_text=self._v2t,
                beam_size=beam_size,
                min_length=self.beam_min_length,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
                verbose=verbose,
            )
        else:
            return super()._treesearch_factory(device, verbose=verbose)

    def share(self):
        """
        Share internal states between parent and child instances.
        """
        shared = super().share()
        if hasattr(self, 'wordlist'):
            shared['wordlist'] = self.wordlist
        if hasattr(self, 'reranker'):
            shared['reranker'] = self.reranker
        return shared


