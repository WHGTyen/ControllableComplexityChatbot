import json

from parlai.zoo.blender.blender_3B import download
from parlai.core.opt import Opt

from controllable_blender import ControllableBlender
from demo_utils import start_interaction

agent_opt = json.load(open("blender_3B.opt", 'r'))


# Set to "vocab" for vocabulary restriction, "rerank" for candidate reranking
agent_opt["inference"] = "rerank"

# Same top-k sampling configs for all settings described in the paper
agent_opt["beam_size"] = 20
agent_opt["topk"] = 40

# Settings for rerank methods (not used if "inference" == "vocab")
agent_opt["rerank_cefr"] = "B2"                             # CEFR level to adjust reranking. Possible values: ['A2', 'B1', 'B2', 'C1', 'C2'].
agent_opt["rerank_tokenizer"] = "distilroberta-base"        # Tokenizer from Huggingface Transformers. Must be compatible with "rerank_model"
agent_opt["rerank_model"] = "complexity_model"              # Model fine-tuned on complexity data
agent_opt["rerank_model_device"] = "cuda"                   # Device for complexity model
agent_opt["penalty_stddev"] = 2                             # Controls how harshly sub-tokens are penalised (lower = harsher). Use -1 to remove penalties
agent_opt["filter_path"] = "data/filter.txt"                # Path to list of English words to ensure OOV words are not generated. Capitalised words are ignored. Use empty string to remove filter

# Settings for vocab methods (not used if "inference" == "rerank")
agent_opt["wordlist_path"] = "data/sample_wordlist.txt"          # Path to list of vocab the chatbot is restricted to


download(agent_opt["datapath"])

agent = ControllableBlender(agent_opt)
agent.set_interactive_mode(True)

# See https://github.com/facebookresearch/ParlAI/blob/main/parlai/scripts/interactive.py
interaction_opts = {
    "display_examples": False,
    "display_add_fields": "",
    "interactive_task": True,
    "outfile": "",
    "save_format": "conversations",
    "log_keep_fields": "all",
    "task": "interactive",
    "datatype": "test"
}

start_interaction(agent, Opt(interaction_opts))
