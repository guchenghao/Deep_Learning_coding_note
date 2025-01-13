import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

# * ds表示数据集dataset, lang表示语言language
def get_or_build_tokenizer(config, ds, lang):
    
    # * config['tokenizer_file'] = "../tokenizers/tokenizer_{0}.json"
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer()
