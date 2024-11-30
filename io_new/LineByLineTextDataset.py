import sys
import os
from typing import Dict

import torch

from common.tools import logger
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


# 为了全词掩码，对应的data
class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, train_file_path: str, block_size: int):
        assert os.path.isfile(train_file_path), f"Input file path {train_file_path} not found"
        print(f"Creating features from dataset file at {train_file_path}")

        with open(train_file_path, encoding="utf-8") as f:
            train_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        lines = train_lines

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        print(batch_encoding.keys())
        input_ids = batch_encoding["input_ids"]
        # token_type_ids = batch_encoding["token_type_ids"]
        attention_mask = batch_encoding["attention_mask"]

        self.examples = [{"input_ids": torch.tensor(input_ids[i], dtype=torch.long),
                          # "token_type_ids": torch.tensor(token_type_ids[i], dtype=torch.long),
                          "attention_mask": torch.tensor(attention_mask[i], dtype=torch.float)} for i in range(len(input_ids))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]