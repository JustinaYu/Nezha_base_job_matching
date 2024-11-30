import sys
import os
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DataCollatorForWholeWordMask, BertTokenizer, PreTrainedTokenizer, \
    DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import torch
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from config.config import config
from io_new.bert_processor import BertProcessor


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
                          "attention_mask": torch.tensor(attention_mask[i], dtype=torch.long)} for i in range(len(input_ids))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

# test wwm
tokenizer = BertTokenizer.from_pretrained("./vocab.txt")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train_dataset = LineByLineTextDataset(tokenizer, train_file_path="./test.tokens",
                                      block_size=20)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=data_collator)

for batch in train_dataloader:
    print(batch.keys())
    print("input_ids:")
    print(batch["input_ids"])
    # print("token_type_ids:")
    # print(batch["token_type_ids"])
    print("attention_mask:")
    print(batch["attention_mask"])
    print("labels:")
    print(batch["labels"])


# vocab有21个词，需要在model时+5
# 最后一个batch不填满不丢弃
# parser = ArgumentParser()
#     # data
# parser.add_argument("--ngram", default=3, type=int, help="n-gram num")
# parser.add_argument("--mask_strategy", default="n-gram", type=str, help="n-gram or wwm(whole word mask)")
#
# parser.add_argument("--input_file", default="total_resume_data.tokens", type=str, help="the input files to train")
# parser.add_argument("--cached_examples_file", default="cached_examples_file_bert.pt", type=str,
#                         help="the cached file to store examples to train")
# parser.add_argument("--do_data", action='store_true', help="whether to process document file(documents)")
# parser.add_argument("--do_train", action='store_true', help="whether to train")
# parser.add_argument("--epochs", default=6, type=int)
# parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
#                         help='Number of steps to accumulate before update')
# # data
# parser.add_argument('--num_workers', type=int, default=5, help='Number of threads to load dataset')
# parser.add_argument("--train_batch_size", default=2, type=int)
# parser.add_argument("--eval_batch_size", default=2, type=int)
# parser.add_argument("--train_max_seq_len", default=20, type=int)
# parser.add_argument("--warmup_proportion", default=0.1, type=int, help="for optimizer")
# parser.add_argument("--weight_decay", default=0.01, type=float, help="avoid overfit, for optimizer")
# parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="for optimizer")
# parser.add_argument("--grad_clip", default=1.0, type=float, help="gradient clip, for optimizer")
# parser.add_argument("--learning_rate", default=2e-5, type=float, help="basic learning rate, for optimizer")
# parser.add_argument('--seed', type=int, default=42, help="global random seed")
# parser.add_argument('--dupe_factor', type=int, default=2,
#                         help="Number of times to duplicate the input data (with different masks).")
# parser.add_argument('--masked_lm_prob', type=float, default=0.15, help="Masked LM probability.")
# parser.add_argument('--short_seq_prob', type=float, default=0.1,
#                         help="Probability of creating sequences which are shorter than the maximum length.")
# parser.add_argument('--max_predictions_per_seq', type=int, default=5,
#                         help="Maximum number of masked LM predictions per sequence.")
#     # store path
# parser.add_argument("--checkpoint_name", default="bert_checkpoint.pth", type=str,
#                         help="the store path of outputs of training.")
# parser.add_argument("--save_step", default=100, type=int,
#                         help="Number of updates steps before two checkpoint saves if save_strategy=steps")
# parser.add_argument("--save_total_limit", default=10, type=int,
#                         help="The value limit the total amount of checkpoints. ")
#     # eval
# parser.add_argument('--eval_size', type=int, default=1000, help='eval data size')
# parser.add_argument('--eval_step', type=int, default=5, help='Validation interval steps')
#
#     # 断点续训
# parser.add_argument('--from_trained', action='store_true', help='whether to train from pretrained ckpt')
# parser.add_argument('--ckpt_dir', type=str, default='model/checkpoints/xx.ckpt')
#
# args = parser.parse_args()
#
# # test ngram
# print("1")
# processor = BertProcessor(vocab_path="./vocab.txt", args=args)
# print("2")
# documents = processor.get_documents("./test.tokens")
# print(len(documents))
# print(len(documents[0]))
# print("3")
# examples = processor.create_examples(documents, Path("data") / args.cached_examples_file)
# print("4")
# total_dataset = processor.create_dataset(examples)
# print("5")
# train_dataloader = DataLoader(total_dataset, shuffle=True, batch_size=2)
#
# for batch in train_dataloader:
#     input_ids, labels = batch
#     print(input_ids)
#     print(labels)


