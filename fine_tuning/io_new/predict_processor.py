import collections
import random
from typing import List

import torch
import sys
import os

from fine_tuning.io_new import fine_tuning_tokenization
from common.tools import logger
from callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset
import json


class InputExample(object):
    '''
    A single set of features of data.
    '''

    def __init__(self, input_ids, attention_mask, resume_id):
        self.input_ids = input_ids
        # 填充的标识
        self.attention_mask = attention_mask
        self.resume_id = resume_id


class PredictProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, vocab_path, document_file, args):
        self.tokenizer = fine_tuning_tokenization.FullTokenizer(vocab_file=vocab_path)
        self.args = args
        self.document_file = document_file

    def truncate_seq(self, tokens, max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        total_length = len(tokens)
        if total_length <= max_length:
            return tokens
        tokens = tokens[:max_length]
        return tokens

    # 创建能够输入到nezha模型中的数据 nezha model
    def create_examples_from_document(self, tokens, resumeid, max_seq_length):
        # 截断
        resumeid = int(resumeid)
        max_num_tokens = max_seq_length - 2
        tokens = tokens.split()
        tokens = self.truncate_seq(tokens, max_num_tokens)
        res_tokens = ["[CLS]"]
        res_tokens.extend(tokens)
        res_tokens.append("[SEP]")
        input_ids = self.tokenizer.convert_tokens_to_ids(res_tokens)
        attention_mask = [1] * len(input_ids)
        ### process to examples
        assert len(input_ids) <= max_seq_length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        instance = InputExample(input_ids=input_ids, attention_mask=attention_mask, resume_id=resumeid)
        return instance

    def extract_leaf_values(self, data):
        """
        递归提取 JSON 中所有叶子节点的值，并用空格连接。
        """
        if isinstance(data, dict):
            # 如果是字典，递归提取每个值
            tmp = []
            for value in data.values():
                tmp.append(self.extract_leaf_values(value))
            return ' '.join(item for item in tmp if item)
        elif isinstance(data, list):
            # 如果是列表，递归提取每个元素
            tmp2 = []
            for item in data:
                tmp2.append(self.extract_leaf_values(item))
            return ' '.join(item for item in tmp2 if item)
        elif isinstance(data, (str, int, float)):
            # 如果是叶子节点，返回其值
            return str(data)
        else:
            # 如果是其他类型（如 None），返回空字符串
            return ''

    def change_json_to_text(self, json_data):
        keys = ["profileEduExps", "profileSocialExps", "profileLanguage","profileProjectExps", "profileSkills",
                "profileAwards","profileWorkExps", "profileDesire"]
        results = []
        for key in keys:
            value = json_data.get(key, "")
            # 提取叶子节点的值并拼接
            res = self.extract_leaf_values(value)
            if value == "":
                results.append("[NAN]")
            else:
                results.append(res)
        # 用空格连接八个键的结果
        text = ' '.join(results)
        return text

    def create_examples(self, cached_examples_file):
        '''
        Creates examples for data
        '''
        # load examples from cache.
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            with open(self.document_file, 'r') as file:
                document_data = json.load(file)
            examples = []
            vocab_words = list(self.tokenizer.vocab.keys())
            # masked and get the two sentences
            pbar = ProgressBar(n_total=len(document_data.keys()))
            for resumeid in document_data.keys():
                document = document_data[resumeid]
                document = self.change_json_to_text(document)
                examples.append(self.create_examples_from_document(document, resumeid, self.args.train_max_seq_len))
                pbar.batch_step(step=1, info={}, bar_type='create examples')
            random.shuffle(examples)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_dataset(self, examples: List[InputExample]):
        print("run create_dataset function")
        all_input_ids = torch.tensor([f.input_ids for f in examples], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in examples], dtype=torch.float)
        all_resume_id = torch.tensor([f.resume_id for f in examples], dtype=torch.int)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_resume_id)
        return dataset
