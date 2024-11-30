import collections
import random
from typing import List

import torch
import sys
import os

from io_new import tokenization
from common.tools import logger
from callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset
import numpy as np


class InputExample(object):
    '''
    A single set of features of data.
    '''

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, vocab_path, args):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path)
        self.args = args

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

    def create_masked_lm_data(self, tokens, masked_lm_prob, max_predictions_per_seq, vocab_words):
        labels = [-100] * len(tokens)
        cand_indexes = []
        output_tokens = list(tokens)
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[NAN]" or token == "[UNK]":
                continue
            cand_indexes.append(i)
        # cand_indexes现在包含所有非特殊词的token
        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))
        # 计算需要掩码的数量
        ngrams = np.arange(1, self.args.ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, self.args.ngram + 1)
        pvals /= pvals.sum(keepdims=True)
        # 每个n对应的概率
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)

        random.shuffle(ngram_indexes)
        # ngram_indexes每一项是一个数组，数组中包含了一个位置token的所有n-gram token，已被打散
        masked_lms = []
        covered_indexes = set()
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                # 已达到需要掩码的数量
                break
            if not cand_index_set:
                continue
            if cand_index_set[0][0] in covered_indexes:
                # 1-gram 即token已被掩码则跳过
                continue
            # 从一个一维数组中随机取样,取得n
            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                   pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = list(cand_index_set[n - 1])
            n -= 1
            while len(masked_lms) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = list(cand_index_set[n - 1])
                n -= 1
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            # 所有需要掩码的当前n-gram的token
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            # 如果有covered就不掩码了
            if is_any_index_covered:
                continue
            for index in index_set:
                labels[index] = self.tokenizer.convert_token_to_id(tokens[index])
                covered_indexes.add(index)
                if random.random() < 0.8:
                    mask_word = "[MASK]"
                else:
                    if random.random() < 0.5:
                        mask_word = tokens[index]
                    else:
                        # 闭区间随机数
                        mask_word = vocab_words[random.randint(6, len(vocab_words) - 1)]
                output_tokens[index] = mask_word
                masked_lms.append(MaskedLmInstance(index=int(index), label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        input_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        return input_ids, labels

    def create_examples_from_document(self, tokens, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_words):
        max_num_tokens = max_seq_length - 2
        tokens = self.truncate_seq(tokens, max_num_tokens)
        res_tokens = ["[CLS]"]
        res_tokens.extend(tokens)
        res_tokens.append("[SEP]")
        input_ids, labels = self.create_masked_lm_data(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words)
        attention_mask = [1] * len(input_ids)
        ### process to examples
        assert len(input_ids) <= max_seq_length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            labels.append(-100)
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(labels) == max_seq_length
        instance = InputExample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return instance

    # 已修改完毕
    def get_documents(self, input_file):
        all_documents = []
        with open(input_file, "r") as reader:
            while True:
                # 处理utf-8编码
                line = reader.readline()
                if not line:
                    break
                # 去除行空格
                line = line.strip()
                # 行分词
                tokens = self.tokenizer.tokenize(line)
                if tokens:
                    # 往最后一个document对应的数组里添加行对应tokens
                    all_documents.append(tokens)
        all_documents = [x for x in all_documents if x]
        random.shuffle(all_documents)
        return all_documents

    def create_examples(self, all_documents, cached_examples_file):
        '''
        Creates examples for data
        '''
        # load examples from cache.
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            vocab_words = list(self.tokenizer.vocab.keys())
            # masked and get the two sentences
            pbar = ProgressBar(n_total=len(all_documents))
            for document in all_documents:
                examples.append(self.create_examples_from_document(document, self.args.train_max_seq_len, self.args.masked_lm_prob,
                                                           self.args.max_predictions_per_seq, vocab_words))
                pbar.batch_step(step=1, info={}, bar_type='create examples')
            random.shuffle(examples)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_dataset(self, examples: List[InputExample]):
        print("run create_dataset function")
        all_input_ids = torch.tensor([f.input_ids for f in examples], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in examples], dtype=torch.float)
        all_labels = torch.tensor([f.labels for f in examples], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
        return dataset

