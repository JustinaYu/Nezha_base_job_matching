import sys
import os

import torch
from accelerate import Accelerator
from torch.optim import AdamW, Adam, Adagrad
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler

from argparse import ArgumentParser
from config.fine_tuning_config import config
from torch.utils.data import DataLoader, random_split
from common.tools import logger, init_logger
from common.tools import seed_everything
from transformers import NezhaConfig
from fine_tuning.customize_model.NezhaBaseNetwork import NezhaBaselNetwork
from fine_tuning.io_new.fine_tuning_processor import FineTuningProcessor
from fine_tuning.io_new.process_source_data import spilt_source_data
from fine_tuning.train.fine_tuning_trainer import Trainer


def run_train(args, writer, accelerator):
    if args.process_source_json:
        print("process the source train.json file.")
        spilt_source_data(config['data_dir'] / args.input_file, config['data_dir'] / args.resume_data_json_file, config['data_dir'] / args.position_id_json_file)

    processor = FineTuningProcessor(vocab_path=config['nezha_vocab_ngram_path'], document_file= config['data_dir'] / args.resume_data_json_file, position_file= config['data_dir'] / args.position_id_json_file, args=args)
    examples = processor.create_examples(config['data_dir'] / args.cached_examples_file)
    print(len(examples))
    total_dataset = processor.create_dataset(examples)
    # 计算训练和验证样本量
    print("here")
    eval_size = args.eval_size
    train_size = len(total_dataset) - eval_size
    print(f"Total dataset size: {len(total_dataset)}")
    print(f"Train size: {train_size}, Eval size: {eval_size}")

    train_dataset, eval_dataset = random_split(total_dataset, [train_size, eval_size])
    # 验证eval_dataset每次是不是一样
    # print(eval_dataset.indices)
    # 存储eval_dataset
    torch.save(train_dataset, config['train_data_dir'] / args.cached_train_dataset)

    """
    num_workers 控制数据加载时使用的子进程数量，可以设置为 CPU 核心数的一半或等于核心数（物理）
    """
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
    #                                   num_workers=args.num_workers)
    # eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.eval_batch_size,
    #                                  num_workers=args.num_workers)
    # print("train_dataloader")
    # torch.set_printoptions(threshold=float('inf'))
    
    # for step, batch in enumerate(train_dataloader):
    #     input_ids, attention_mask, label = batch
    #     print(input_ids)
    #     break
    # for step, batch in enumerate(eval_dataloader):
    #     input_ids, attention_mask, label = batch
    #     print(input_ids)
    #     break



def main():
    parser = ArgumentParser()
    # 来自pre-training阶段
    parser.add_argument("--do_init", action='store_true', help="whether to init from the ckpt stored in the pretraining stage.")
    parser.add_argument('--init_nezha_ckpt', type=str, default="pretraining_model/6204-ckpt-20241109_081621.pt", help='the ckpt file from the pretraining stage.')


    # 总开关及总设置
    parser.add_argument("--do_train", action='store_true', help="whether to train")
    parser.add_argument('--seed', type=int, default=42, help="global random seed")

    # 1. 数据构建
    # data source
    parser.add_argument("--input_file", default="train.json", type=str,
                        help="the input files to train")
    parser.add_argument("--resume_data_json_file", default="resumeid_document.json", type=str,
                        help="the resumeid: document file after processing source json.")
    parser.add_argument("--position_id_json_file", default="resumeid_positionid.json", type=str,
                        help="the resumeid: positionid file after processing source json.")
    parser.add_argument("--process_source_json", action='store_true', help="whether to spilt the source json file(that is data/train.json)")
    parser.add_argument("--cached_examples_file", default="cached_examples_file_bert.pt", type=str,
                        help="the cached file to store examples to train")
    parser.add_argument("--train_max_seq_len", default=2048, type=int)

    # save eval_dataset
    parser.add_argument("--cached_eval_dataset", default="eval_dataset.pt", type=str,
                             help="the cached file to store eval dataset spilt in run_fine_tuning.py")
    # save train dataset
    parser.add_argument("--cached_train_dataset", default="train_dataset.pt", type=str,
                             help="the cached file to store train dataset spilt in run_fine_tuning.py")

    # dataloader setting
    parser.add_argument('--num_workers', type=int, default=5, help='Number of threads to load dataset')
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument('--eval_size', type=int, default=1000, help='eval data size')



    # 2. 训练设置
    # 2.0 gradient accumulation
    parser.add_argument("--do_accumulate", action='store_true', help="whether to do gradient accumulation")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of steps to accumulate before update')

    # 2.1 train loops
    parser.add_argument("--epochs", default=3, type=int)

    # 2.2 optimizer
    parser.add_argument("--optimizer_name", default='adamw', type=str, help="the name of the optimizer, adamw, adam or adagrad")
    parser.add_argument("--warmup_proportion", default=0.1, type=int, help="for optimizer")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="gradient clip, for optimizer")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="avoid overfit, for optimizer")

    # 2.2.1 adamw
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="for optimizer")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="basic learning rate, for optimizer")
    # 2.2.2 adam 同adamw
    # 2.2.3 adagrad
    parser.add_argument("--adagrad_epsilon", default=1e-10, type=float, help="for optimizer")
    parser.add_argument("--adagrad_learning_rate", default=2e-3, type=float, help="basic learning rate, for optimizer")

    # 2.3 checkpoint
    parser.add_argument("--checkpoint_name", default="nezha_checkpoint.pth", type=str,
                        help="the store path of outputs of training.")
    parser.add_argument("--save_step", default=1000, type=int,
                        help="Number of updates steps before two checkpoint saves if save_strategy=steps, work if we "
                             "in the last stage, if we in the early stage we will store the ckpt file each epoch")
    parser.add_argument("--save_total_limit", default=10, type=int,
                        help="The value limit the total amount of checkpoints")

    # 2.4 evaluation
    parser.add_argument('--eval_step', type=int, default=100, help='Validation interval steps')

    # 2.4 断点续训
    parser.add_argument('--from_trained', action='store_true', help='whether to train from pretrained ckpt')
    parser.add_argument('--ckpt_dir', type=str, default='fine_tuning/finetuning_model/checkpoints/20.0-ckpt.pt')

    args = parser.parse_args()
    seed_everything(args.seed)
    # init_logger(log_file=config['log_dir'] / "train.log")
    init_logger()
    # 启用梯度累积
    if args.do_accumulate:
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    else:
        accelerator = Accelerator()
    writer = SummaryWriter(config["writer_dir"])


    if args.do_train:  # conduct dataset generation
        run_train(args, writer, accelerator)


if __name__ == '__main__':
    main()
