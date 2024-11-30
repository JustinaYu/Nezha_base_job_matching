import sys
import os

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler

from io_new.bert_processor import BertProcessor
from train.trainer import Trainer

from argparse import ArgumentParser
from config.config import config
from torch.utils.data import DataLoader, random_split
from common.tools import logger, init_logger
from common.tools import seed_everything
from transformers import BertTokenizer, DataCollatorForLanguageModeling, NezhaForMaskedLM, NezhaConfig
from io_new.LineByLineTextDataset import LineByLineTextDataset


def run_train(args, writer, accelerator):
    if args.mask_strategy == "wwm":
        # 全词掩码版本dataset
        tokenizer = BertTokenizer.from_pretrained(config['nezha_vocab_wwm_path'])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.masked_lm_prob)
        total_dataset = LineByLineTextDataset(tokenizer, train_file_path=config['data_dir'] / args.input_file,
                                              block_size=args.train_max_seq_len)

    # ngram版本dataset
    # get train dataset
    elif args.mask_strategy == "n-gram":
        processor = BertProcessor(vocab_path=config['nezha_vocab_ngram_path'], args=args)
        documents = processor.get_documents(config['data_dir'] / args.input_file) if args.do_data else [[]]
        examples = processor.create_examples(documents, config['data_dir'] / args.cached_examples_file)
        print(len(examples))
        total_dataset = processor.create_dataset(examples)

    else:
        print("Unknown mask strategy")
        raise NotImplementedError

    # 计算训练和验证样本量
    print("here")
    eval_size = args.eval_size
    train_size = len(total_dataset) - eval_size
    print(f"Total dataset size: {len(total_dataset)}")
    print(f"Train size: {train_size}, Eval size: {eval_size}")

    train_dataset, eval_dataset = random_split(total_dataset, [train_size, eval_size])

    """
    num_workers 控制数据加载时使用的子进程数量，可以设置为 CPU 核心数的一半或等于核心数（物理）
    """
    if args.mask_strategy == "wwm":
        # 全词掩码版本dataloader
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                      num_workers=args.num_workers, collate_fn=data_collator)
        eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.eval_batch_size,
                                     num_workers=args.num_workers, collate_fn=data_collator)
    elif args.mask_strategy == "n-gram":
        # ngram版本dataloader
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                      num_workers=args.num_workers)
        eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.eval_batch_size,
                                     num_workers=args.num_workers)
    else:
        print("Unknown mask strategy")
        raise NotImplementedError

    # model
    logger.info("initializing model")
    nezha_config = NezhaConfig.from_json_file(config['nezha_config_file'])

    model = NezhaForMaskedLM(config=nezha_config)

    device = accelerator.device
    model.to(device)

    # the total update times
    # update_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    update_total = int(len(train_dataloader) / args.epochs)
    warmup_steps = int(update_total * args.warmup_proportion)
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=update_total)

    # load_state 断点续训
    if args.from_trained:  # 加载之前保存的训练状态
        logger.info(f"Loading trained model : args.ckpt_dir")
        ckpt = torch.load(args.ckpt_dir, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        lr_scheduler.load_state_dict(ckpt['lr_state'])

    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(train_dataloader,
                                                                                            eval_dataloader, model,
                                                                                            optimizer, lr_scheduler)
    # mixed precision fp16现在可以通过设置training_args，accelerator或者pytorch内置的amp实现，不需要apex
    # callback
    """
    断点续训可以通过Trainer实现
    """
    logger.info("initializing callbacks")

    # train
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.epochs)
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", update_total)

    trainer = Trainer(args = args,
                      model=model,
                      device=device,
                      checkpoint_name=args.checkpoint_name,
                      save_step=args.save_step,
                      writer=writer,
                      accelerator=accelerator,
                      epochs=args.epochs,
                      logger=logger,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      grad_clip=args.grad_clip,
                      eval_step=args.eval_step,
                      gradient_accumulation_steps=args.gradient_accumulation_steps)
    trainer.train(train_data=train_dataloader, eval_data=eval_dataloader, seed=args.seed)


def main():
    parser = ArgumentParser()

    # 总开关及总设置
    parser.add_argument("--do_train", action='store_true', help="whether to train")
    parser.add_argument('--seed', type=int, default=42, help="global random seed")

    # 1. 数据构建
    # data source
    parser.add_argument("--input_file", default="nezha_total_resume_data.tokens", type=str,
                        help="the input files to train")

    # 1.1 mlm total setting
    parser.add_argument("--mask_strategy", default="n-gram", type=str, help="n-gram or wwm(whole word mask)")
    parser.add_argument("--train_max_seq_len", default=2048, type=int)
    parser.add_argument('--masked_lm_prob', type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument('--max_predictions_per_seq', type=int, default=200,
                        help="Maximum number of masked LM predictions per sequence.")

    # 1.2 mlm == ngram
    parser.add_argument("--ngram", default=3, type=int, help="n-gram num")
    parser.add_argument("--cached_examples_file", default="cached_examples_file_bert.pt", type=str,
                        help="the cached file to store examples to train")
    parser.add_argument("--do_data", action='store_true', help="whether to process document file(documents), just work in ngram mode")

    # 1.3 mlm == wwm（NONE）
    # 1.4 dataloader setting
    parser.add_argument('--num_workers', type=int, default=5, help='Number of threads to load dataset')
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument('--eval_size', type=int, default=1000, help='eval data size')



    # 2. 训练设置
    # 2.0 gradient accumulation
    parser.add_argument("--do_accumulate", action='store_true', help="whether to do gradient accumulation")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of steps to accumulate before update, it is used when do_accumulate')
    # 2.1 train loops
    parser.add_argument("--epochs", default=3, type=int)
    # 2.2 optimizer
    parser.add_argument("--warmup_proportion", default=0.1, type=int, help="for optimizer")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="avoid overfit, for optimizer")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="for optimizer")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="gradient clip, for optimizer")
    parser.add_argument("--learning_rate", default=1e-7, type=float, help="basic learning rate, for optimizer")

    # 2.3 checkpoint
    parser.add_argument("--checkpoint_name", default="bert_checkpoint.pth", type=str,
                        help="the store path of outputs of training.")
    parser.add_argument("--save_step", default=3102, type=int,
                        help="Number of updates steps before two checkpoint saves if save_strategy=steps, work if we "
                             "in the last stage, if we in the early stage we will store the ckpt file each epoch")
    parser.add_argument("--save_total_limit", default=5, type=int,
                        help="The value limit the total amount of checkpoints")

    # 2.4 evaluation
    parser.add_argument('--eval_step', type=int, default=100, help='Validation interval steps')

    # 2.4 断点续训
    parser.add_argument('--from_trained', action='store_true', help='whether to train from pretrained ckpt')
    parser.add_argument('--ckpt_dir', type=str, default='pretraining_model/checkpoints/6204-ckpt-20241109_081621.pt')

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
