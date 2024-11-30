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
from fine_tuning.customize_model.NezhaCapsnet import NezhaCapsuleNetwork
from fine_tuning.io_new.fine_tuning_processor import FineTuningProcessor
from fine_tuning.io_new.process_source_data import spilt_source_data
# from fine_tuning.train.fine_tuning_trainer import Trainer
# 应用fgm
from fine_tuning.train.fgm_fine_tuning_trainer import Trainer
from datetime import datetime


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
    # torch.save(eval_dataset, config['eval_data_dir'] / args.cached_eval_dataset)

    """
    num_workers 控制数据加载时使用的子进程数量，可以设置为 CPU 核心数的一半或等于核心数（物理）
    """
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                      num_workers=args.num_workers)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.eval_batch_size,
                                     num_workers=args.num_workers)


    # 自定义模型
    logger.info("initializing model")
    nezha_config = NezhaConfig.from_json_file(config['nezha_config_file'])
    model = NezhaBaselNetwork(config=nezha_config)
    # model = NezhaCapsuleNetwork(config=nezha_config, args=args)

    # ema
    ema_model = torch.optim.swa_utils.AveragedModel(model)



    device = accelerator.device
    model.to(device)
    
    # ema
    ema_model.to(device)
    
    """
    optimizer 优化策略
    - Adamw有权重衰减,其余都是默认值0
    - Adam和Adamw的参数重叠
    - Adagrad单独设置
    """
    if args.optimizer_name == "adamw":
        params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer_name == "adam":
        optimizer = Adam(lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer_name == "adagrad":
        optimizer = Adagrad(lr=args.adagrad_learning_rate, eps=args.adagrad_epsilon)
    else:
        raise ValueError("Invalid optimizer name {}".format(args.optimizer_name))

    """
    scheduler 优化策略
    """
    # update_total = int(len(train_dataloader) * args.epochs)
    if args.do_accumulate:
        update_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    else:
        update_total = int(len(train_dataloader) * args.epochs)
    warmup_steps = int(update_total * args.warmup_proportion)
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=update_total)

    # load_state 断点续训
    if args.from_trained:  # 加载之前保存的训练状态
        logger.info(f"Loading trained model : {args.ckpt_dir}")
        ckpt = torch.load(args.ckpt_dir, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        lr_scheduler.load_state_dict(ckpt['lr_state'])
    # 断点续训和初始化不能一起用
    if args.do_init: # 使用pretraining阶段的ckpt初始化
        logger.info(f"Init the pre-trained nezha model : {args.init_nezha_ckpt}")
        # 只对能够匹配的权重状态进行更新
        nezha_ckpt = torch.load(args.init_nezha_ckpt, map_location=torch.device('cpu'))
        pretrained_dict = nezha_ckpt['model_state']
        # 获取当前的模型状态
        model_dict = model.state_dict()
        # 只保留匹配的键
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


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
                      ema_model=ema_model, # ema
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
    # 来自pre-training阶段
    parser.add_argument("--do_init", action='store_true', help="whether to init from the ckpt stored in the pretraining stage.")
    # pretraining_model/6204-ckpt-20241109_081621.pt
    # fine_tuning/finetuning_model/checkpoints/2000-ckpt-20241119_040049.pt
    # fine_tuning/finetuning_model/checkpoint-second-3epoch/900.0-ckpt-20241124_120949.pt
    # fine_tuning/finetuning_model/800.0-ckpt-20241127_124024.p
    # fine_tuning/finetuning_model/1100.0-ckpt-20241127_154014.pt
    parser.add_argument('--init_nezha_ckpt', type=str, default="fine_tuning/finetuning_model/checkpoint-second-3epoch/900.0-ckpt-20241124_120949.pt", help='the ckpt file from the pretraining stage.')


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

    # dataloader setting
    parser.add_argument('--num_workers', type=int, default=5, help='Number of threads to load dataset')
    parser.add_argument("--train_batch_size", default=6, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument('--eval_size', type=int, default=1000, help='eval data size')



    # 2. 训练设置
    # 2.0 gradient accumulation
    parser.add_argument("--do_accumulate", action='store_true', help="whether to do gradient accumulation")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
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
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="basic learning rate, for optimizer")
    # 2.2.2 adam 同adamw
    # 2.2.3 adagrad
    parser.add_argument("--adagrad_epsilon", default=1e-10, type=float, help="for optimizer")
    parser.add_argument("--adagrad_learning_rate", default=2e-5, type=float, help="basic learning rate, for optimizer")

    # 2.3 checkpoint
    parser.add_argument("--checkpoint_name", default="nezha_checkpoint.pth", type=str,
                        help="the store path of outputs of training.")
    parser.add_argument("--save_step", default=100, type=int,
                        help="Number of updates steps before two checkpoint saves if save_strategy=steps, work if we "
                             "in the last stage, if we in the early stage we will store the ckpt file each epoch")
    parser.add_argument("--save_total_limit", default=10, type=int,
                        help="The value limit the total amount of checkpoints")

    # 2.4 evaluation
    parser.add_argument('--eval_step', type=int, default=100, help='Validation interval steps')

    # 2.4 断点续训
    parser.add_argument('--from_trained', action='store_true', help='whether to train from pretrained ckpt')
    parser.add_argument('--ckpt_dir', type=str, default='fine_tuning/finetuning_model/checkpoints/2000-ckpt-20241119_040049.pt')

    # 3 优化
    # 3.1 fgm
    parser.add_argument('--ema_decay', type=float, default=0.999, help='the decay rate used in the ema model')


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
