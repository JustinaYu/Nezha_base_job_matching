import torch
import sys
import os
import glob

from torch.nn.utils import clip_grad_norm_

from config.config import config

from callback.progressbar import ProgressBar
from common.tools import seed_everything
from datetime import datetime


class Trainer(object):
    def __init__(self, args,
                 model,
                 device,
                 checkpoint_name,
                 save_step,
                 writer,
                 accelerator,
                 epochs,
                 logger,
                 optimizer,
                 lr_scheduler,
                 eval_step,
                 gradient_accumulation_steps,
                 grad_clip=1.0,
                 ):
        self.args = args
        self.model = model
        self.epochs = epochs
        self.logger = logger
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.writer = writer
        self.checkpoint_name = checkpoint_name
        self.save_step = save_step
        self.eval_step = eval_step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.checkpoint_files = glob.glob(os.path.join(config["checkpoint_dir"], '*.pt'))
        self.checkpoint_files.sort(key=os.path.getctime)

    def cleanup_checkpoints(self, new_checkpoint_file):
        self.checkpoint_files.append(new_checkpoint_file)
        max_checkpoints = self.args.save_total_limit
        # 如果文件数量超过最大值，则删除最旧的文件
        if len(self.checkpoint_files) > max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            os.remove(old_checkpoint)

    def save_info(self, epoch, best):
        if hasattr(self.model, 'module'):
            print("has attr: module")
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def train_batch(self, step, batch, train_data, eval_data, epoch):
        if self.args.mask_strategy == "wwm":
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
        elif self.args.mask_strategy == "n-gram":
            input_ids, attention_mask, labels = batch
        else:
            raise NotImplementedError
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # if step % 100 == 0:
        self.writer.add_scalar('Loss/train(training stage)', loss.item(), global_step=epoch * len(train_data) + step)
        self.accelerator.backward(loss)
        # 梯度裁剪
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        # 梯度更新
        self.optimizer.step()
        # 更新学习率
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        update_step = (epoch * len(train_data) + step + 1) / self.gradient_accumulation_steps  if self.args.do_accumulate else (epoch * len(train_data) + step + 1)
        # save if in the latter stage
        self.accelerator.wait_for_everyone()
        if update_step % (self.eval_step) == 0:
            # 每eval_step做验证
            self.validate(eval_data, update_step)
        if self.accelerator.is_main_process:
            self.logger.info(f"EPOCH {epoch + 1}  -- STEP {step} : {loss:.4f} -- Loss value")
            if update_step % self.save_step == 0:
                # 每save_step保存状态
                unwrap_model = self.accelerator.unwrap_model(self.model)
                unwrap_optim = self.accelerator.unwrap_model(self.optimizer)
                unwrap_lr = self.accelerator.unwrap_model(self.lr_scheduler)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_checkpoint_file = config["checkpoint_dir"] / f"{str(update_step)}-ckpt-{timestamp}.pt"
                self.cleanup_checkpoints(new_checkpoint_file)
                torch.save({
                    'model_state': unwrap_model.state_dict(),
                    'optim_state': unwrap_optim.state_dict(),
                    'lr_state': unwrap_lr.state_dict()},
                    new_checkpoint_file)

    def validate(self, eval_data, update_step):
        # self.logger.info(f"Start validation{str(update_step/self.eval_step)}.")
        self.model.eval()
        losses = []
        # forbid gradient computation
        with torch.no_grad():
            # eval_pbar = ProgressBar(n_total=len(eval_data), disable=not self.accelerator.is_main_process)
            for step, batch in enumerate(eval_data):
                if self.args.mask_strategy == "wwm":
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                elif self.args.mask_strategy == "n-gram":
                    input_ids, attention_mask, labels = batch
                else:
                    raise NotImplementedError
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                print(f"LOSS:{str(loss.item())}-STEP:{step}")
                # eval_pbar.batch_step(step=step, info={}, bar_type='eval a batch')
                all_losses = self.accelerator.gather(loss)
                if self.accelerator.is_main_process:
                    losses.extend(all_losses.cpu().numpy())
            # eval_pbar.close()
            print("finished")
            if self.accelerator.is_main_process:
                loss_mean = sum(losses) / len(losses)
                self.writer.add_scalar('Loss/eval(training stage)', loss_mean, global_step=update_step / self.eval_step)
                self.logger.info(f"Finished validation{str(update_step / self.eval_step)}. LOSS - {str(loss_mean)}")
        # transfer to train mode
        self.model.train()

    def train(self, train_data, eval_data, seed):
        seed_everything(seed)
        # train cycle
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch}/{self.epochs}")
            self.logger.info(f"Batch Num: {len(train_data)}")
            for step, batch in enumerate(train_data):
                self.model.train()
                if self.args.do_accumulate:
                    with (self.accelerator.accumulate(self.model)):
                        self.train_batch(step, batch, train_data, eval_data, epoch)
                else:
                    self.train_batch(step, batch, train_data, eval_data, epoch)
