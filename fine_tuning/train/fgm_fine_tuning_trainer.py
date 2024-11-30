import torch
import sys
import os
import glob

from torch.nn.utils import clip_grad_norm_

from config.fine_tuning_config import config

from callback.progressbar import ProgressBar
from common.tools import seed_everything
from datetime import datetime

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer(object):
    def __init__(self, args,
                 model,
                 ema_model,
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
        self.ema_model = ema_model
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
        self.fgm = FGM(self.model)

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
        input_ids, attention_mask, label = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, label=label)
        loss = outputs.loss
        # if step % 100 == 0:
        self.accelerator.backward(loss)
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        
        #fgm
        self.fgm.attack()
        outputs_adv = self.model(input_ids=input_ids, attention_mask=attention_mask, label=label)
        loss_adv = outputs_adv.loss
        self.accelerator.backward(loss_adv)
        self.fgm.restore()
        
        # 梯度裁剪
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        # 梯度更新
        self.optimizer.step()
        # 更新学习率
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # ema
        self.ema_model.update_parameters(self.model)
        
        update_step = (epoch * len(train_data) + step + 1) / self.gradient_accumulation_steps if self.args.do_accumulate else (epoch * len(train_data) + step + 1)
        # save
        self.accelerator.wait_for_everyone()
        if update_step % self.eval_step == 0:
            # 每eval_step做验证
            self.validate(eval_data, update_step)
        if self.accelerator.is_main_process:
            if (epoch * len(train_data) + step + 1) % self.gradient_accumulation_steps == 0:
                self.logger.info(f"EPOCH {epoch + 1}  -- STEP {update_step} : {loss:.4f} -- Loss value")
                self.writer.add_scalar('Loss/train(fine_tuning stage)', loss.item(), global_step=update_step)
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
                input_ids, attention_mask, label = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, label=label)
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
                self.writer.add_scalar('Loss/eval(fine_tuning stage)', loss_mean, global_step=update_step / self.eval_step)
                self.logger.info(f"Finished validation{str(update_step / self.eval_step)}. LOSS - {str(loss_mean)}")
        # transfer to train mode
        self.model.train()

    def train(self, train_data, eval_data, seed):
        seed_everything(seed)
        # train cycle
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch}/{self.epochs}")
            for step, batch in enumerate(train_data):
                self.model.train()
                if self.args.do_accumulate:
                    with (self.accelerator.accumulate(self.model)):
                        self.train_batch(step, batch, train_data, eval_data, epoch)
                else:
                    self.train_batch(step, batch, train_data, eval_data, epoch)
        # ema
        torch.optim.swa_utils.update_bn(train_data, self.ema_model)

        # save ema
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ema_checkpoint_file = config["ema_checkpoint_dir"] / f"ema-ckpt-{timestamp}.pt"

        torch.save({'model_state': self.ema_model.state_dict(),},ema_checkpoint_file)
