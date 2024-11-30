import torch
from accelerate import Accelerator

from argparse import ArgumentParser
from config.fine_tuning_config import config
from torch.utils.data import DataLoader
from common.tools import logger, init_logger
from common.tools import seed_everything
from transformers import NezhaConfig
from fine_tuning.customize_model.NezhaBaseNetwork import NezhaBaselNetwork
from fine_tuning.customize_model.NezhaCapsnet import NezhaCapsuleNetwork
import torchmetrics

def run_eval(args, accelerator):
    eval_dataset = torch.load(config['eval_data_dir'] / args.cached_eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    # 自定义模型
    logger.info("initializing model")
    nezha_config = NezhaConfig.from_json_file(config['nezha_config_file'])
    model = NezhaBaselNetwork(config=nezha_config)
    # model = NezhaCapsuleNetwork(config=nezha_config, args=args)

    device = accelerator.device
    model.to(device)

    # load权重
    if args.do_load:  # 使用pretraining阶段的ckpt初始化
        logger.info(f"load ckpt from the fine-tuning stage model : {args.load_model_ckpt}")
        model_ckpt = torch.load(args.load_model_ckpt, map_location=torch.device('cpu'))
        finetuned_dict = model_ckpt['model_state']
        # 获取当前的模型状态
        cur_model_dict = model.state_dict()
        # 只保留匹配的键
        finetuned_dict = {k: v for k, v in finetuned_dict.items() if k in cur_model_dict}
        cur_model_dict.update(finetuned_dict)
        model.load_state_dict(cur_model_dict)

    eval_dataloader, model = accelerator.prepare(eval_dataloader, model)

    logger.info("Start predicting")
    model.eval()
    torch.set_printoptions(threshold=float('inf'))
    f1 = torchmetrics.F1Score(num_classes=args.num_classes, task='multiclass', average='weighted').to(device)
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            print(str(step))
            input_ids, attention_mask, label_ids = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, max_idx = logits.max(dim=-1)
            # print(input_ids) 
            # print(logits)
            # print(max_idx)
            # print("step0")
            accelerator.wait_for_everyone()
            # print("step1")
            gathered_label_ids = accelerator.gather(label_ids)
            # print("step2")
            gathered_idx = accelerator.gather(max_idx)
            # print("step3")
            if accelerator.is_main_process:
                f1.update(gathered_idx, gathered_label_ids)
    print("finish cycle")        
    # 计算f1 score
    accelerator.wait_for_everyone() 
    print(f"F1 score on device {accelerator.device}: ready to compute.")     
    final_f1 = f1.compute()
    print(final_f1)
    print("Finish Prediction!")
        
def main():
    parser = ArgumentParser()
    # 来自fine-tuning阶段
    parser.add_argument("--do_load", action='store_true',
                        help="whether to load from the ckpt stored in the fine-tuning stage.")
    # fine_tuning/finetuning_model/1100.0-ckpt-20241127_154014.pt
    # fine_tuning/finetuning_model/checkpoint-second-3epoch/900.0-ckpt-20241124_120949.pt
    # fine_tuning/finetuning_model/checkpoints/300.0-ckpt-20241127_180207.pt
    # fine_tuning/finetuning_model/checkpoints-alpha/400.0-ckpt-20241125_104052.pt
    # fine_tuning/finetuning_model/checkpoints/1600.0-ckpt-20241128_045102.pt
    # fine_tuning/finetuning_model/checkpoints-alpha/400.0-ckpt-20241125_104052.pt
    # fine_tuning/finetuning_model/checkpoints/1100.0-ckpt-20241129_161406.pt
    # fine_tuning/finetuning_model/ema-checkpoints/ema-ckpt-20241129_163124.pt
    parser.add_argument('--load_model_ckpt', type=str,
                        default="fine_tuning/finetuning_model/checkpoints/200.0-ckpt-20241129_131027.pt",
                        help='the ckpt file from the fine tuning stage of the whole model.')

    # 总开关及总设置
    parser.add_argument("--do_eval", action='store_true', help="whether to eval in the evaluation dataset from the train.json")
    parser.add_argument('--seed', type=int, default=42, help="global random seed")

    # 数据设置
    # 把evaluation dataset存储起来
    parser.add_argument("--cached_eval_dataset", default="eval_dataset.pt", type=str,
                        help="the cached file to store eval dataset spilt in run_fine_tuning.py")
    parser.add_argument("--train_max_seq_len", default=2048, type=int)
    parser.add_argument('--num_workers', type=int, default=5, help='Number of threads to load dataset')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)

    # 下游任务设置
    parser.add_argument("--num_classes", default=51, type=int)


    args = parser.parse_args()
    seed_everything(args.seed)
    init_logger()
    accelerator = Accelerator()

    if args.do_eval:  # conduct dataset generation
        run_eval(args, accelerator)


if __name__ == '__main__':
    main()
