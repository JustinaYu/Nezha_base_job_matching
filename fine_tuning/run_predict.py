import torch
from accelerate import Accelerator

from argparse import ArgumentParser
from config.fine_tuning_config import config
from torch.utils.data import DataLoader
from common.tools import logger, init_logger
from common.tools import seed_everything
from transformers import NezhaConfig
from fine_tuning.customize_model.NezhaBaseNetwork import NezhaBaselNetwork
from fine_tuning.io_new.predict_processor import PredictProcessor
from fine_tuning.io_new.process_source_data import spilt_test_data
import csv


def run_predict(args, accelerator):
    if args.process_test_json:
        print("process the test.json file.")
        spilt_test_data(config['predict_data_dir'] / args.input_file, config['predict_data_dir'] / args.test_resume_data_json_file)

    processor = PredictProcessor(vocab_path=config['nezha_vocab_ngram_path'],
                                 document_file=config['predict_data_dir'] / args.test_resume_data_json_file, args=args)
    examples = processor.create_examples(config['predict_data_dir'] / args.test_cached_examples_file)
    print(len(examples))
    predict_dataset = processor.create_dataset(examples)
    predict_dataloader = DataLoader(predict_dataset, shuffle=True, batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    # 自定义模型
    logger.info("initializing model")
    nezha_config = NezhaConfig.from_json_file(config['nezha_config_file'])
    model = NezhaBaselNetwork(config=nezha_config)

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

    predict_dataloader, model = accelerator.prepare(predict_dataloader, model)

    logger.info("Start predicting")
    model.eval()
    resume_ids = []
    position_ids = []
    with torch.no_grad():
        for step, batch in enumerate(predict_dataloader):
            input_ids, attention_mask, resume_id = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            max_val, max_idx = logits.max(dim=-1)
            # Gather predictions and ids from all GPUs
            gathered_resume_ids = accelerator.gather(resume_id).cpu().tolist()
            gathered_position_ids = accelerator.gather(max_idx).cpu().tolist()
        
            # Append gathered data to lists
            resume_ids.extend(gathered_resume_ids)
            position_ids.extend(gathered_position_ids)
    if accelerator.is_main_process:
        sorted_results = sorted(zip(resume_ids, position_ids), key=lambda x: x[0])
        res_path = config['predict_data_dir'] / 'predict.csv'
        with open(res_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['resumeRecordId', 'positionID'])
            for item1, item2 in sorted_results:
                writer.writerow([item1, item2])

    print("Finish Prediction!")
def main():
    parser = ArgumentParser()
    # 来自fine-tuning阶段
    parser.add_argument("--do_load", action='store_true',
                        help="whether to load from the ckpt stored in the fine-tuning stage.")
    parser.add_argument('--load_model_ckpt', type=str,
                        default="fine_tuning/finetuning_model/checkpoints/180.0-ckpt-20241123_160538.pt",
                        help='the ckpt file from the fine tuning stage of the whole model.')

    # 总开关及总设置
    parser.add_argument("--do_predict", action='store_true', help="whether to predict")
    parser.add_argument('--seed', type=int, default=42, help="global random seed")

    # 数据设置
    parser.add_argument("--input_file", default="test.json", type=str,
                        help="the input files to predict")
    parser.add_argument("--test_resume_data_json_file", default="test_resumeid_document.json", type=str,
                        help="the resumeid: document file after processing test.json.")
    parser.add_argument("--process_test_json", action='store_true', help="whether to transform the json file(that is data/test.json)")
    parser.add_argument("--test_cached_examples_file", default="test_cached_examples_file.pt", type=str,
                        help="the cached file to store examples to predict")
    parser.add_argument("--train_max_seq_len", default=2048, type=int)
    parser.add_argument('--num_workers', type=int, default=5, help='Number of threads to load dataset')
    parser.add_argument("--batch_size", default=16, type=int)


    args = parser.parse_args()
    seed_everything(args.seed)
    # init_logger(log_file=config['log_dir'] / "train.log")
    init_logger()
    accelerator = Accelerator()

    if args.do_predict:  # conduct dataset generation
        run_predict(args, accelerator)


if __name__ == '__main__':
    main()
