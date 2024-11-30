from pathlib import Path

BASE_DIR = Path('./fine_tuning')
config = {
    # defined
    'data_dir': BASE_DIR / 'data',
    'eval_data_dir': BASE_DIR / 'data/eval',
    'train_data_dir': BASE_DIR / 'data/train',
    'predict_data_dir': BASE_DIR / 'data/predict',
    # defined
    'log_dir': BASE_DIR / 'output/log',
    # defined
    'writer_dir': BASE_DIR / "finetuning_model/TSboard",
    # defined
    'checkpoint_dir': BASE_DIR / "finetuning_model/checkpoints",
    # defined
    'ema_checkpoint_dir': BASE_DIR / "finetuning_model/ema-checkpoints",
    # defined
    'nezha_vocab_wwm_path': BASE_DIR / 'data/vocab_wwm.txt',
    'nezha_vocab_ngram_path': BASE_DIR / 'data/vocab_ngram.txt',
    # defined
    'nezha_config_file': BASE_DIR / 'finetuning_model/nezha_config.json',
}
