from pathlib import Path

BASE_DIR = Path('.')
config = {
    # defined
    'data_dir': BASE_DIR / 'data',
    # defined
    'log_dir': BASE_DIR / 'output/log',
    # defined
    'writer_dir': BASE_DIR / "pretraining_model/TSboard",
    # defined
    'checkpoint_dir': BASE_DIR / "pretraining_model/checkpoints",
    # defined
    'nezha_vocab_wwm_path': BASE_DIR / 'data/vocab_wwm.txt',
    'nezha_vocab_ngram_path': BASE_DIR / 'data/vocab_ngram.txt',
    # defined
    'nezha_config_file': BASE_DIR / 'pretraining_model/nezha_config.json',
}
