# Nezha_base_job_matching
A solution to the job matching competition in iFLYTEK based on the Nezha model.

## Job Matching
https://challenge.xfyun.cn/topic/info?type=match-challenge-s3
In this competition, the model is required to help filter out the resumes that best suit the job position. A large amount of encrypted and desensitized data of job JDs and job seekers' resumes as training samples. 

We can regard the person-job matching task as a **text classification** task. Since the data was encrypted and desensitized, instead of using a pre-trained model, we *pre-trained* the model from scratch using this data and further *fine-tuned* it afterwards.


## Project Introduction
All of the following use the latest version.
* Framework: **pytorch**
* Library: **transformers**
* DDP: **accelerate**
* Model: **Nezha** (https://arxiv.org/abs/1909.00204)

GPU source: Nivida A100-40G * 4 and Nivida A100-80G * 2

```
├── __init__.py
├── callback
│   ├── __init__.py
│   └── progressbar.py
├── common
│   ├── __init__.py
│   └── tools.py
├── config
│   ├── __init__.py
│   ├── config.py                   # the config file for the pre-training stage
│   └── fine_tuning_config.py       # the config file for the fine-tuning stage
├── data
│   ├── vocab_ngram.txt             # the vocab-file for n-gram mlm task
│   └── vocab_wwm.txt               # the vocab-file for wwm task
├── fine_tuning
│   ├── __init__.py
│   ├── customize_model             
│   ├── data                        
│   ├── finetuning_model
│   ├── io_new
│   ├── losses
│   ├── output
│   ├── run_eval.py
│   ├── run_fine_tuning.py
│   ├── run_predict.py
│   ├── run_train_1epoch.py
│   ├── test_data.py
│   └── train
├── io_new
│   ├── LineByLineTextDataset.py
│   ├── __init__.py
│   ├── bert_processor.py           # to build the dataset for the pre-training stage
│   └── tokenization.py
├── output # not use
│   └── log
├── pretraining_model
│   ├── TSboard # for tensorboard
│   ├── checkpoints                 # to store the ckpt in the pre-training stage
│   └── nezha_config.json           # the config for nezha in the pre-training stage
├── run_pretraining.py              # pre-training file
├── test                            # scripts and data for test
│   ├── data
│   ├── test.py
│   └── vocab.txt
├── test.py
└── train
    ├── __init__.py
    └── trainer.py                 # the trainer file for the pre-training stage

``````

## Data Requirements
* train.json (20000 samples) : used in the pre-training and fine-tuning stages.
* test.json (6000+ samples) : used in the pre-training and prediction stage.

### pre-training
We extracted all the resume data from train.json and test.json as a dataset and pre-trained them for the task of n-gram mask prediction. 

### fine-tuning
We take 1000 samples from train.json as the validation set. Since the track is closed, we use the model's f1 score on the validation set as the improvement target.


## Commands

### Pretrain the nezha model:

```PYTHONPATH=. accelerate launch run_pretraining.py --do_train```


### Fine-tune the model:

```PYTHONPATH=. accelerate launch fine_tuning/run_fine_tuning.py --do_train --do_accumulate --do_init```

We have two model choices for fine-tuning: *NezhaBaseNetwork* and *NezhaCapsnet*. You can choose them in the run_*.py file. Their structure is as follows:

*NezhaBaseNetwork*: Nezha (pooled_output) -> Dropout -> Linear

*NezhaCapsnet*: Nezha (sequence_output) -> Bi-LSTM -> Bi-GRU -> Capsule network -> Multi-sample Dropout -> Linear

### Evaluate the performance of fine-tuned model:

```PYTHONPATH=. accelerate launch fine_tuning/run_eval.py --do_load --do_eval```

You can modify the ckpt file by modifying the parameters in the run_eval.py file.

### Predict the job positions corresponding to the resumes in test.json:

```PYTHONPATH=. accelerate launch fine_tuning/run_predict.py --do_load --do_predict```

the results will be stored in fine_tuning/data/predict/predict.csv.

### export the results of fine-tuned model in train.json：

```PYTHONPATH=. accelerate launch fine_tuning/run_train_1epoch.py --do_eval```


## Tricks need to be concerned

* gradient accumulate
* mixed-precision calculation
* multi-sample dropout
* fgm
* ema
* voting
* loss function: focal loss / dice loss / label smoothing loss
* optimizer: adamw / adam / adagrad

* contrast learning ?
* Model Distillation ？
