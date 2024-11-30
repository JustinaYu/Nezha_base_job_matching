from dataclasses import dataclass
from typing import Optional
import sys
import os

import torch
from torch import nn
from transformers import NezhaModel

from fine_tuning.losses.FocalLoss import FocalLoss
from fine_tuning.losses.DSCLoss import MultiDSCLoss
from fine_tuning.losses.LabelSmoothingLoss import LabelSmoothingCrossEntropy


@dataclass
class ClassificationOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class NezhaBaselNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.nezha = NezhaModel(config)
        # 参考nezhaforsequenceclassification写法
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 一层dropout正则化
        self.dropout = nn.Dropout(classifier_dropout)
        # 一层linear全连接层 in_feature:hidden_size, out_feature:num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        """
        loss 优化策略
        """
        if self.config.loss_function == "focal_loss":
            # alpha_value = [0.1345, 0.0285, 0.0575, 0.0435, 0.04, 0.035, 0.5255, 0.2385, 0.781, 0.058, 0.143, 0.7395,
            #      0.0425, 0.058, 1.1265, 1.8095, 0.11, 0.484, 0.288, 0.171, 0.3595, 0.0325, 0.055, 0.2445,
            #      0.0255, 0.025, 0.5035, 0.2085, 0.036, 0.0845, 0.028, 0.038, 0.028, 0.023, 0.024, 0.079,
            #      0.0425, 0.1875, 0.126, 0.0335, 0.081, 0.0355, 0.0345, 0.055, 0.036, 0.059, 0.039, 0.0815,
            #      0.035, 0.1055, 0.34]
            alpha_value = [0.0743,0.3509,0.1739,0.2299,0.25,0.2857,0.0190,0.0419,0.0128,0.1724,0.0699,0.0135,0.2353,
                           0.1724,0.0089,0.0055,0.0909,0.02077,0.0347,0.0585,0.0278,0.3077,0.1818,0.0409,0.3922,0.4,
                           0.0199,0.0480,0.2778,0.1183,0.3571,0.2632,0.3571,0.4348,0.4167,0.1266,0.2353,0.0533,0.0794,
                           0.2985,0.1235,0.2817,0.2899,0.1818,0.2778,0.1695,0.2564,0.1227,0.2857,0.0948,0.0294]
            # alpha_value = [0.25,0.5,0.25,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.5,0.25,0.25,0.25,0.25,0.25,0.25,
            #                0.25,0.25,0.5,0.25,0.25,0.5,0.5,0.25,0.25,0.5,0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,
            #                0.5,0.25,0.5,0.5,0.25,0.5,0.25,0.5,0.25,0.5,0.25,0.25]
            # alpha_value = 0.25
            self.loss_fn = FocalLoss(num_class=51, alpha=alpha_value)
        elif self.config.loss_function == "label_smoothing":
            self.loss_fn = LabelSmoothingCrossEntropy()
        elif self.config.loss_function == "dice_loss":
            self.loss_fn = MultiDSCLoss()
        elif self.config.loss_function == "CE":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid loss function")

    def forward(self, input_ids, attention_mask=None, label=None):
        # 需要计算loss在output中输出,output包含了logit和loss
        outputs = self.nezha(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        preds = torch.argmax(logits, dim=-1)
        # print(preds)
        loss = None
        if label is not None:
            label = label.view(-1).to(torch.long)
            # 需要修改label的类型
            loss = self.loss_fn(logits, label)

        return ClassificationOutput(loss=loss, logits=logits)
