from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import NezhaModel

from fine_tuning.customize_model.NezhaBaseNetwork import ClassificationOutput
from fine_tuning.losses.FocalLoss import FocalLoss
from fine_tuning.losses.DSCLoss import MultiDSCLoss
from fine_tuning.losses.LabelSmoothingLoss import LabelSmoothingCrossEntropy


# core caps_layer with squash func
class Caps_Layer(nn.Module):
    def __init__(self, batch_size, input_dim_capsule, num_capsule, dim_capsule,
                 routings, kernel_size=(9, 1), share_weights=True,
                 activation='default'):
        # batch_size 即gru的输入(batch_size, seq_length, gru_len * 2)第一维
        # input_dim_capsule 即gru_len * 2
        # num_capsule
        super(Caps_Layer, self).__init__()

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size  # 暂时没用到
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash # 默认使用squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(batch_size, input_dim_capsule, self.num_capsule * self.dim_capsule))

    def forward(self, x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
        else:
            print('add later')
        # 此时u_hat_vecs是(batch_size, seq_length, self.num_capsule * self.dim_capsule)
        

        batch_size = x.size(0)
        input_num_capsule = x.size(1) # 认为seq_length是上一层capsule的数量，每个特征的细节存在gru_len * 2个数中
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 转成(batch_size, num_capsule, input_num_capsule, dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule, input_num_capsule)
        # digit capsule
        for i in range(self.routings):
            b = b.permute(0, 2, 1) # 这里的b是对raw weight的初始化，(batch_size, input_num_capsule, num_capsule)
            c = torch.nn.functional.softmax(b, dim=2) # raw weight转向 routing weight 对每个输入capsule的weight做softmax
            c = c.permute(0, 2, 1) # c转换成shape(batch_size, num_capsule, input_num_capsule)
            b = b.permute(0, 2, 1) # b转换成shape(batch_size, num_capsule, input_num_capsule)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  #  batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            # 输出和输入的相似度
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1, t_epsilon=1e-7):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + t_epsilon)
        return x / scale


# multi-sample dropout
class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_rate, num_samples):
        super(MultiSampleDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        self.drop_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        outputs = []
        for _ in range(self.num_samples):
            output = self.drop_layer(x)
            outputs.append(output)
        res = torch.mean(torch.stack(outputs), dim=0)
        return res.flatten(start_dim=1)


# dense layer after the capsule network
class Dense_Layer(nn.Module):
    def __init__(self, dropout_p, num_capsule, dim_capsule, num_classes):
        super(Dense_Layer, self).__init__()
        self.fc = nn.Sequential(
            # r-dropout和multi-sample dropout
            MultiSampleDropout(dropout_rate=dropout_p, num_samples=2),
            nn.Linear(num_capsule * dim_capsule, num_classes),  # num_capsule*dim_capsule -> num_classes
        )

    def forward(self, x):
        return self.fc(x)


class NezhaCapsuleNetwork(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.nezha = NezhaModel(config, add_pooling_layer=False)
        # nezha的sequenceoutput的size是(batch_size, seq_len, hidden_size)
        # 双向的注意输出
        self.lstm = nn.LSTM(input_size=768, hidden_size=384, num_layers=2, batch_first=True, dropout=0.2,
                             bidirectional=True)
        self.gru = nn.GRU(input_size=768, hidden_size=384, num_layers=2, batch_first=True, dropout=0.2,
                           bidirectional=True)
        # # 胶囊网络
        self.capsule = Caps_Layer(batch_size=args.train_batch_size, input_dim_capsule=768, num_capsule=10, dim_capsule=16, routings=3)
        # 参考nezhaforsequenceclassification写法
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = Dense_Layer(dropout_p=classifier_dropout, num_capsule=10, dim_capsule=16, num_classes=51)
        """
        loss 优化策略
        """
        if self.config.loss_function == "focal_loss":
            # alpha_value = [0.1345, 0.0285, 0.0575, 0.0435, 0.04, 0.035, 0.5255, 0.2385, 0.781, 0.058, 0.143, 0.7395,
            #      0.0425, 0.058, 1.1265, 1.8095, 0.11, 0.484, 0.288, 0.171, 0.3595, 0.0325, 0.055, 0.2445,
            #      0.0255, 0.025, 0.5035, 0.2085, 0.036, 0.0845, 0.028, 0.038, 0.028, 0.023, 0.024, 0.079,
            #      0.0425, 0.1875, 0.126, 0.0335, 0.081, 0.0355, 0.0345, 0.055, 0.036, 0.059, 0.039, 0.0815,
            #      0.035, 0.1055, 0.34]
            alpha_value = [0.0743, 0.3509, 0.1739, 0.2299, 0.25, 0.2857, 0.0190, 0.0419, 0.0128, 0.1724, 0.0699, 0.0135,
                           0.2353,
                           0.1724, 0.0089, 0.0055, 0.0909, 0.02077, 0.0347, 0.0585, 0.0278, 0.3077, 0.1818, 0.0409,
                           0.3922, 0.4,
                           0.0199, 0.0480, 0.2778, 0.1183, 0.3571, 0.2632, 0.3571, 0.4348, 0.4167, 0.1266, 0.2353,
                           0.0533, 0.0794,
                           0.2985, 0.1235, 0.2817, 0.2899, 0.1818, 0.2778, 0.1695, 0.2564, 0.1227, 0.2857, 0.0948,
                           0.0294]
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
        sequence_output = outputs[0]
        lstm_output,_ = self.lstm(sequence_output)
        gru_output,_ = self.gru(lstm_output)
        capsule_output = self.capsule(gru_output)
        logits = self.classifier(capsule_output)

        preds = torch.argmax(logits, dim=-1)
        # print(preds)
        loss = None
        if label is not None:
            label = label.view(-1).to(torch.long)
            # 需要修改label的类型
            loss = self.loss_fn(logits, label)

        return ClassificationOutput(loss=loss, logits=logits)
