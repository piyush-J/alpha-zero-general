import sys

sys.path.append('..')
import argparse
import functools
import math
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import (AutoModelForSequenceClassification, DistilBertModel,
                          DistilBertTokenizer)

from utils import *

from .SMTLogic import PREV_ACTIONS_EMBED

print = functools.partial(print, flush=True)

class MyTransformerEncoder(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int = 8, d_hid: int = 1024,
                 nlayers: int = 6, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) #  used to register a buffer that should not to be considered a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)] #  pe is a buffer, so we can access it with self.pe, ignore the IDE warning
        return self.dropout(x)

class SMTNNet(nn.Module):
    def __init__(self, game, args):
        super(SMTNNet, self).__init__()
        # game params
        self.board_x = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # self.pretrained_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        # self.pretrained_model.classifier = torch.nn.Linear(in_features=768, out_features=768, bias=True)

        # self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        # self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        # self.bn1 = nn.BatchNorm2d(args.num_channels)
        # self.bn2 = nn.BatchNorm2d(args.num_channels)
        # self.bn3 = nn.BatchNorm2d(args.num_channels)
        # self.bn4 = nn.BatchNorm2d(args.num_channels)
        
        if self.args.one_hot:
            self.embedding_size = self.action_size + 1 # one-hot, in case of embedding layer use self.args.embedding_size
            if self.args.time_series:
                raise Warning("Time series not implemented for one-hot encoding")
        else: # embedding
            self.embedding_size = self.args.embedding_size
            if self.args.time_series:
                self.transformer_encoder = MyTransformerEncoder(ntoken=self.action_size + 1, d_model=self.embedding_size)
            else:
                self.embeddings = nn.Embedding(self.action_size + 1, self.args.embedding_size, padding_idx=0)

        self.fc11 = nn.Linear(self.embedding_size*(PREV_ACTIONS_EMBED), 64)
        self.fc_bn11 = nn.BatchNorm1d(64)

        self.fc12 = nn.Linear(self.board_x, 64)
        self.fc_bn12 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(128, 32)
        self.fc_bn2 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32, self.action_size)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, s, prior_a):
        #                                                           s: batch_size x board_x
        # s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        # s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        # s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        # s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        # s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        # s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        # s = self.pretrained_model(**s)[0] # batch_size x 768
        # e = self.embeddings(prior_a) # batch_size x size of the prior action sequence x embedding_size
        
        if self.args.one_hot:
            e = F.one_hot(prior_a.to(torch.int64), num_classes = self.embedding_size).to(torch.float32) 
        else: # embedding
            if self.args.time_series:
                e = self.transformer_encoder(prior_a) # e: batch_size x size of the prior action sequence x embedding_size
            else:
                e = self.embeddings(prior_a) 
            
        e = torch.reshape(e, (e.shape[0], -1)) # batch_size x (embedding_size * size of the prior action sequence)
        e = F.dropout(F.relu(self.fc_bn11(self.fc11(e))), p=self.args.dropout, training=self.training)  # batch_size x 64
        s = F.dropout(F.relu(self.fc_bn12(self.fc12(s))), p=self.args.dropout, training=self.training)  # batch_size x 64
        s = torch.cat([s, e], 1) # batch_size x 128
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 32

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v) # TODO: https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
