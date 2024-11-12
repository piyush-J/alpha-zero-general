import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from utils import *

class CdrlNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(CdrlNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 2, stride=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 2, stride=1)
        # self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1) 

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-2)*(self.board_y-2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # s: batch_size x board_x x board_y
        # s = F.pad(s, (0, self.board_y - s.size(2), 0, self.board_x - s.size(1)), "constant", -1)
        # print("s after pad: ", s.shape)
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        # print("s after view: ", s.shape)
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y?
        # print("s after conv1: ", s.shape)
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y?
        # print("s after conv2: ", s.shape)
        s = s.view(-1, self.args.num_channels*(self.board_x-2)*(self.board_y-2))


        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
