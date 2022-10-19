#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: lstm_model.py
@time: 2022/10/19 15:53
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt    #画图的

from mycode.util import conf
from mycode.util.util import MSLE

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


class Covid19Dataset(Dataset):
	def __init__(self, data):
		self.data = data
		self.x = []
		self.y = []
		for i in range(len(data) - WINDOW_SIZE - FORECAST_SIZE + 1):
			x = self.data.loc[i:i + WINDOW_SIZE-1, :]
			feature = torch.tensor(x.values, dtype=torch.float)
			y = self.data.loc[i + WINDOW_SIZE:i + WINDOW_SIZE+FORECAST_SIZE-1, :]
			label = torch.tensor(y.values, dtype=torch.float)
			self.x.append(feature)
			self.y.append(label.view(-1))

	def __len__(self):
		return len(self.x)

	def __getitem__(self, item):
		return self.x[item], self.y[item]


class Block(nn.Module):
	def __init__(self):
		super(Block, self).__init__()

	def forward(self, x, x_input):
		x_out = torch.max((1 + x) * x_input[:, -1, :], torch.tensor(0.0))
		return x_out


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# 3层lstm
		self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=3, batch_first=True)
		self.linear = nn.Sequential(
			# nn.Linear(64, 32),
			# nn.ReLU(),
			nn.Linear(64, FORECAST_SIZE)
		)
		self.block = Block()

	def forward(self, x_input):
		x,(_,_) = self.lstm(x_input)
		x = self.linear(x[:,-1,:])
		y = self.block(x, x_input)
		return y


if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	all_data = pd.read_csv(conf.tmp_data_paht+"all_data_new.csv")
	train_ = all_data[(all_data['time'] >= '2022-01-01 01:00:00') & (all_data['time'] < '2022-05-01 01:00:00')].reset_index(
		drop=True)
	WINDOW_SIZE = 24 * 14
	FORECAST_SIZE = 24 * 7
	BATCH_SIZE = 128
	LEARNING_RATE = 0.001
	NUM_EPOCH = 50
	for id in [
		 "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]:
		data = train_[train_["flow_id"] == id]
		train = data[data["time"] < '2022-04-23 01:00:00'].reset_index(drop=True)
		test = data[data["time"] >= '2022-04-07 01:00:00'].reset_index(drop=True)
		ds_train = Covid19Dataset(train.loc[:, ["flow"]])
		dl_train = DataLoader(
			ds_train,
			batch_size=BATCH_SIZE,
			shuffle=False,
			drop_last=False,
			pin_memory=False)
		ds_test = Covid19Dataset(test.loc[:, ["flow"]])
		dl_test = DataLoader(
			ds_test,
			batch_size=BATCH_SIZE,
			shuffle=False,
			drop_last=False,
			pin_memory=False)
		criterion = nn.MSELoss()
		model = Net().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		for epoch in range(NUM_EPOCH):
			model.train()
			train_loss = 0.0
			for x,y in dl_train:
				x = x.to(device)
				y = y.to(device)
				optimizer.zero_grad()
				outputs = model(x)
				batch_loss = criterion(outputs, y)
				batch_loss.backward()
				optimizer.step()
				train_loss += batch_loss.item()
			print('[{:03d}/{:03d}] Train Loss: {:3.6f}'.format(epoch + 1, NUM_EPOCH, train_loss / len(ds_train)))
		plt.figure()
		plt.plot(range(FORECAST_SIZE), y[-1].cpu(), label="true")
		plt.plot(range(FORECAST_SIZE), outputs[-1].cpu().detach().numpy(), label="pre")
		plt.legend()
		plt.show()