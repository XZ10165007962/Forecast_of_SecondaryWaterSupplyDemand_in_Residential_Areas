#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: nn_model.py
@time: 2022/10/8 22:29
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch import nn

from mycode.util import conf
from mycode.util.util import MSLE

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def fueat(data):
	feat_col = []
	for i in range(1, 4):
		data[f"flow_{i}+1"] = data.groupby(["flow_id"])["flow"].shift(24*i+1)
		data[f"flow_{i}"] = data.groupby(["flow_id"])["flow"].shift(24*i)
		data[f"flow_{i}-1"] = data.groupby(["flow_id"])["flow"].shift(24*i-1)
		data[f"flow_{i}+1_roll"] = data.groupby(["flow_id"])["flow"].shift(24 * i + 1).rolling(6).mean()
		data[f"flow_{i}_roll"] = data.groupby(["flow_id"])["flow"].shift(24 * i).rolling(6).mean()
		data[f"flow_{i}-1_roll"] = data.groupby(["flow_id"])["flow"].shift(24 * i - 1).rolling(6).mean()
		feat_col.extend([f"flow_{i}+1",f"flow_{i}",f"flow_{i}-1",f"flow_{i}+1_roll",f"flow_{i}_roll",f"flow_{i}-1_roll"])
	for i in range(24*7, 24*7+13):
		data[f"flow_lag_{i}"] = data.groupby(["flow_id"])["flow"].shift(i)
		data[f"flow_roll_{i}"] = data.groupby(["flow_id"])["flow"].shift(i).rolling(6).mean()
		feat_col.extend([f"flow_lag_{i}", f"flow_roll_{i}"])
	data["time"] = pd.to_datetime(data["time"])
	data["month"] = data["time"].dt.month
	# data["week"] = data["time"].dt.week
	# data["day"] = data["time"].dt.day
	data["dayofweek"] = data["time"].dt.dayofweek
	# data["dayofyear"] = data["time"].dt.dayofyear
	data["hour"] = data["time"].dt.hour
	data['sin_hour'] = np.sin(2 * np.pi * data["hour"] / 24)
	data['cos_hour'] = np.cos(2 * np.pi * data["hour"] / 24)
	feat_col.extend(['sin_hour','cos_hour'])
	return data, feat_col


class FML_model(nn.Module):
	def __init__(self, input_dim):
		super(FML_model, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1)
		)
		self.critertion = nn.MSELoss(reduction="mean")

	def forward(self, input):
		return self.net(input).squeeze(1)

	def cal_loss(self, pred, target):
		return self.critertion(pred, target)


class dataSets(Dataset):
	def __init__(self, X, y, model='train'):
		super().__init__()
		self.model = model
		if model == 'test':
			data = X
			self.data = torch.FloatTensor(data)
		else:
			target = y
			data = X
			if model == 'train':
				self.target = torch.FloatTensor(target)
				self.data = torch.FloatTensor(data)
			else:
				self.target = torch.FloatTensor(target)
				self.data = torch.FloatTensor(data)
		self.dim = self.data.shape[1]

	def __getitem__(self, item):
		if self.model == 'train' or self.model == 'dev':
			return self.data[item], self.target[item]
		else:
			return self.data[item]

	def __len__(self):
		return len(self.data)


if __name__ == '__main__':
	data = pd.read_csv(conf.tmp_data_paht+"all_data_new.csv")
	train1 = data[(data['time']>='2022-01-01 01:00:00')&(data['time']<'2022-05-01 01:00:00')].reset_index(drop=True)
	train2 = data[(data['time']>='2022-05-08 01:00:00')&(data['time']<'2022-06-01 01:00:00')].reset_index(drop=True)
	train3 = data[(data['time']>='2022-06-08 01:00:00')&(data['time']<'2022-07-21 01:00:00')].reset_index(drop=True)
	train4 = data[(data['time']>='2022-07-28 01:00:00')&(data['time']<'2022-08-21 01:00:00')].reset_index(drop=True)
	data, feat = fueat(train1)
	train_ = data[data['time'] >= '2022-02-15 01:00:00']
	# train_ = data.dropna()
	pre_list = []
	pre_list_knn = []
	pre_list_svr = []
	pre_list_bagg = []
	pre_list_rand = []
	test_list = []
	batch_size = 64
	learning_rate = 0.001
	num_epoch = 200
	for id in [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]:
		model = FML_model(len(feat))
		print(f"+++++++++++++++++{id}+++++++++++++++++++++")
		train = train_[train_["flow_id"] == id]
		X_data, y_data = train[feat], train["flow"]
		X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=2022)
		train_set = dataSets(X_train.values, y_train.values)
		test_set = dataSets(X_test.values, y_test.values, "test")
		train_loader = DataLoader(
			train_set, batch_size,
			shuffle=False,
			drop_last=False,
			pin_memory=False
		)
		test_loader = DataLoader(
			test_set, batch_size,
			shuffle=False,
			drop_last=False,
			pin_memory=False
		)
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		for epoch in range(num_epoch):
			model.train()
			train_loss = 0.0
			for i, data in enumerate(train_loader):
				inputs, labels = data
				optimizer.zero_grad()
				outputs = model(inputs)
				batch_loss = criterion(outputs, labels)
				batch_loss.backward()
				optimizer.step()
				train_loss += batch_loss.item()
			print('[{:03d}/{:03d}] Train Loss: {:3.6f}'.format(epoch + 1,num_epoch,train_loss / len(train_loader),))
		model.eval()
		output = []
		for input in test_loader:
			pred = model(input)
			output.append(pred.detach())
		output = torch.cat(output, dim=0).numpy()
		pre_list.extend(output)
		test_list.extend(y_test)
	print("mlse:", MSLE(test_list, pre_list, flag=1))
	#
	# 	pre_list.extend(pre_list_all)
	# 	test_list.extend(y_test)
	# print("mlse:", MSLE(test_list, pre_list, flag=1))
	# print("knn mlse:", MSLE(test_list, pre_list_knn, flag=1))
	# print("svr mlse:", MSLE(test_list, pre_list_svr, flag=1))
	# print("bagg mlse:", MSLE(test_list, pre_list_bagg, flag=1))
	# print("rand mlse:", MSLE(test_list, pre_list_rand, flag=1))
