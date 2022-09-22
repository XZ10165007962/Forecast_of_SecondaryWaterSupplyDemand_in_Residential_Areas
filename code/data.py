# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         data
# Description:
# Author:       xinzhuang
# Date:         2022/9/12
# Function:
# Version：
# Notice:
# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import conf
from util import data_cleaning, out_liner

# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)

"""
整个数据 2022-01-01 01:00:00 开始
训练集划分为
test1 开始 2022-04-01 01:00:00 2160  结束 2022-04-08 00:00:00  2328
"""
"""
test1 开始 2022-05-01 01:00:00 2881  结束 2022-05-08 00:00:00  3048
test2 开始 2022-06-01 01:00:00 3625  结束 2022-06-08 00:00:00  3792
test3 开始 2022-07-21 01:00:00 4825  结束 2022-07-28 00:00:00  4992
test4 开始 2022-08-21 01:00:00 5569  结束 2022-08-28 00:00:00  5736
"""


def all_data():
	data_ = pd.read_csv(conf.train_data_path + "hourly_dataset.csv")
	data_["time_index"] = np.arange(1, data_.shape[0] + 1)
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	all_data = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		if i == 0:
			data = data_.loc[:, ["time", "time_index", flow, "train or test"]].rename(columns={flow: "flow"})
			data["flow_id"] = flow
			all_data = data
		else:
			data = data_.loc[:, ["time", "time_index", flow, "train or test"]].rename(columns={flow: "flow"})
			data["flow_id"] = flow
			all_data = pd.concat([all_data, data], axis=0)
	all_data.reset_index(drop=True,inplace=True)
	return all_data


def get_data(data_, time_index):
	data_ = data_[data_["time_index"] <= time_index]
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	all_data = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		print(f"=============={flow}==============")
		if i == 0:
			data = data_[data_["flow_id"] == flow].reset_index(drop=True)
			data = data.loc[:, ["time", "time_index", "flow", "train or test", "flow_id"]]
			data = data_cleaning(data)
			all_data = data
		else:
			data = data_[data_["flow_id"] == flow].reset_index(drop=True)
			data = data.loc[:, ["time", "time_index", "flow", "train or test", "flow_id"]]
			if flow == "flow_19":
				data.loc[[2, 3, 4, 5, 6], ["flow"]] = [0.922571378, 0.774033298, 0.387016649, 0.65820085, 1.105177825]
			elif flow == "flow_18":
				data.loc[[0, 1, 2, 3, 4, 5], ["flow"]] = [2.902, 2.269, 1.055, 0.677, 0.891, 1.572]
			data = data_cleaning(data)
			all_data = pd.concat([all_data, data], axis=0)
	all_data.reset_index(drop=True,inplace=True)
	all_data["flow_true"] = all_data["flow"]
	return all_data


def split_data(data_, split_col, split_flag, label_col, feature_col):
	train_data = data_[data_[split_col] < split_flag-2]
	val_data = data_[data_[split_col] == split_flag-1]
	test_data = data_[data_[split_col] == split_flag]
	train_x = train_data.loc[:, feature_col]
	train_y = train_data.loc[:, [label_col]]
	val_x = val_data.loc[:, feature_col]
	val_y = val_data.loc[:, [label_col]]
	test_x = test_data.loc[:, feature_col]
	test_y = test_data.loc[:, [label_col]]
	return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':
	all_data = all_data()
	all_data = get_data(all_data, 2880)
	print(all_data.head())
	# all_data.to_csv(conf.tmp_data_paht + "all_data.csv", index=False)