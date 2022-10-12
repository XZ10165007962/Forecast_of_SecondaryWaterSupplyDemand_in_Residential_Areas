#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: Data.py
@time: 2022/10/8 14:20
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

from mycode.util import conf
from mycode.util.util import abnormal_data, fill_nan

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)



def get_all_data():
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
	all_data["day_time"] = list(map(lambda x: str(x)[:10], all_data["time"]))
	data_["day_time"] = list(map(lambda x: str(x)[:10], data_["time"]))
	weather_data = pd.read_csv(conf.train_data_path + "weather.csv")
	epi_data = pd.read_csv(conf.train_data_path + "epidemic.csv").rename(columns={"jzrq": "day_time"})
	epi_data = epi_data.fillna(0)
	# all_data = all_data.merge(weather_data.loc[:, ["time", "R", "fx", "T", "U", "fs", "V", "P"]], on=["time"],
	# 						  how="left")
	# all_data = all_data.merge(epi_data.loc[:, ["day_time", "zz", "wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]],
	# 						  on=["day_time"], how="left")
	data_ = data_.merge(weather_data.loc[:, ["time", "R", "fx", "T", "U", "fs", "V", "P"]], on=["time"],
						how="left")
	data_ = data_.merge(epi_data.loc[:, ["day_time", "zz", "wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]],
						on=["day_time"], how="left")
	del all_data["day_time"]
	del data_["day_time"]
	all_data = abnormal_data(all_data)
	all_data = fill_nan(all_data)
	all_data = abnormal_data(all_data)
	all_data = fill_nan(all_data)
	# all_data["time"] = pd.to_datetime(all_data["time"])
	# all_data["dayofyear"] = all_data["time"].dt.dayofyear

	data_.to_csv(conf.tmp_data_paht + "hour_data.csv", index=False)
	all_data.reset_index(drop=True, inplace=True)
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


def get_result():
	all_data = pd.read_csv(conf.predict_data_path + "sub_all.csv")
	all_data = all_data[all_data["train or test"] != "train"]
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	sub = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		temp = all_data[all_data["flow_id"] == flow].reset_index(drop=True)
		if i == 0:
			sub = temp.loc[:, ["time", "flow"]]
		else:
			sub = pd.concat([sub, temp.loc[:, ["flow"]]], axis=1)
	sub.columns = ["time",
				   "flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10",
				   "flow_11",
				   "flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
				   ]
	sub.to_csv(conf.predict_data_path + "sub.csv", index=False)


if __name__ == '__main__':
	all_data = get_all_data()
	all_data.to_csv(conf.tmp_data_paht + "all_data_new.csv", index=False)
	# get_result()
	# data = pd.read_csv(conf.predict_data_path+"orinal_data.csv")
	# data = data[["train or test", "time", "flow_id", "knn", "svr", "bagging", "randomtree"]]
	# data = data[data["train or test"] != "train"]
	# pre_list = data[["knn", "svr", "bagging", "randomtree"]].values
	# # 排序
	# pre_list = np.sort(pre_list, axis=1)
	# sub_pre = []
	# for i in tqdm(pre_list):
	# 	diff = [0,0]
	# 	for j in range(2):
	# 		for k in range(j, j+2):
	# 			diff[j] += abs(i[k]-i[k+1])
	# 	min_index = diff.index(min(diff))
	# 	sub_pre.append((i[min_index]+i[min_index+1]+i[min_index+2])/3)
	# data["flow"] = sub_pre
	# print(data.iloc[:20, 3:])
	# flow_id = [
	# 	"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
	# 	"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	# ]
	# sub = pd.DataFrame()
	# for i, flow in enumerate(flow_id):
	# 	temp = data[data["flow_id"] == flow].reset_index(drop=True)
	# 	if i == 0:
	# 		sub = temp.loc[:, ["time", "flow"]]
	# 	else:
	# 		sub = pd.concat([sub, temp.loc[:, ["flow"]]], axis=1)
	# sub.columns = ["time",
	# 			   "flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10",
	# 			   "flow_11",
	# 			   "flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	# 			   ]
	# sub.to_csv(conf.predict_data_path + "sub_1.csv", index=False)
