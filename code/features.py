#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: features.py
@time: 2022/9/15 10:44
@version:
@desc: 
"""
import copy

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def get_features(data_, lag=24, rolling=2):
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6",	"flow_7", "flow_8",	"flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	all_data = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		if i == 0:
			data = data_.loc[:, ["time", flow]].rename(columns={flow: "flow"})
			data["flow_id"] = flow
			all_data = data
		else:
			data = data_.loc[:, ["time", flow]].rename(columns={flow: "flow"})
			data["flow_id"] = flow
			all_data = pd.concat([all_data, data], axis=0)
	features = ["flow"]
	all_data, time_feature = time_features(all_data)
	features.extend(time_feature)
	# 获取滞后特征
	for i in range(lag+1):
		all_data[f"flow_lag_{i}"] = all_data.groupby(["flow_id"])["flow"].shift(i)
		features.append(f"flow_lag_{i}")
		all_data[f"flow_lag_{i}_roll"] = all_data.groupby(["flow_id"])["flow"].shift(i).rolling(rolling).mean()
		features.append(f"flow_lag_{i}_roll")
	# 同比
	all_data["flow_lag_25"] = all_data.groupby(["flow_id"])["flow"].shift(25)
	all_data["flow_lag_49"] = all_data.groupby(["flow_id"])["flow"].shift(49)
	all_data["flow_lag_25_49_mean"] = list(map(lambda x,y: (x+y)/2, all_data["flow_lag_25"], all_data["flow_lag_49"]))
	features.append("flow_lag_25")
	features.append("flow_lag_49")
	features.append("flow_lag_25_49_mean")
	# 计算统计特征
	funcs = ["mean", "sum", "median", "max", "min", "std"]
	simple = copy.deepcopy(all_data)
	flag = 2
	for func in funcs:
		all_data[f"flow_id_{func}"] = all_data.groupby(["flow_id"])["flow"].transform(func)
		features.append(f"flow_id_{func}")
	print(all_data.head())
	for func in funcs:
		simple[f"month_{func}"] = simple.groupby(["month", "flow_id"])["flow"].transform(func)
		simple[f"month_weekday_{func}"] = simple.groupby(["day", "is_weekday", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple.rename(
				columns={f"month_{func}":f"month_{func}_{i}", f"month_weekday_{func}":f"month_weekday_{func}_{i}"},
				inplace=True
			)
			features.append(f"month_{func}_{i}")
			features.append(f"month_weekday_{func}_{i}")
			simple["month"] = list(map(lambda x: x+1, simple["month"]))
			all_data = all_data.merge(simple.loc[:, ["month", "flow_id", f"month_{func}_{i}"]].drop_duplicates(),
									  on=["month", "flow_id"], how="left")
			all_data = all_data.merge(simple.loc[:, ["month", "flow_id", f"month_weekday_{func}_{i}"]].drop_duplicates(),
									  on=["month", "flow_id"], how="left")
	for func in funcs:
		simple[f"week_{func}"] = simple.groupby(["week", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple.rename(
				columns={f"week_{func}": f"week_{func}_{i}"},
				inplace=True
			)
			features.append(f"week_{func}_{i}")
			simple["week"] = list(map(lambda x: x+1, simple["week"]))
			all_data = all_data.merge(simple.loc[:, ["week", "flow_id", f"week_{func}_{i}"]].drop_duplicates(),
									  on=["week", "flow_id"], how="left")
	for func in funcs:
		simple[f"day_{func}"] = simple.groupby(["dayofyear", "flow_id"])["flow"].transform(func)
		simple[f"day_free_{func}"] = simple.groupby(["month", "is_free", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple.rename(
				columns={f"day_{func}": f"day_{func}_{i}", f"day_free_{func}": f"day_free_{func}_{i}"},
				inplace=True
			)
			features.append(f"day_{func}_{i}")
			features.append(f"day_free_{func}_{i}")
			simple["dayofyear"] = list(map(lambda x: x+1, simple["dayofyear"]))
			all_data = all_data.merge(simple.loc[:, ["dayofyear", "flow_id", f"day_{func}_{i}"]].drop_duplicates(),
									  on=["dayofyear", "flow_id"], how="left")
			all_data = all_data.merge(simple.loc[:, ["dayofyear", "flow_id", f"day_{func}_{i}"]].drop_duplicates(),
									  on=["dayofyear", "flow_id"], how="left")

	return all_data, features


def time_features(data_):
	data_["time"] = pd.to_datetime(data_["time"])
	data_["month"] = data_["time"].dt.month
	data_["week"] = data_["time"].dt.week
	data_["day"] = data_["time"].dt.day
	data_["dayofweek"] = data_["time"].dt.dayofweek
	data_["dayofyear"] = data_["time"].dt.dayofyear
	data_["hour"] = data_["time"].dt.hour
	# data_["minute"] = data_["time"].dt.minute
	# data_["second"] = data_["time"].dt.second
	# data_["is_month_end"] = data_["time"].dt.is_month_end
	# data_["is_month_start"] = data_["time"].dt.is_month_start
	data_["is_weekday"] = list(map(lambda x: 1 if x >= 5 else 0, data_["dayofweek"]))
	data_["is_free"] = list(map(lambda x: 1 if x >= 7 and x <= 23 else 0, data_["hour"]))
	features = ["month", "day", "dayofweek", "hour", "is_weekday", "is_free"]
	return data_, features