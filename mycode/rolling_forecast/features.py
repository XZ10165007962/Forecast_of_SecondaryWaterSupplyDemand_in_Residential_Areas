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

from mycode.rolling_forecast import conf

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def get_features(all_data, lag=24, rolling=2):
	print("特征生成")
	features = []
	all_data, time_feature = time_features(all_data)
	features.extend(time_feature)
	# 获取滞后特征
	for i in range(24, 43):
		all_data[f"flow_lag_{i}"] = all_data.groupby(["flow_id"])["flow"].shift(i)
		features.append(f"flow_lag_{i}")
		all_data[f"flow_lag_{i}_roll"] = all_data.groupby(["flow_id"])["flow"].shift(i).rolling(rolling).mean()
		features.append(f"flow_lag_{i}_roll")
	# 差分
	all_data["id_diff"] = all_data.groupby(["flow_id"])["flow"].shift(24).diff()
	# 同比
	week_lag = 3
	for i in range(1, week_lag+1):
		if i >= 2:
			all_data[f"flow_lag_{24 * i - 1}"] = all_data.groupby(["flow_id"])["flow"].shift(24 * i - 1)
			all_data[f"flow_lag_{24 * i}"] = all_data.groupby(["flow_id"])["flow"].shift(24 * i)
			all_data[f"flow_lag_{24 * i + 1}"] = all_data.groupby(["flow_id"])["flow"].shift(24 * i + 1)
			features.append(f"flow_lag_{24 * i - 1}")
			features.append(f"flow_lag_{24 * i}")
			features.append(f"flow_lag_{24 * i + 1}")

			all_data[f"flow_lag_{24 * i - 1}_{24 * i}_{24 * i + 1}_mean"] = \
				all_data.loc[:, [f"flow_lag_{24 * i - 1}", f"flow_lag_{24 * i}", f"flow_lag_{24 * i + 1}"]].mean(axis=1)
			features.append(f"flow_lag_{24 * i - 1}_{24 * i}_{24 * i + 1}_mean")
			all_data[f"flow_lag_{24 * i}_{24 * i + 1}_mean"] = \
				all_data.loc[:, [f"flow_lag_{24 * i}", f"flow_lag_{24 * i + 1}"]].mean(axis=1)
			features.append(f"flow_lag_{24 * i}_{24 * i + 1}_mean")
		else:
			all_data[f"flow_lag_{24 * i}_{24 * i + 1}_mean"] = \
				all_data.loc[:, [f"flow_lag_{24 * i}", f"flow_lag_{24 * i + 1}"]].mean(axis=1)
			features.append(f"flow_lag_{24 * i}_{24 * i + 1}_mean")

	tongbi_col = [f"flow_lag_{24 * i}" for i in range(1, week_lag+1)]
	funcs = ["mean", "sum", "median", "max", "min", "std"]
	for func in funcs:
		all_data[f"tongbi_{func}"] = all_data[tongbi_col].apply(func, axis=1)
		features.append(f"tongbi_{func}")
	# 计算统计特征
	flag = 3
	for func in funcs:
		all_data[f"flow_id_{func}"] = all_data.groupby(["flow_id"])["flow"].transform(func)
		features.append(f"flow_id_{func}")
	for func in funcs:
		simple = copy.deepcopy(all_data)
		simple[f"month_{func}"] = simple.groupby(["month", "flow_id"])["flow"].transform(func)
		simple[f"month_weekday_{func}"] = simple.groupby(["month", "is_weekday", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple[f"month_{func}_{i}"] = simple[f"month_{func}"]
			simple[f"month_weekday_{func}_{i}"] = simple[f"month_weekday_{func}"]
			features.append(f"month_{func}_{i}")
			features.append(f"month_weekday_{func}_{i}")
			simple["month"] = list(map(lambda x: x + i, simple["month"]))
			all_data = all_data.merge(simple.loc[:, ["month", "flow_id", f"month_{func}_{i}"]].drop_duplicates(),
									  on=["month", "flow_id"], how="left")
			all_data = all_data.merge(
				simple.loc[:, ["month", "is_weekday", "flow_id", f"month_weekday_{func}_{i}"]].drop_duplicates(),
				on=["month", "is_weekday", "flow_id"], how="left")
	for func in funcs:
		simple = copy.deepcopy(all_data)
		simple[f"week_{func}"] = simple.groupby(["week", "flow_id"])["flow"].transform(func)
		simple[f"week_free_{func}"] = simple.groupby(["week", "is_free", "flow_id"])["flow"].transform(func)
		simple[f"week_weekday_{func}"] = simple.groupby(["week", "is_weekday", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple[f"week_{func}_{i}"] = simple[f"week_{func}"]
			simple[f"week_free_{func}_{i}"] = simple[f"week_free_{func}"]
			simple[f"week_weekday_{func}_{i}"] = simple[f"week_weekday_{func}"]
			features.append(f"week_{func}_{i}")
			features.append(f"week_free_{func}_{i}")
			features.append(f"week_weekday_{func}_{i}")
			simple["week"] = list(map(lambda x: x + i, simple["week"]))
			all_data = all_data.merge(simple.loc[:, ["week", "flow_id", f"week_{func}_{i}"]].drop_duplicates(),
									  on=["week", "flow_id"], how="left")
			all_data = all_data.merge(
				simple.loc[:, ["week", "flow_id", "is_free", f"week_free_{func}_{i}"]].drop_duplicates(),
				on=["week", "is_free", "flow_id"], how="left")
			all_data = all_data.merge(
				simple.loc[:, ["week", "flow_id", "is_weekday", f"week_weekday_{func}_{i}"]].drop_duplicates(),
				on=["week", "is_weekday", "flow_id"], how="left")
	for func in funcs:
		simple = copy.deepcopy(all_data)
		simple[f"day_{func}"] = simple.groupby(["dayofyear", "flow_id"])["flow"].transform(func)
		simple[f"day_free_{func}"] = simple.groupby(["dayofyear", "is_free", "flow_id"])["flow"].transform(func)
		for i in range(1, flag):
			simple[f"day_{func}_{i}"] = simple[f"day_{func}"]
			simple[f"day_free_{func}_{i}"] = simple[f"day_free_{func}"]
			features.append(f"day_{func}_{i}")
			features.append(f"day_free_{func}_{i}")
			simple["dayofyear"] = list(map(lambda x: x+i, simple["dayofyear"]))
			all_data = all_data.merge(simple.loc[:, ["dayofyear", "flow_id", f"day_{func}_{i}"]].drop_duplicates(),
									  on=["dayofyear", "flow_id"], how="left")
			all_data = all_data.merge(simple.loc[:, ["dayofyear", "is_free", "flow_id", f"day_free_{func}_{i}"]].drop_duplicates(),
									  on=["dayofyear", "is_free", "flow_id"], how="left")
	all_data["week_gep"] = all_data["week_mean_1"] / all_data["week_mean_2"]
	all_data["month_gep"] = all_data["month_mean_1"] / all_data["month_mean_2"]
	all_data["day_gep"] = all_data["day_mean_1"] / all_data["day_mean_2"]
	all_data["week_gep_pre"] = all_data["week_gep"] * all_data["flow_lag_24"]
	all_data["month_gep_pre"] = all_data["month_gep"] * all_data["flow_lag_24"]
	all_data["day_gep_pre"] = all_data["day_gep"] * all_data["flow_lag_24"]
	features.extend(["week_gep", "month_gep", "day_gep", "week_gep_pre", "month_gep_pre", "day_gep_pre"])

	all_data["flow_jiaqi"] = all_data.groupby(["flow_id", "jiaqi"])["flow"].transform("mean")
	all_data["flow_jiaqi_latter"] = all_data.groupby(["flow_id", "jiaqi_latter"])["flow"].transform("mean")
	all_data["flow_jiaqi_after"] = all_data.groupby(["flow_id", "jiaqi_after"])["flow"].transform("mean")

	cats = ["is_weekday", "is_free", "jiaqi", "jiaqi_after", "jiaqi_latter"]
	for cat in cats:
		all_data[cat] = all_data[cat].astype('category')
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
	features = ["month", "day", "dayofweek", "hour", "is_weekday", "is_free", "dayofyear"]
	jiaqibiao = [1,2,3,31,32,33,34,35,36,37,93,94,95,120,121,122,123,124,154,155,156]
	jiaqibiao_after = [30, 92, 119, 153]
	jiaqibiao_latter = [4, 38, 96, 125, 157]
	data_["jiaqi"] = list(map(lambda x: 1 if x in jiaqibiao else 0, data_["dayofyear"]))
	data_["jiaqi_after"] = list(map(lambda x: 1 if x in jiaqibiao_after else 0, data_["dayofyear"]))
	data_["jiaqi_latter"] = list(map(lambda x: 1 if x in jiaqibiao_latter else 0, data_["dayofyear"]))
	features.extend(["jiaqi", "jiaqi_after", "jiaqi_latter"])
	return data_, features

if __name__ == '__main__':
	all_data_ = pd.read_csv(conf.tmp_data_paht + "all_data.csv")
	data, features = get_features(all_data_)
	data.to_csv(conf.tmp_data_paht + "feature_data.csv", index=False)
	# all_data_ = pd.read_csv(conf.tmp_data_paht + "feature_data.csv")
	# print(all_data_.corr())