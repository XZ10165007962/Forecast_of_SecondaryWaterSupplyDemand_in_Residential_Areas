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

import conf
from util import data_cleaning, out_liner, abnormal_data, fill_nan

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
	all_data = all_data.merge(weather_data.loc[:, ["time", "R", "fx", "T", "U", "fs", "V", "P"]], on=["time"],
							  how="left")
	all_data = all_data.merge(epi_data.loc[:, ["day_time", "zz", "wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]],
							  on=["day_time"], how="left")
	data_ = data_.merge(weather_data.loc[:, ["time", "R", "fx", "T", "U", "fs", "V", "P"]], on=["time"],
						how="left")
	data_ = data_.merge(epi_data.loc[:, ["day_time", "zz", "wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]],
						on=["day_time"], how="left")
	del all_data["day_time"]
	del data_["day_time"]
	all_data = abnormal_data(all_data)
	all_data = fill_nan(all_data)
	# all_data["time"] = pd.to_datetime(all_data["time"])
	# all_data["dayofyear"] = all_data["time"].dt.dayofyear

	data_.to_csv(conf.tmp_data_paht + "hour_data.csv", index=False)
	all_data.reset_index(drop=True, inplace=True)
	return all_data


def get_data():
	data_ = get_all_data()
	print(data_.shape)
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	all_data = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		print(f"========={flow}===========")
		if i == 0:
			data = data_[data_["flow_id"] == flow].reset_index(drop=True)
			data = data.loc[:,
				   ["time", "time_index", "flow", "train or test", "flow_id", "R", "fx", "T", "U", "fs", "V", "P", "zz",
					"wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]]
			data = data_cleaning(data)
			data = out_liner(data)
			all_data = data
		else:
			data = data_[data_["flow_id"] == flow].reset_index(drop=True)
			data = data.loc[:,
				   ["time", "time_index", "flow", "train or test", "flow_id", "R", "fx", "T", "U", "fs", "V", "P", "zz",
					"wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]]
			if flow == "flow_19":
				data.loc[[2, 3, 4, 5, 6], ["flow"]] = [0.922571378, 0.774033298, 0.387016649, 0.65820085, 1.105177825]
				# 27.238 / 37.175
				data.loc[[3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063], ["flow"]] = [
					2.35 / (27.238 / 37.175), 4.488 / (27.238 / 37.175), 3.776 / (27.238 / 37.175),
					3.322 / (27.238 / 37.175), 2.35 / (27.238 / 37.175), 2.412 / (27.238 / 37.175),
					2.51 / (27.238 / 37.175), 2.082 / (27.238 / 37.175), 1.364 / (27.238 / 37.175),
					1.258 / (27.238 / 37.175), 1.326 / (27.238 / 37.175)]
				# 21 / 23.976
				data.loc[[3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809], ["flow"]] = [
					2.866 / (21 / 23.976), 2.248 / (21 / 23.976), 2.122 / (21 / 23.976),
					2.04 / (21 / 23.976), 1.492 / (21 / 23.976), 2.3 / (21 / 23.976),
					2.046 / (21 / 23.976), 2.242 / (21 / 23.976), 3.644 / (21 / 23.976)
				]
				# 7.914 / 4.444
				data.loc[[3815, 3816, 3817, 3818, 3819], ["flow"]] = [
					2.754 / (7.914 / 4.444), 1.478 / (7.914 / 4.444), 1.346 / (7.914 / 4.444),
					1.2 / (7.914 / 4.444), 1.136 / (7.914 / 4.444),
				]
				# 22.708 / 23.987
				data.loc[[3811, 3812, 3813], ["flow"]] = [
					6.466 / (22.708 / 23.987), 8.382 / (22.708 / 23.987), 7.86 / (22.708 / 23.987)
				]
			elif flow == "flow_18":
				data.loc[[0, 1, 2, 3, 4, 5], ["flow"]] = [2.902, 2.269, 1.055, 0.677, 0.891, 1.572]
				# 35.372 / 33.82
				data.loc[[3053, 3054, 3055, 3056, 3057], ["flow"]] = [
					4.5 / (35.372 / 33.82), 9.052 / (35.372 / 33.82), 8.424 / (35.372 / 33.82), 7.272 / (35.372 / 33.82), 6.124 / (35.372 / 33.82)]
				# 20.336 / 28.136
				data.loc[[3061, 3062, 3063, 3064, 3065], ["flow"]] = [
					2.056 / (20.336 / 28.136), 2.564 / (20.336 / 28.136), 3.048 / (20.336 / 28.136),
					5.32 / (20.336 / 28.136), 7.348 / (20.336 / 28.136)]
				# 24.159 / 22.04
				data.loc[[3797, 3798, 3799], ["flow"]] = [
					4.459 / (24.159 / 22.04), 9.548 / (24.159 / 22.04), 10.152 / (24.159 / 22.04)]
				#  9.116 / 9.257
				data.loc[[3071, 3072, 3073], ["flow"]] = [
					5.6 / (9.116 / 9.257), 2.416 / (9.116 / 9.257), 1.1 / (9.116 / 9.257)]
				# 15.224 / 15.648
				data.loc[[3801, 3802, 3803], ["flow"]] = [
					5.288 / (15.224 / 15.648), 4.936 / (15.224 / 15.648), 5 / (15.224 / 15.648)]
				# 61.088 / 59.4
				data.loc[[3809, 3810, 3811, 3812, 3813], ["flow"]] = [
					6.496 / (61.088 / 59.4), 8.432 / (61.088 / 59.4), 12.472 / (61.088 / 59.4),
					17.6 / (61.088 / 59.4), 16.088 / (61.088 / 59.4)
				]
			elif flow == "flow_10":
				# 7.607 / 7.932
				data.loc[[3053, 3054, 3055], ["flow"]] = [0.381/(7.607 / 7.932), 3.088/(7.607 / 7.932), 4.128/(7.607 / 7.932)]
				# 3.175 / 5.84
				data.loc[[3061, 3062, 3063], ["flow"]] = [0.879 / (3.175 / 5.84), 1.176 / (3.175 / 5.84),
														  1.12 / (3.175 / 5.84)]
				# 24.748 / 24.161
				data.loc[[3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805], ["flow"]] = [
					2.5 / (24.748 / 24.161), 4.694 / (24.748 / 24.161), 4.24 / (24.748 / 24.161)
				, 2.846 / (24.748 / 24.161), 2.838 / (24.748 / 24.161), 2.822 / (24.748 / 24.161)
				, 2.568 / (24.748 / 24.161), 1.448 / (24.748 / 24.161), 0.792 / (24.748 / 24.161)]
				# 5.848 / 5.756
				data.loc[[3807, 3808, 3809], ["flow"]] = [
					1.298 / (5.848 / 5.756), 1.59 / (5.848 / 5.756), 2.96 / (5.848 / 5.756)]
				# 21.093 / 15.45
				data.loc[[3813, 3814, 3815], ["flow"]] = [
					8.992 / (21.093 / 15.45), 5.112 / (21.093 / 15.45), 6.989 / (21.093 / 15.45)]
				# 22.308 / 24.94
				data.loc[[4997, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005], ["flow"]] = [
					1.56 / (22.308 / 24.946), 3.596 / (22.308 / 24.946), 4.254 / (22.308 / 24.946),
					2.97 / (22.308 / 24.946), 2.604 / (22.308 / 24.946),
					2.4 / (22.308 / 24.946), 1.932 / (22.308 / 24.946), 1.39 / (22.308 / 24.946),
					1.602 / (22.308 / 24.946)
				]
			elif flow == "flow_12":
				# 2.776 / 2.68
				data.loc[[3051, 3052, 3053], ["flow"]] = [0.346 / (2.776 / 2.68), 0.39 / (2.776 / 2.68),
														  2.04 / (2.776 / 2.68)]
				# 15.64 / 20.612
				data.loc[[3059, 3060, 3061, 3062, 3063, 3064, 3065], ["flow"]] = [
					2.9 / (15.64 / 20.612), 2.636 / (15.64 / 20.612), 1.264 / (15.64 / 20.612), 1.584 / (15.64 / 20.612),
					1.892 / (15.64 / 20.612), 2.548 / (15.64 / 20.612), 2.816 / (15.64 / 20.612)]
				# 39.503 / 36.023
				data.loc[[3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809], ["flow"]] = [
					0.223 / (39.503 / 36.023), 0.748 / (39.503 / 36.023), 3.236 / (39.503 / 36.023),
					5.908 / (39.503 / 36.023), 5.252 / (39.503 / 36.023), 3.792 / (39.503 / 36.023),
					2.688 / (39.503 / 36.023), 2.696 / (39.503 / 36.023), 2.88 / (39.503 / 36.023),
				2.488/ (39.503 / 36.023), 2.028 / (39.503 / 36.023), 1.396 / (39.503 / 36.023),1.24 / (39.503 / 36.023),
				2.136 / (39.503 / 36.023),2.792 / (39.503 / 36.023)]
				# 40.37 / 34.9
				data.loc[[3811, 3812, 3813, 3814, 3815], ["flow"]] = [
					5.54 / (40.37 / 34.9), 8.864 / (40.37 / 34.9), 9.416 / (40.37 / 34.9),
					6.828 / (40.37 / 34.9), 9.722 / (40.37 / 34.9)]
				# 33.76 / 31.104
				data.loc[[3857, 3858, 3859, 3860, 3861, 3862, 3863], ["flow"]] = [
					2.932 / (33.76 / 31.104), 4.116 / (33.76 / 31.104), 5.54 / (33.76 / 31.104),
					8.864 / (33.76 / 31.104), 9.416 / (33.76 / 31.104), 6.828 / (33.76 / 31.104),
					9.722 / (33.76 / 31.104)
				]
				linshi = [3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888]
				linshi1 = [3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864]
				data.loc[linshi, ["flow"]] = data.loc[linshi1, ["flow"]]
				data.loc[[3888], ["flow"]] = (376.938 - 2.932 / (33.76 / 31.104)- 4.116 / (33.76 / 31.104)- 5.54 / (33.76 / 31.104)-
					8.864 / (33.76 / 31.104)- 9.416 / (33.76 / 31.104)- 6.828 / (33.76 / 31.104)-
					9.722 / (33.76 / 31.104))
				data.loc[[3888], ["flow"]] = data.loc[[3888], ["flow"]].values - data.loc[linshi, ["flow"]].sum().values
			elif flow == "flow_8":
				# 28.1 / 37.6
				data.loc[[4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003], ["flow"]] = [
					0.6 / (28.1 / 37.6), 0.5 / (28.1 / 37.6), 0.4 / (28.1 / 37.6),
					0.3 / (28.1 / 37.6), 0.5 / (28.1 / 37.6), 1.3 / (28.1 / 37.6),
					3.2 / (28.1 / 37.6), 2.6 / (28.1 / 37.6), 2.8 / (28.1 / 37.6),
					2.7 / (28.1 / 37.6), 1.5 / (28.1 / 37.6), 1.7 / (28.1 / 37.6)
				]
			elif flow == "flow_14":
				# 28.1 / 37.6
				a = [
					1.7,0.5,0.4,0.4,0.4,2,4,4.3,3.2,2.8,2.7,2.7,1.7,1.5,1.4,1.8,1.9,2.9,3.3,5.2,7.2,8.9,6.3,3.8,1,0.6,
					0.2,0.2,0.4,1.6,3.3,4.1,3.1,2.5,3,2.7,1.8,1.3,1.2,1,1.5,2.2,3.9,6,7.8,8.4,6.2,3.7,0.7,0.3,0.4,0.2,
					0.3,1.4,3.1,3.1,3.1,2.4,2.5,2.4,1.9,1.3,1.6,1.9,2.7,3.4,3.6,4.9,6.9,7.5,7.2,3.7,1.4,0.8,0.2,0.6,0.8,
					1.9,3.6,3.3,2.6,2.4,2.2,2.3,1.2,1.5,0.9,1.2,1.8,2.7,3.5,5,7.3,8,6.5,3
				]
				index = [4992+i for i in range(len(a))]
				data.loc[index, ["flow"]] = a

			data = data_cleaning(data)
			data = out_liner(data)
			all_data = pd.concat([all_data, data], axis=0)
	all_data.reset_index(drop=True, inplace=True)
	all_data["pre"] = 0
	print(all_data.shape)
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
	# all_data = get_data()
	all_data.to_csv(conf.tmp_data_paht + "all_data_new.csv", index=False)
	# get_result()