#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: features.py
@time: 2022/9/15 10:44
@version:
@desc: 
"""
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


def get_features(data_, lag=1, shift=1):
	time_features(data_)


def time_features(data_):
	data_["time"] = pd.to_datetime(data_["time"])
	data_["month"] = data_["time"].dt.month
	data_["week"] = data_["time"].dt.week
	data_["day"] = data_["time"].dt.month
	data_["dayofweek"] = data_["time"].dt.dayofweek
	data_["hour"] = data_["time"].dt.hour
	data_["minute"] = data_["time"].dt.minute
	data_["second"] = data_["time"].dt.second
	data_["is_month_end"] = data_["time"].dt.is_month_end
	data_["is_month_start"] = data_["time"].dt.is_month_start
	data_["is_weekday"] = list(map(lambda x: 1 if x >= 5 else 0, data_["dayofweek"]))
	data_["is_free"] = list(map(lambda x:))
	print(data_.head())