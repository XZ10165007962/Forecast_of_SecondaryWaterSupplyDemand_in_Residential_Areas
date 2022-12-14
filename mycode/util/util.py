#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: util.py
@time: 2022/10/8 14:21
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

from mycode.util import conf

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


def MSLE(y, y_hut, flag=0):
    n = len(y)
    err = 0
    for i, j in zip(y, y_hut):
        err += pow((np.log(1 + i) - np.log(1 + j)), 2)
    if flag:
        return 1 / (err / n * 20 + 1)
    else:
        return 1 / (err / n + 1)


def abnormal_data(data):
    data = data.reset_index(drop=True)
    data_ = data[data["train or test"] == "train"]
    def q1(x):
        return x.quantile(0.25)
    def q2(x):
        return x.quantile(0.75)
    print("异常值识别")
    data_["time"] = pd.to_datetime(data_["time"])
    data_["month"] = data_["time"].dt.month
    data_["hour"] = data_["time"].dt.hour
    data_["id_hour_std"] = data_.groupby(["month", "hour", "flow_id"])["flow"].transform("std")
    data_["id_hour_mean"] = data_.groupby(["month", "hour", "flow_id"])["flow"].transform("mean")
    data_["id_hour_q1"] = data_.groupby(["month", "hour", "flow_id"])["flow"].transform(q1)
    data_["id_hour_q3"] = data_.groupby(["month", "hour", "flow_id"])["flow"].transform(q2)
    data_["lower_1"] = data_["id_hour_mean"] - 3 * data_["id_hour_std"]
    data_["upper_1"] = data_["id_hour_mean"] + 3 * data_["id_hour_std"]
    data_["lower_2"] = data_["id_hour_q1"] - 1.5*(data_["id_hour_q3"] - data_["id_hour_q1"])
    data_["upper_2"] = data_["id_hour_q3"] + 1.5*(data_["id_hour_q3"] - data_["id_hour_q1"])
    data_["lower"] = list(map(lambda x,y: x if x > y else y, data_["lower_1"], data_["lower_2"]))
    data_["upper"] = list(map(lambda x,y: x if x < y else y, data_["upper_1"], data_["upper_2"]))
    data_["outer"] = list(map(lambda x,y,z: 1 if x < y or x > z else 0, data_["flow"], data_["lower"], data_["upper"]))
    data_["flow"] = list(map(lambda x,y: np.nan if x == 1 else y, data_["outer"], data_["flow"]))
    index = data_.index
    data.loc[index, ["flow"]] = data_["flow"]
    return data


def fill_nan(data):
    data = data.reset_index(drop=True)
    data_ = data[data["train or test"] == "train"]
    print("数据填充")
    fillna_col = []
    for i in range(1, 29):
        data_[f"flow_lag_{i}"] = data_.groupby(["flow_id"])["flow"].shift(24*i)
        fillna_col.append(f"flow_lag_{i}")
    for i in range(1, 29):
        data_[f"flow_lag_{-i}"] = data_.groupby(["flow_id"])["flow"].shift(-24*i)
        fillna_col.append(f"flow_lag_{-i}")
    data_["fill_data"] = data_[fillna_col].mean(axis=1)
    data_["fill_flag"] = pd.isna(data_["flow"])
    data_["flow"] = list(map(lambda x,y,z: z if y else x, data_["flow"], data_["fill_flag"], data_["fill_data"]))
    index = data_.index
    data.loc[index, ["flow"]] = data_["flow"]
    return data


if __name__ == '__main__':
    data = pd.read_csv(conf.tmp_data_paht + "all_data.csv")
    print(data.head())
    fill_nan(data)