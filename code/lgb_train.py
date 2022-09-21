#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: tree_train.py
@time: 2022/9/15 11:15
@version:
@desc: 
"""
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

import data
import features
import tree_model
import conf
from util import MSLE

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


if __name__ == '__main__':
    print("获取数据")
    # all_data_ = data.get_data()
    all_data_ = pd.read_csv(conf.tmp_data_paht+"all_data.csv")
    label = ["label"]
    time_index_ = 2160
    for i in range(24*7+1):
        print(f"=================={i}=====================")
        time_index = time_index_ + i
        all_data = all_data_[all_data_["time_index"] <= time_index]
        all_data, feature = features.get_features(all_data)
        if i == 0:
            all_data.to_csv(conf.tmp_data_paht+"feature_data.csv", index=False)
        train_x, train_y, val_x, val_y, test_x, test_y = data.split_data(all_data, "time_index", time_index, "label", feature)
        val_pred, test_pred = tree_model.lgb_model(train_x, train_y, test_x, val_x, val_y)
        # 将预测数据拼接回原始数据
        test_index = test_x.index + 1
        all_data_.loc[test_index, ["flow"]] = test_pred
        # 每一天输出一次损失
        if (i != 0) and (i % 24 == 0):
            err_data = all_data_[(all_data_["time_index"] >= time_index_+1) & (all_data_["time_index"] <= time_index+1)]
            print("测试集mae:", mean_absolute_error(err_data["flow"].values, err_data["flow_true"].values))
            print("测试集mse:", mean_squared_error(err_data["flow"].values, err_data["flow_true"].values))
            print("测试集msle:", MSLE(err_data["flow"].values, err_data["flow_true"].values))
    all_data.to_csv(conf.tmp_data_paht + "pre_data.csv", index=False)