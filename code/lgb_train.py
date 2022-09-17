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
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from keras.losses import MeanSquaredLogarithmicError

import data
import features
import tree_model

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)


if __name__ == '__main__':
    print("获取数据")
    all_data_ = data.get_data()
    label = ["label"]
    time_index_ = 2160
    for i in range(25):
        print(f"=================={i}=====================")
        time_index = time_index_ + i
        all_data, feature = features.get_features(all_data_)
        train_x, train_y, val_x, val_y, test_x, test_y = data.split_data(all_data, "time_index", time_index, "label", feature)
        # print(train_x.head())
        # print(train_y.head())
        # print(val_x.head())
        # print(val_y.head())
        # print(test_x.head())
        # print(test_y.head())
        val_pred, test_pred = tree_model.lgb_model(train_x, train_y, test_x, val_x, val_y)
        print("测试集mae:", mean_absolute_error(test_y, test_pred))
        print("测试集mse:", mean_squared_error(test_y, test_pred))
        # print("测试集msge:", MeanSquaredLogarithmicError(test_y, test_pred))
        # 将预测数据拼接回原始数据
        test_index = test_x.index + 1
        print(test_index)
        all_data_.loc[test_index, ["flow"]] = test_pred
        print(all_data_.loc[test_index, ["flow"]])