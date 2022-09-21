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

# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)


def MSLE(y, y_hut):
    n = len(y)
    err = 0
    for i, j in zip(y, y_hut):
        err += pow((np.log(1 + i) - np.log(1 + j)), 2)
    return 1 / (err / n + 1)


def data_cleaning(data_):
    print("数据清洗")
    data = data_
    nan_list = []

    def tef():
        ind_min = min(nan_list)
        nan_list.append(ind_min - 1)
        nan_len = len(nan_list)
        num = (nan_len // 24) + 1
        temp_index = [i - 24 * num for i in nan_list]
        if min(nan_list) < 0 or min(temp_index) < 0:
            pass
        else:
            temp1 = data.loc[nan_list, ["flow"]].sum()
            temp2 = data.loc[temp_index, ["flow"]].sum()
            scale = temp1 / temp2
            data_.loc[nan_list, ["flow"]] = (data_.loc[temp_index, ["flow"]] * scale).values

    for ind in data["time_index"]:
        if ind + 1 in data["time_index"]:
            if data.loc[ind - 1, ["flow"]].isna().values and data.loc[ind, ["flow"]].isna().values:
                nan_list.append(ind - 1)
            elif data.loc[ind - 1, ["flow"]].isna().values and ~data.loc[ind, ["flow"]].isna().values:
                nan_list.append(ind - 1)
                tef()
                nan_list = []
        else:
            if data.loc[ind - 1, ["flow"]].isna().values:
                nan_list.append(ind - 1)
                tef()
            elif ~data.loc[ind - 1, ["flow"]].isna().values:
                if nan_list:
                    tef()
                else:
                    pass
    return data_


def out_liner(data_):
    print("异常值处理")
    data = data_
    # 计算四分位数
    Q1 = np.percentile(data["flow"], 25)
    Q3 = np.percentile(data["flow"], 75)
    # 计算IQR Interquartile range 四分位差
    IQR = Q3 - Q1
    # Outlier Step
    outlier_step = 1.5 * IQR

    # 判断离群点
    outlier_index = data[(data["flow"] > Q3 + outlier_step) | (data["flow"] < Q1 - outlier_step)].index

    # 计算一天前该点的index
    replace_index = [i - 24 for i in outlier_index]
    data.loc[outlier_index, ["flow"]] = data.loc[replace_index, ["flow"]].values
    return data
