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

from code.direct_prediction import conf

# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)


def MSLE(y, y_hut, flag=0):
    n = len(y)
    err = 0
    for i, j in zip(y, y_hut):
        err += pow((np.log(1 + i) - np.log(1 + j)), 2)
    if flag:
        return 1 / (err / n * 20 + 1)
    else:
        return 1 / (err / n + 1)


def data_cleaning(data_):
    data = data_[data_["train or test"] == "train"]
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
            temp1 = data_.loc[nan_list, ["flow"]].sum()
            temp2 = data_.loc[temp_index, ["flow"]].sum()
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
                nan_list = []
            elif ~data.loc[ind - 1, ["flow"]].isna().values:
                if nan_list:
                    tef()
                    nan_list = []
                else:
                    pass
    return data_


def out_liner(data_):
    print("异常值处理")
    flow_id = data_["flow_id"].unique()[0]
    if flow_id == "flow_4":
        # 192.8 / 467.1
        use_index = data_[data_["time_index"] == 4329].index
        data_.loc[use_index, ["flow"]] = 39.7 / (192.8 / 467.1)
    if flow_id == "flow_8":
        # 0.7 / 0.7
        use_index = data_[data_["time_index"] == 147].index
        data_.loc[use_index - 1, ["flow"]] = 0.4
        data_.loc[use_index, ["flow"]] = 0.3
        # 5.8 / 8.3
        use_index = data_[data_["time_index"] == 177].index
        data_.loc[use_index - 1, ["flow"]] = 3.2 / (5.8 / 8.3)
        data_.loc[use_index, ["flow"]] = 2.6 / (5.8 / 8.3)
        # 37.4 / 43.4
        use_index = data_[data_["time_index"] == 179].index
        data_.loc[use_index, ["flow"]] = 2.7 / (37.4 / 43.4)
        data_.loc[use_index + 1, ["flow"]] = 37.4 / (37.4 / 43.4)
        # 37.4 / 45.6
        use_index = data_[data_["time_index"] == 203].index
        data_.loc[use_index, ["flow"]] = 2.7 / (37.4 / 45.6)
        data_.loc[use_index + 1, ["flow"]] = 37.4 / (37.4 / 45.6)
        # - / 4.5
        use_index = data_[data_["time_index"] == 227].index
        data_.loc[use_index, ["flow"]] = 2.7
        data_.loc[use_index + 1, ["flow"]] = 1.8
    if flow_id == "flow_11":
        # 2.1 / 2.5
        use_index = data_[data_["time_index"] == 147].index
        data_.loc[use_index - 1, ["flow"]] = 1.4 / (2.1 / 2.5)
        data_.loc[use_index, ["flow"]] = 0.7 / (2.1 / 2.5)
        # 2.5 / 2.7
        use_index = data_[data_["time_index"] == 149].index
        data_.loc[use_index - 1, ["flow"]] = 1.3 / (2.5 / 2.7)
        data_.loc[use_index, ["flow"]] = 1.2 / (2.5 / 2.7)
        # 2.5 / 2.7
        use_index = data_[data_["time_index"] == 179].index
        data_.loc[use_index - 1, ["flow"]] = 1.3 / (2.5 / 2.7)
        data_.loc[use_index, ["flow"]] = 1.2 / (2.5 / 2.7)
        # 102.8 / 119
        use_index = data_[data_["time_index"] == 203].index
        data_.loc[use_index, ["flow"]] = 73 / (102.8 / 119)
        data_.loc[use_index + 1, ["flow"]] = 95 / (102.8 / 119)
        # 119 / 133
        use_index = data_[data_["time_index"] == 227].index
        data_.loc[use_index, ["flow"]] = 73 / (102.8 / 119) / (119 / 133)
        data_.loc[use_index + 1, ["flow"]] = 95 / (102.8 / 119) / (119 / 133)
    if flow_id == "flow_12":
        use_index = data_[data_["time_index"] == 4096].index
        data_.loc[use_index, ["flow"]] = 0
    if flow_id == "flow_13":
        # 50.4 / 43.3
        use_index = data_[data_["time_index"] == 155].index
        data_.loc[use_index, ["flow"]] = 3 / (50.4 / 43.3)
        data_.loc[use_index + 1, ["flow"]] = 47.4 / (50.4 / 43.3)
        # 7.5 / 6.9
        use_index = data_[data_["time_index"] == 177].index
        data_.loc[use_index - 1, ["flow"]] = 4 / (7.5 / 6.9)
        data_.loc[use_index, ["flow"]] = 3.5 / (7.5 / 6.9)
        # 43.3 / 46.9
        use_index = data_[data_["time_index"] == 179].index
        data_.loc[use_index, ["flow"]] = 3 / (50.4 / 43.3) / (43.3 / 46.9)
        data_.loc[use_index + 1, ["flow"]] = 47.4 / (50.4 / 43.3) / (43.3 / 46.9)
        # 46.9 / 53.3
        use_index = data_[data_["time_index"] == 203].index
        data_.loc[use_index, ["flow"]] = 3 / (50.4 / 43.3) / (43.3 / 46.9) / (46.9 / 53.3)
        data_.loc[use_index + 1, ["flow"]] = 47.4 / (50.4 / 43.3) / (43.3 / 46.9) / (46.9 / 53.3)
        # 53.3 / 45.8
        use_index = data_[data_["time_index"] == 227].index
        data_.loc[use_index, ["flow"]] = 3 / (50.4 / 43.3) / (43.3 / 46.9) / (46.9 / 53.3) / (53.3 / 45.8)
        data_.loc[use_index + 1, ["flow"]] = 47.4 / (50.4 / 43.3) / (43.3 / 46.9) / (46.9 / 53.3) / (53.3 / 45.8) - 40.7
    if flow_id == "flow_14":
        # 0.5 / 0.4
        use_index = data_[data_["time_index"] == 149].index
        data_.loc[use_index - 1, ["flow"]] = 0.2 / (0.5 / 0.4)
        data_.loc[use_index, ["flow"]] = 0.3 / (0.5 / 0.4)
        # 0.36 / 0.4
        use_index = data_[data_["time_index"] == 172].index
        data_.loc[use_index - 1, ["flow"]] = 0.2 / (0.36 / 0.4)
        data_.loc[use_index, ["flow"]] = 0.16 / (0.36 / 0.4)
        # 59.4 / 61.4
        use_index = data_[data_["time_index"] == 203].index
        data_.loc[use_index, ["flow"]] = 3.2 / (59.4 / 61.4)
        data_.loc[use_index + 1, ["flow"]] = 56.2 / (59.4 / 61.4)
        # 61.4 / 52.9
        use_index = data_[data_["time_index"] == 227].index
        data_.loc[use_index, ["flow"]] = 3.2 / (59.4 / 61.4) / (61.4 / 52.9)
        data_.loc[use_index + 1, ["flow"]] = 56.2 / (59.4 / 61.4) / (61.4 / 52.9) - 46.7
        # 39.1 / 33.8
        use_index = data_[data_["time_index"] == 2416].index
        data_.loc[use_index, ["flow"]] = 2.6 / (39.1 / 33.8)
        data_.loc[use_index + 1, ["flow"]] = 2.8 / (39.1 / 33.8)
    if flow_id == "flow_15":
        # 1.3 / 2.3
        use_index = data_[data_["time_index"] == 149].index
        data_.loc[use_index - 1, ["flow"]] = 0.4 / (1.3 / 2.3)
        data_.loc[use_index, ["flow"]] = 0.9 / (1.3 / 2.3)
        # 37.6 / 38.6
        use_index = data_[data_["time_index"] == 179].index
        data_.loc[use_index, ["flow"]] = 2.5 / (37.6 / 38.6)
        data_.loc[use_index + 1, ["flow"]] = 35.1 / (37.6 / 38.6)
        # 38.6 / 44.9
        use_index = data_[data_["time_index"] == 203].index
        data_.loc[use_index, ["flow"]] = 2.5 / (37.6 / 38.6) / (38.6 / 44.9)
        data_.loc[use_index + 1, ["flow"]] = 35.1 / (37.6 / 38.6) / (38.6 / 44.9)
        # 44.9 / 35.2
        use_index = data_[data_["time_index"] == 227].index
        data_.loc[use_index, ["flow"]] = 2.5 / (37.6 / 38.6) / (38.6 / 44.9) / (44.9 / 35.2)
        data_.loc[use_index + 1, ["flow"]] = 35.1 / (37.6 / 38.6) / (38.6 / 44.9) / (44.9 / 35.2) - 31.9
    if flow_id == "flow_16":
        # 1.7 / 1.7
        use_index = data_[data_["time_index"] == 149].index
        data_.loc[use_index - 1, ["flow"]] = 0.6
        data_.loc[use_index, ["flow"]] = 1.1
        # 12.3 / 16.3
        use_index = data_[data_["time_index"] == 177].index
        data_.loc[use_index - 1, ["flow"]] = 6.5 / (12.3 / 16.3)
        data_.loc[use_index, ["flow"]] = 5.8 / (12.3 / 16.3)
        # 87 / 99.9
        use_index = data_[data_["time_index"] == 179].index
        data_.loc[use_index, ["flow"]] = 5.4 / (87 / 99.9)
        data_.loc[use_index + 1, ["flow"]] = 81.6 / (87 / 99.9)
        # 99.9 / 108
        use_index = data_[data_["time_index"] == 203].index
        data_.loc[use_index, ["flow"]] = 5.4 / (87 / 99.9) / (99.9 / 108)
        data_.loc[use_index + 1, ["flow"]] = 81.6 / (87 / 99.9) / (99.9 / 108)
        # 108 / 91.5
        use_index = data_[data_["time_index"] == 227].index
        data_.loc[use_index, ["flow"]] = 5.4 / (87 / 99.9) / (99.9 / 108) / (108 / 91.5)
        data_.loc[use_index + 1, ["flow"]] = 81.6 / (87 / 99.9) / (99.9 / 108) / (108 / 91.5) - 81.7
    if flow_id == "flow_17":
        # 7 / 7.3
        use_index = data_[data_["time_index"] == 150].index
        data_.loc[use_index - 1, ["flow"]] = 1.7 / (7 / 7.3)
        data_.loc[use_index, ["flow"]] = 5.3 / (7 / 7.3)
        # 25.2 / 26
        use_index = data_[data_["time_index"] == 177].index
        data_.loc[use_index - 1, ["flow"]] = 13.6 / (25.2 / 26)
        data_.loc[use_index, ["flow"]] = 11.6 / (25.2 / 26)
        # 146.5 / 170.4
        use_index = data_[data_["time_index"] == 179].index
        data_.loc[use_index, ["flow"]] = 8.8 / (146.5 / 170.4)
        data_.loc[use_index + 1, ["flow"]] = 137.7 / (146.5 / 170.4)
        # 170.4 / 193.1
        use_index = data_[data_["time_index"] == 203].index
        data_.loc[use_index, ["flow"]] = 8.8 / (146.5 / 170.4) / (170.4 / 193.1)
        data_.loc[use_index + 1, ["flow"]] = 137.7 / (146.5 / 170.4) / (170.4 / 193.1)
        # 193.1 / 156.9
        use_index = data_[data_["time_index"] == 227].index
        data_.loc[use_index, ["flow"]] = 8.8 / (146.5 / 170.4) / (170.4 / 193.1) / (193.1 / 156.9)
        data_.loc[use_index + 1, ["flow"]] = 137.7 / (146.5 / 170.4) / (170.4 / 193.1) / (193.1 / 156.9) - 135.6
    if flow_id == "flow_18":
        use_index = data_[data_["time_index"] == 2415].index
        data_.loc[use_index, ["flow"]] = 5.976 / (77.313 / 63.685)
    if flow_id == "flow_19":
        # 2.053 / 2.261
        use_index = data_[data_["time_index"] == 1133].index
        data_.loc[use_index, ["flow"]] = 0.401 / (2.053 / 2.261)
        data_.loc[use_index + 1, ["flow"]] = 1.184 / (2.053 / 2.261)
        # 1.604 / 1.466
        use_index = data_[data_["time_index"] == 1348].index
        data_.loc[use_index, ["flow"]] = 0.198 / (1.604 / 1.466)
        data_.loc[use_index + 1, ["flow"]] = 0.338 / (1.604 / 1.466)
        # 40.97 / 40.267
        use_index = data_[data_["time_index"] == 2392].index
        data_.loc[use_index, ["flow"]] = 2.204 / (40.97 / 40.267)
        data_.loc[use_index + 1, ["flow"]] = 2.858 / (40.97 / 40.267)
        # 2.263 / 1.843
        use_index = data_[data_["time_index"] == 2548].index
        data_.loc[use_index, ["flow"]] = 0.252 / (2.263 / 1.843)
        data_.loc[use_index + 1, ["flow"]] = 0.371 / (2.263 / 1.843)
    return data_


def abnormal_data(data_):
    columns = data_.columns
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
    return data_[columns]


def fill_nan(data_):
    columns = data_.columns
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
    return data_[columns]


if __name__ == '__main__':
    data = pd.read_csv(conf.tmp_data_paht + "all_data.csv")
    print(data.head())
    fill_nan(data)