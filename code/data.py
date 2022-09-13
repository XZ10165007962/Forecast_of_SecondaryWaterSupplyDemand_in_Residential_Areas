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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)


data = pd.read_csv("../data/use_data/training_dataset/daily_dataset.csv")
stand = StandardScaler()
data1 = data.loc[:, ["flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6"]]
data1 = stand.fit_transform(data1)
data1 = pd.DataFrame(data1,columns=["flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6"])
data1.plot.line()
plt.show()
# for i in [
#     ["flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7"],
#     ["flow_8", "flow_9", "flow_10", "flow_11", "flow_12", "flow_13", "flow_14"],
#     ["flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"]]:
#     data1 = data.loc[:, i]
#     data1.plot.line()
#     plt.show()