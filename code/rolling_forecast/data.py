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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import conf
from util import data_cleaning, out_liner

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


def all_data():
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
	all_data.reset_index(drop=True,inplace=True)
	return all_data


def get_data(data_, time_index):
	data_ = data_[data_["time_index"] <= time_index]
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	all_data = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		if i == 0:
			data = data_[data_["flow_id"] == flow].reset_index(drop=True)
			data = data.loc[:, ["time", "time_index", "flow", "train or test", "flow_id"]]
			data = data_cleaning(data)
			all_data = data
		else:
			data = data_[data_["flow_id"] == flow].reset_index(drop=True)
			data = data.loc[:, ["time", "time_index", "flow", "train or test", "flow_id"]]
			if flow == "flow_19":
				data.loc[[2, 3, 4, 5, 6], ["flow"]] = [0.922571378, 0.774033298, 0.387016649, 0.65820085, 1.105177825]
			elif flow == "flow_18":
				data.loc[[0, 1, 2, 3, 4, 5], ["flow"]] = [2.902, 2.269, 1.055, 0.677, 0.891, 1.572]
			data = data_cleaning(data)
			all_data = pd.concat([all_data, data], axis=0)
	all_data.reset_index(drop=True,inplace=True)
	all_data["flow_true"] = all_data["flow"]
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
	data = pd.read_csv(conf.tmp_data_paht + "test_data.csv")
	result_data = pd.DataFrame()

	# 循环获取测试结果
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	for flow in flow_id:
		temp = data[data["flow_id"] == flow]
		if flow == "flow_1":
			_temp = temp[temp["train or test"] != "train"]
			result_data["time"] = _temp["time"]
		test1 = temp[temp["train or test"] == "test1"]
		test2_1 = temp[temp["time_index"].isin([
			3625,3626,3627,3628,3629,3630,3631,3632,3633,3634,3635,3636,3637,3638,3639,3640,3641,3642,3643,3644,3645,
			3646,3647,3648,3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,
			3667,3668,3669,3670,3671,3672,3673,3674,3675,3676,3677,3678,3679,3680,3681,3682,3683,3684,3685,3686,3687,
			3688,3689,3690,3691,3692,3693,3694,3695,3696,3697,3698,3699,3700,3701,3702,3703,3704,3705,3706,3707,3708,
			3709,3710,3711,3712,3713,3714,3715,3716,3717,3718,3719,3720,3721,3722,3723,3724,3725,3726,3727,3728,3729,
			3730,3731,3732,3733,3734,3735,3736,3737,3738,3739,3740,3741,3742,3743,3744,3745,3746,3747,3748,3749,3750,
			3751,3752,3753,3754,3755,3756,3757,3758,3759,3760,3761,3762,3763,3764,3765,3766,3767,3768,3769,3770,3771,
			3772,3773,3774,3775,3776,3777,3778,3779,3780,3781,3782,3783,3784,3785,3786,3787,3788,3789])]
		test2_2 = temp[temp["time_index"].isin([3622,3623,3624])]
		test2 = pd.concat([test2_1, test2_2], axis=0)
		test3_1 = temp[temp["time_index"].isin([
			4825,4826,4827,4828,4829,4830,4831,4832,4833,4834,4835,4836,4837,4838,4839,4840,4841,4842,4843,4844,4845,4846,4847,4848,4849,4850,4851,4852,4853,4854,4855,4856,4857,4858,4859,4860,4861,4862,4863,4864,4865,4866,4867,4868,4869,4870,4871,4872,4873,4874,4875,4876,4877,4878,4879,4880,4881,4882,4883,4884,4885,4886,4887,4888,4889,4890,4891,4892,4893,4894,4895,4896,4897,4898,4899,4900,4901,4902,4903,4904,4905,4906,4907,4908,4909,4910,4911,4912,4913,4914,4915,4916,4917,4918,4919,4920,4921,4922,4923,4924,4925,4926,4927,4928,4929,4930,4931,4932,4933,4934,4935,4936,4937,4938,4939,4940,4941,4942,4943,4944,4945,4946,4947,4948,4949,4950,4951,4952,4953,4954,4955,4956,4957,4958,4959,4960,4961,4962,4963,4964,4965,4966,4967,4968,4969,4970,4971,4972,4973,4974,4975,4976,4977,4978,4979,4980,4981,4982,4983,4984,4985,4986,4987,4988,4989])]
		test3_2 = temp[temp["time_index"].isin([4822,4823,4824])]
		test3 = pd.concat([test3_1, test3_2], axis=0)
		test4_1 = temp[temp["time_index"].isin([
			5569,5570,5571,5572,5573,5574,5575,5576,5577,5578,5579,5580,5581,5582,5583,5584,5585,5586,5587,5588,5589,5590,5591,5592,5593,5594,5595,5596,5597,5598,5599,5600,5601,5602,5603,5604,5605,5606,5607,5608,5609,5610,5611,5612,5613,5614,5615,5616,5617,5618,5619,5620,5621,5622,5623,5624,5625,5626,5627,5628,5629,5630,5631,5632,5633,5634,5635,5636,5637,5638,5639,5640,5641,5642,5643,5644,5645,5646,5647,5648,5649,5650,5651,5652,5653,5654,5655,5656,5657,5658,5659,5660,5661,5662,5663,5664,5665,5666,5667,5668,5669,5670,5671,5672,5673,5674,5675,5676,5677,5678,5679,5680,5681,5682,5683,5684,5685,5686,5687,5688,5689,5690,5691,5692,5693,5694,5695,5696,5697,5698,5699,5700,5701,5702,5703,5704,5705,5706,5707,5708,5709,5710,5711,5712,5713,5714,5715,5716,5717,5718,5719,5720,5721,5722,5723,5724,5725,5726,5727,5728,5729])]
		test4_2 = temp[temp["time_index"].isin([5562,5563,5564,5565,5566,5567,5568])]
		test4 = pd.concat([test4_1, test4_2], axis=0)
		__result_data = pd.DataFrame()
		__result_data = pd.concat([test1, test2], axis=0)
		__result_data = pd.concat([__result_data, test3], axis=0)
		__result_data = pd.concat([__result_data, test4], axis=0)
		result_data[flow] = __result_data["flow"].values * 0.9
	result_data.to_csv(conf.predict_data_path + "sub.csv", index=False)


if __name__ == '__main__':
	# all_data = all_data()
	# all_data = get_data(all_data, 2880)
	# all_data.to_csv(conf.tmp_data_paht + "all_data.csv", index=False)
	get_result()