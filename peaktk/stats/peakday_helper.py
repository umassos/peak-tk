
import pandas as pd
import numpy as np


def identify_monthly_peak(monthly_demand, top_k):
		
	temp_df = pd.DataFrame(monthly_demand)
	temp_df.columns = ["gt_demand"]
	temp_df["gt_peak"] = False
	temp_df.loc[(temp_df.index.isin(temp_df.nlargest(top_k,'gt_demand').index)),"gt_peak"] = True

	return temp_df["gt_peak"].tolist()
#         

def identify_yearly_peak(data_df, k):
	gt_peak = []
	for y in np.unique(data_df.index.year):
		yearly_demand = data_df[data_df.index.year == y]["Load"].tolist()
		gt_peak += identify_top_k(yearly_demand, k)
	return gt_peak

def identify_top_k(demand, k):
	temp_df = pd.DataFrame(demand)
	temp_df.columns = ["demand"]
	temp_df["gt_peak"] = False
	temp_df.loc[(temp_df.index.isin(temp_df.nlargest(k,'demand').index)),"gt_peak"] = True

	return temp_df["gt_peak"].tolist()
#         



