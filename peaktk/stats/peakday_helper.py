
import pandas as pd
import numpy as np


def find_monthly_peak(y_test, idx_test, top_k):
	
	year_df = pd.DataFrame(y_test, idx_test)
	y_true = []
	for month in range(1, 13):
	    monthly_df = year_df[(year_df.index.month==month)]
	    y_true += get_monthly_peak(monthly_df[0].tolist(), top_k)
	return y_true

# def identify_monthly_peak(data_df, current_year=2020):
# 	y_true = []
# 	for month in range(1, 13):
# 	    monthly_df = year_df[(year_df.index.month==month)]
# 	    y_true += peakday_helper.identify_monthly_peak(monthly_df["MaxPower"].tolist(), 3)
	    

def get_monthly_peak(monthly_demand, top_k):
		
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
def find_peak_hour_gt(y_data, num_peak_hours):
	# y_data: list
	# num_peak_hours: int
    result_list = []
    for y in y_data:
        idx = np.argpartition(y, -num_peak_hours)[-num_peak_hours:]
        result = np.array([False]*24)
        result[idx] = True
        result_list.append(result)
    return np.array(result_list)


