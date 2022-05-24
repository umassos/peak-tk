
import datetime
import numpy as np
import pandas as pd

def scale(data_df, min=0, max=1):
    # 1: Min Max Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(daily_df.values)
    # 2: merge with same day last week X: 48 hr, y: 24 hr
    db_df = pd.DataFrame(scaled_data)
    db_df.columns = daily_df.columns
    X_scaled, y_scaled, idx_scaled = preprocessing.combine_seven_days_before_all(db_df, "Load")

    return (X_scaled, y_scaled, idx_scaled)

def create_dataset(df, previous=1):
    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)

def combine_sameday_week_before_hourly_data(data_df, y_column):

	X = []
	y = []
	idx = []
	column_size = len(data_df.columns)
	for i in range(0, data_df.shape[0]-(7*24), 24):
		a = data_df.iloc[i:i+(1*24)].copy()
		b = data_df.iloc[i+(7*24):i+(8*24)].copy()
		b[y_column] = 0
		temp_x = pd.concat([a,b]).stack().to_frame().T
		temp_x.columns = [str(i) for i in range(0,(24*column_size*2))]
		X.append(temp_x)
		temp_y = data_df.iloc[i+(7*24):i+(8*24),data_df.columns.get_loc(y_column)] # load data 
		y.append(temp_y.values)
		
		idx.append(temp_y.index.values)
#     X = data_df.iloc[0:data_df.shape[0]-7]
	X = pd.concat(X, axis=0)
	y = np.asarray(y, dtype=np.float32)
	return X, y, idx

def combine_sameday_week_before_hourly_data2(data_df, y_column):

	X = []
	y = []
	idx = []
	column_size = len(data_df.columns)
	for i in range(0, data_df.shape[0]-(7*24), 24):
		a = data_df.iloc[i:i+(1*24)].copy()
		b = data_df.iloc[i+(7*24):i+(8*24)].copy()
		temp_y = b[y_column].values
		b[y_column] = 0
		temp_x = pd.concat([a,b]).values
		X.append(temp_x)
		y.append(temp_y)
		idx.append(data_df.iloc[i+(7*24):i+(8*24)].index)
#     X = data_df.iloc[0:data_df.shape[0]-7]
	X = np.asarray(X).astype('float32')
	y = np.asarray(y)
	return X, y, idx

def combine_sameday_week_before_daily_data(data_df, y_index):

	X = []
	y = []
	idx = []
	column_size = len(data_df.columns)
	for i in range(0, data_df.shape[0]-7, 1):
		a = data_df.iloc[i:i+(1)].copy()
		b = data_df.iloc[i+(7):i+(8)].copy()
		b[y_index] = 0
		temp_x = pd.concat([a,b]).stack().to_frame().T
		temp_x.columns = [str(i) for i in range(0,(column_size*2))]
		X.append(temp_x)
		temp_y = data_df[y_index].iloc[i+(7):i+(8)] # load data 
		y.append(temp_y.values)
		
		idx.append(temp_y.index.values)
#     X = data_df.iloc[0:data_df.shape[0]-7]
	X = pd.concat(X, axis=0)
	y = np.asarray(y, dtype=np.float32)
	return X, y, idx


def combine_seven_days_before_all(data_df, y_columns):
	X = []
	y = []
	idx = []

	for i in range(0, data_df.shape[0]-7, 1):
#         print(daily_df.iloc[i+7])
		temp_x = data_df.iloc[i:i+8].copy().values
		temp_x[-1][-1] = 0
		temp_y = data_df.iloc[i+7:i+8][y_columns].copy().values
		X.append(temp_x)
		y.append(temp_y)

		idx.append(data_df.iloc[i+7:i+8].index.values)

	X = np.asarray(X).astype('float32')
	y = np.asarray(y)

	return X, y, idx

def convert_string_to_datetime(dataset):
    expt = 0
    count = 0
    prev_d = None
    dt_list = []
    for idx in dataset.index:
        s = idx.split(" ")
        hr = int(s[-1])-1
        if hr > 23:
            print(dt)
            dt_list.append(None)
            continue
        if expt != hr:
            print(dt)
        expt = (hr + 1) % 24
        dt = s[0] + " " + str(hr)
        s2 = s[0].split("/")
        day = s2[1]
        month = s2[0]
        year = s2[2]
        dt_idx = datetime.datetime.strptime(dt, '%m/%d/%y %H')
        if prev_d != year:
            print(year)
            prev_d = year
            print(count/24)
            count = 0
        count += 1
        dt_list.append(dt_idx)
    
    return dt_list


def convert_hourly_to_daily_data(data_df):
    # convert hourly data to daily data
    daily_df = data_df.groupby(by=data_df.index.date).agg({
        'WCMA DryBulb':['max','min']
        ,'WCMA DewPnt':['max','min']
        ,'CT DryBulb':['max','min']
        ,'CT DewPnt':['max','min']
        ,'NH DryBulb':['max','min']
        ,'NH DewPnt':['max','min']
        ,'MWH Load_amt     ':['max']})
    # daily_df = daily_df.reset_index(level=[0,1])
    daily_df.columns = [
        "Max_WCMADryBulb","Min_WCMADryBulb"
        ,"Max_WCMADewPnt","Min_WCMADewPnt"
        ,"Max_CTDryBulb","Min_CTDryBulb"
        ,"Max_CTDewPnt","Min_CTDewPnt"
        ,"Max_NHDryBulb","Min_NHDryBulb"
        ,"Max_NHDewPnt","Min_NHDewPnt"
        ,"MaxPower"]
    daily_df.index.name = "date"

    return daily_df

def find_monthly_peak(daily_df, k):
    # k: number of peakday per month

    df = daily_df.copy()
    years = set(df.index.year.values)
    months = set(df.index.month.values)
    for year in years:
        for month in months:
            mask = (df.index.month == month) & (df.index.year == year)
            df.loc[mask,"isMonthlyPeak"] = False
            # df.loc[mask,"isMonthlyPeak"] = 
            df.loc[mask, "MonthlyRank"] = df.loc[mask, "MaxPower"].rank(method='first', ascending=False)
            # df[mask].apply(lambda x: x.replace(x.nlargest(k), True), axis=1)
            # df.loc[mask].nlargest(k, "MaxPower")["isMonthlyPeak"] = True
            df.loc[mask & (df["MonthlyRank"] <= k),"isMonthlyPeak"] = True

    return df







