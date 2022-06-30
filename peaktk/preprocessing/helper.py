# Author: Nikko 2021

import numpy as np
import pandas as pd
import glob
import datetime
from peaktk.preprocessing import preprocessing
from sklearn.preprocessing import MinMaxScaler


def train_test_split(X, y, idx, test_size=365):
    # train
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    idx_train = idx[:-test_size]
    # test
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    idx_test = idx[-test_size:]
    return (X_train, X_test, y_train, y_test, idx_train, idx_test)

# def train_test_split(daily_df):
#     # Pre-processing
#     # 1: Min Max Scale
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(daily_df.values)
#     # 2: merge with same day last week X: 48 hr, y: 24 hr
#     db_df = pd.DataFrame(scaled_data)
#     db_df.columns = daily_df.columns
#     X_scaled, y_scaled, idx_scaled = preprocessing.combine_seven_days_before_all(db_df, "Load")
#     X, y, idx = preprocessing.combine_seven_days_before_all(daily_df, "Load")

#     return (X_scaled, y_scaled, idx_scaled, X, y, idx, scaler)


    # X_train = X_scaled[:-366]
    # y_train = y_scaled[:-366]
    # X_test = X_scaled[-366:]
    # y_test = y_scaled[-366:]

    # return (X_train, y_train, X_test, y_test)


def merge_ISONE_dataset(isone_path, weather_path):
    all_files = glob.glob(isone_path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files))

    isone_df = pd.read_csv(isone_file)
    weather_df = pd.read_csv(weather_file)

def create_all_features(dataset):
    # load feature
    # drop columns
    # weather features
    features = ['Load', 'temperature', 'humidity',
       'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover',
       'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability']
    df = dataset.copy()
    df = df[features]
    # calendar features
    df = add_calendar_data(df)
    return df


def add_calendar_data(dataset):
    # read index for date
    df = dataset.copy()
    df["dayofweek"] = df.index.weekday.tolist()
    df["dayofyear"] = df.index.strftime("%j").astype('int64').tolist()
    df["week"] = df.index.strftime("%W").astype('int64').tolist()
    df["month"] = df.index.month.tolist()
    df["hourofday"] = df.index.hour.tolist()
    return df

def create_X_y_same_day_last_week(df):
    dataX, dataY = [], []
    for i in range(len(df)-7-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)

def create_X_y_previous_days(df, previous=1):
    dataX, dataY = [], []
    for i in range(df.shape[0]-previous-1):
        a = data_df.iloc[i:(i+previous)].copy()
        b = data_df.iloc[(i+previous):(i+previous+1)].copy()
        b.iloc[0:1,-24:] = 0
        temp_x = pd.concat([a,b]).stack().to_frame().T

        dataX.append(temp_x)
        dataY.append(df.iloc[i + previous, y_column])
    return np.array(dataX), np.array(dataY)

def combine_sameday_week_before_hourly_data(data_df, y_index):

    X = []
    y = []
    idx = []
    for idx,value in data_df.groupby(by=data_df.index.date).count().iterrows():
        temp_df = data_df[data_df.index.date == idx].copy()
        count = temp_df.shape[0]
        if count == 23:
            # add one
            temp_df = temp_df.append(temp_df.iloc[22], ignore_index=True)
        if count == 25:
            # remove one
            temp_df = temp_df.drop(index=temp_df.iloc[24:25].index)

        a = data_df.iloc[i:i+(1*24)].copy()
        b = data_df.iloc[i+(7*24):i+(8*24)].copy()
        b[y_index] = 0
        temp_x = pd.concat([a,b]).stack().to_frame().T
        temp_x.columns = [str(i) for i in range(0,(24*column_size*2))]
        X.append(temp_x)
        temp_y = data_df.iloc[i+(7*24):i+(8*24),y_index] # load data 
        y.append(temp_y.values)
        
        idx.append(temp_y.index.values)
#     X = data_df.iloc[0:data_df.shape[0]-7]
    X = pd.concat(X, axis=0)
    y = np.asarray(y, dtype=np.float32)
    return X, y, idx

def convert_string_to_datetime(dataset):
    expt = 0
    count = 0
    prev_d = None
    dt_list = []
    for idx in dataset.index:
        s = idx.split(" ")
        try:
            hr = int(s[-1])-1
        except:
            # print(s)
            dt_list.append(None)
            continue
        if hr > 23:
            # print(dt)
            dt_list.append(None)
            continue
        # if expt != hr:
        #     print(dt)
        expt = (hr + 1) % 24
        dt = s[0] + " " + str(hr)
        s2 = s[0].split("/")
        day = s2[1]
        month = s2[0]
        year = s2[2]
        dt_idx = datetime.datetime.strptime(dt, '%m/%d/%Y %H')
        if prev_d != year:
            # print(year)
            prev_d = year
            # print(count/24)
            count = 0
        count += 1
        dt_list.append(dt_idx)
    
    return dt_list

def combining_hourly_data(data_df, y_column_position=0):

    new_row = []
    idx_list = []
    for idx,value in data_df.groupby(by=data_df.index.date).count().iterrows():
        temp_df = data_df[data_df.index.date == idx].copy()
        count = temp_df.shape[0]
        if count == 23:
            # add one
            temp_df = temp_df.append(temp_df.iloc[22], ignore_index=True)
        if count == 25:
            # remove one
            temp_df = temp_df.drop(index=temp_df.iloc[24:25].index)

        # combine to all row to one
        temp_row = temp_df.stack().to_frame().T
        temp_row.columns = [str(i) for i in range(0,temp_row.shape[1])]
        y_position = [y_column_position + (len(temp_df.columns) * k) for k in range(24)]
        non_y = [k for k in range(len(temp_row.columns)) if k not in y_position]
        # add y to the back
        new_order = non_y + y_position
        temp_row = temp_row[temp_row.columns[new_order]]
        new_row.append(temp_row)
        idx_list.append(idx)

    concat_df = pd.concat(new_row, axis=0)
    concat_df.index = idx_list

    return concat_df

def fix_dst(data_df):

    new_row = []
    idx_list = []
    for idx,value in data_df.groupby(by=data_df.index.date).count().iterrows():
        temp_df = data_df[data_df.index.date == idx].copy()
        count = temp_df.shape[0]
        if count == 23:
            # add one
            temp_df = temp_df.append(temp_df.iloc[22], ignore_index=True)
        if count == 25:
            # remove one
            temp_df = temp_df.drop(index=temp_df.iloc[24:25].index)

        # combine to all row to one
        new_row.append(temp_df)

    concat_df = pd.concat(new_row, axis=0)

    return concat_df
    
def convert_hourly_to_daily_data(data_df):
    # convert hourly data to daily data
    daily_df = data_df.groupby(by=data_df.index.date).agg({
        'temperature':['max','min']
        ,'humidity':['max','min']
        ,'apparentTemperature':['max','min']
        ,'pressure':['max','min']
        ,'windSpeed':['max','min']
        ,'cloudCover':['max','min']
        ,'windBearing':['max','min']
        ,'dewPoint':['max','min']
        ,'precipIntensity':['max','min']
        ,'precipProbability':['max','min']
        ,'Load':['max']})
    # daily_df = daily_df.reset_index(level=[0,1])
    daily_df.columns = [
        "Max_temperature","Min_temperature"
        ,"Max_humidity","Min_humidity"
        ,"Max_apparentTemperature","Min_apparentTemperature"
        ,"Max_pressure","Min_pressure"
        ,"Max_windSpeed","Min_windSpeed"
        ,"Max_cloudCover","Min_cloudCover"
        ,"Max_windBearing","Min_windBearing"
        ,"Max_dewPoint","Min_dewPoint"
        ,"Max_precipIntensity","Min_precipIntensity"
        ,"Max_precipProbability","Min_precipProbability"
        ,"MaxPower"]
    daily_df.index.name = "date"

    return daily_df

def prepare_vpeak_training_data(daily_df, current_year=2020):
    historical_df = daily_df[daily_df.index.year < current_year][["Load"]]
    historical_daily_demand = []
    one_year_daily_demand = []
    for month in range(1, 13):
        monthly_df = historical_df[(historical_df.index.month==month)]
        historical_daily_demand.append(monthly_df["Load"].tolist())
        
        monthly_df = historical_df[(historical_df.index.month==month) & (historical_df.index.year==current_year-1)]
        one_year_daily_demand.append(monthly_df["Load"].tolist())

    return (historical_daily_demand, one_year_daily_demand)
