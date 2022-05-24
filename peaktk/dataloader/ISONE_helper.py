
import glob
import math
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from peaktk.preprocessing import helper_isone
from peaktk.preprocessing import preprocessing

def load_data(DEMAND_PATH, WEATHER_PATH, years=[2018, 2019, 2020]):
    # load ISONE dataset
    path = DEMAND_PATH
    all_df = []
    for y in years:
        y_path = path + str(y)
        all_files = sorted(glob.glob(y_path + "/*.csv"))
        for filename in all_files:
            all_df.append(pd.read_csv(filename, header=None, skiprows=6, names=["D","Date","HourEnding","Load"], parse_dates = [['Date', 'HourEnding']], index_col=0))
    isone_df = pd.concat(all_df).dropna()

    dt_list = helper_isone.convert_string_to_datetime(isone_df)
    isone_df["dt"] = dt_list

    isone_df.dropna(subset=["dt"], inplace=True)
    # change index
    isone_df.set_index("dt", inplace=True)
    infer_dst = np.array([False] * isone_df.shape[0])
    isone_df.index = isone_df.index.tz_localize('US/Eastern', ambiguous=infer_dst)

    # load ISONE weather data
    path = WEATHER_PATH
    all_df = []
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        temp_df = pd.read_csv(filename, index_col="time", parse_dates=["time"])
        infer_dst = np.array([False] * temp_df.shape[0])
        temp_df.index = pd.to_datetime(temp_df.index, unit='s').tz_localize('UTC', ambiguous=infer_dst).tz_convert('US/Eastern')
        all_df.append(temp_df)
    weather_df = pd.concat(all_df).dropna()

    # merge
    load_df2 = isone_df[~isone_df.index.duplicated(keep='first')]
    weather_df2 = weather_df[~weather_df.index.duplicated(keep='first')]

    load_df2 = load_df2.resample('60min').fillna(method='pad')
    weather_df2 = weather_df2.resample('60min').fillna(method='pad')

    merge_df = pd.concat([load_df2, weather_df2], axis=1, join="inner")

    # convert to daily data
    daily_df = helper_isone.convert_hourly_to_daily_data(merge_df)
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df.rename({"MaxPower":"Load"}, axis=1, inplace=True)

    daily_df.index.name = "date"
    
    return daily_df

