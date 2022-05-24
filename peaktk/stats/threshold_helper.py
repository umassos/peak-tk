import pandas as pd
import numpy as np

def find_threshold(df, quantile):
    t_list = []
    for i in range(1,13):
        month_mask = (df.index.month == i)
#         year_mask = (df.index.year == 2016)
        threshold = df[month_mask].round().quantile(quantile)
        t_list.append(threshold)
    return t_list

