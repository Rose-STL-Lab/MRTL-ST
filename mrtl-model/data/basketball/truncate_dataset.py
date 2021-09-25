import pandas as pd
import numpy as np
import csv
import os

def trunc(x, y):
    if not (-6 < x < 6 and -2 < y < 10):
        return pd.Series([-100, -100])
    else:
        return pd.Series([x, y])
    
df = pd.read_pickle('full_data_untrunced_new.pkl')
    
df[['def1_trunc_x', 'def1_trunc_y']] = df.apply(lambda row: trunc(row['def1_trunc_x'], row['def1_trunc_y']), axis=1)
df[['def2_trunc_x', 'def2_trunc_y']] = df.apply(lambda row: trunc(row['def2_trunc_x'], row['def2_trunc_y']), axis=1)
df[['def3_trunc_x', 'def3_trunc_y']] = df.apply(lambda row: trunc(row['def3_trunc_x'], row['def3_trunc_y']), axis=1)
df[['def4_trunc_x', 'def4_trunc_y']] = df.apply(lambda row: trunc(row['def4_trunc_x'], row['def4_trunc_y']), axis=1)
df[['def5_trunc_x', 'def5_trunc_y']] = df.apply(lambda row: trunc(row['def5_trunc_x'], row['def5_trunc_y']), axis=1)

df.to_pickle('full_data_trunced_new.pkl')