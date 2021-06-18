import pandas as pd

raw_data = pd.read_pickle("./full_data_bs.pkl")

for i in range(4):
    quarter_data = raw_data.loc[raw_data['quarter'] == i]
    
    quarter_data.to_pickle("full_data_quarter_{0}.pkl".format(i + 1))