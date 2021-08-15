import pickle5 as pickle

data = pickle.load('full_data_bs.pkl')

data.to_pickle('full_data.pkl')