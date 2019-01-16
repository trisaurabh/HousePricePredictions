import sys
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def load_data():
    df = pd.read_csv('house_sales.csv')
    df['latitude'] = df['latitude'].map(lambda x : round(x,2))
    df['longitude'] = df['longitude'].map(lambda x: round(x,2))
    df = df.drop(columns = ['renovation_date','size_basement'])
    df = df.drop(df[df['price'] > 5000000].index)
    df = df.drop(df[df.size_house > 8000].index)
    df = df.drop(df[df.num_bed > 30].index)
    return df

def modelRF(df):
    RFreg = RandomForestRegressor().fit(df[['num_bed', 'num_bath', 'size_house', 'size_lot', 'is_waterfront', 'zip','avg_size_neighbor_houses', 'avg_size_neighbor_lot']], df['price'])
    return RFreg

def predict(input):
    return modelRF(load_data()).predict(input)

#if __name__ == "__main__":
#    x = sys.argv[1]
#    y = ast.literal_eval(x)
#    z = [float(item) for item in y]
#    print(z)
#    predict([z])


print(predict([[3,	2.25,	2570,	7242,0,	98125,1690,7639]]))
