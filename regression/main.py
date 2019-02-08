print("     (.)~(.)")
print("    (-------)")
print("---ooO-----Ooo----")
print("   REGRESSION")                                       
print("------------------")
print("    ( )   ( )")
print("    /|\   /|\\")

import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100 
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df = df.fillna(-99999)

# Gets 10% of the DF
forecast_out = int(math.ceil(0.01*len(df)))
# Shift columns up
df['label'] = df[forecast_col].shift(-forecast_out)

df = df.dropna()

X = np.array(df.drop(['label']))