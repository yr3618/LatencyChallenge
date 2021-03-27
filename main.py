import numpy as np
import pandas as pd

import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sys import stdin


#Store the data into a data frame
df=pd.read_csv('LatencyData.csv')

#Show the Data frame
#Create functions to calculate SMA and EMA
def SMA(data, period=30, column='LogReturns'):
    return data[column].rolling(window=period).mean()

def EMA(data, period=20, column='LogReturns'):
    return data[column].ewm(span=period, adjust=False).mean()

#Moving average/MACD
def MACD(data, period_long=26, period_short=12, period_signal=9, column='LogReturns'):
    #Short term EMA
    ShortEMA = EMA(data, period=period_short, column=column)
    #Long term EMA
    LongEMA = EMA(data, period=period_long, column=column)
    #Calculate and store MACD
    data['MACD'] = ShortEMA - LongEMA
    data['signal_Line'] = EMA(data, period=period_signal, column='MACD')
    return data

#Create RSI
def RSI(data, period=14, column='LogReturns'):
    delta=data[column].diff(1)
    delta=delta.dropna()
    up=delta.copy()
    down=delta.copy()
    up[up<0]=0
    down[down>0]=0
    data['up']=up
    data['down']=down
    AVG_Gain = SMA(data, period, column='up')
    AVG_Loss = abs(SMA(data, period, column='down'))
    RS= AVG_Gain/AVG_Loss
    RSI=100.0-(100.0/(1.0+RS))
    data['RSI']=RSI
    return data

#Add indicators to the training data set
MACD(df)
RSI(df)
df['SMA']=SMA(df)
df['EMA']=EMA(df)
df['Target']=np.where(df['LogReturns'].shift(-1)>df['LogReturns'],1,0)

#Ask for input
print("Please enter your data:")
for line in stdin:
    if line == '':
        break
    d=[float(x) for x in line.split(',')]

#Make input into a dataframe
df2=pd.DataFrame(d, columns=['LogReturns'])

#Add indicators to the user data set
MACD(df2)
RSI(df2)
df2['SMA']=SMA(df2)
df2['EMA']=EMA(df2)
df2['Target']=np.where(df2['LogReturns'].shift(-1)>df2['LogReturns'],1,0)

#Replace indicators NaN values with 0
df2['SMA'] = df2['SMA'].replace(np.nan, 0)
df2['EMA'] = df2['EMA'].replace(np.nan, 0)
df2['MACD'] = df2['MACD'].replace(np.nan, 0)
df2['RSI'] = df2['RSI'].replace(np.nan, 0)

#Remove first days of data to remove NaN
#df=df[29:]
#Split columns
keep_columns = ['LogReturns', 'MACD', 'signal_Line', 'RSI', 'SMA', 'EMA']
X=df[keep_columns].values
Y=df['Target'].values
X2=df2[keep_columns].values
Y2=df2['Target'].values

#Split the data into two data sets: independent X and dependent Y.
#X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=2)

#Create and train the model
tree=DecisionTreeClassifier().fit(X, Y)

#Check how well it classifies data on the testing set
#print(tree.score(X, Y))



tree_predictions=tree.predict(X2)
from sklearn.metrics import classification_report
print(classification_report(Y2, tree_predictions))
print(df2)
