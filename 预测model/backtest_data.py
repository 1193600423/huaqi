import pandas as pd
import numpy as np

df=pd.read_excel('res_pred_data.xlsx',index_col=0)
print(df)

data=df[['news_add_signal','exchange_true']]
data.rename(columns={'news_add_signal':'signal'},inplace=True)
data.fillna(0,inplace=True)
data.rename(columns={'exchange_true':'close'},inplace=True)
data['open']=data['close']
data['high']=0
data['low']=0
data['volume']=0
data=data[['open','close','signal','low','high','volume']]
data.to_excel('回测数据集/back_test_data.xlsx')