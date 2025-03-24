import pandas as pd
import numpy as np
from datetime import datetime


train_date_start=datetime.strptime('2017-01-01','%Y-%m-%d')
train_date_end=datetime.strptime('2024-01-01','%Y-%m-%d')

def process_data(market_df,economic_df):
    output_df = pd.DataFrame()
    output_df.index=market_df.index

    output_df['中美利差'] = (economic_df.iloc[:,0] - economic_df.iloc[:, 1]) * 100
    output_df['美元指数'] = market_df.iloc[:, 1]
    output_df['汇差'] = (market_df.iloc[:, 0] - market_df.iloc[:, 2]) * 10000
    output_df['中间价价差'] = (market_df.iloc[:, 0] - market_df.iloc[:, 3]) * 10000
    output_df['NDF1M'] = market_df.iloc[:, 7]
    output_df['NDF1Y'] = market_df.iloc[:, 8]
    output_df['CHDF1M'] = market_df.iloc[:, 5]
    output_df['CHDF1Y'] = market_df.iloc[:, 6]
    output_df['掉期点1Y'] = market_df.iloc[:, 4]

    return output_df

data_market=pd.read_excel('本地数据集/市场数据集.xlsx',index_col=0)
data_vix=pd.read_excel('本地数据集/VIX数据.xlsx',index_col=0)
data_economic=pd.read_excel('本地数据集/经济数据集.xlsx',index_col=0)

data_macro=process_data(data_market,data_economic)

TRD_Exchange=data_market['USDCNH.FX']

TRD_Exchange_train=TRD_Exchange[train_date_start:train_date_end]
VIX_train=data_vix[train_date_start:train_date_end]
# 合并数据
data_whole = pd.merge(TRD_Exchange_train, VIX_train, how='left', left_index=True, right_index=True)
data_whole=pd.merge(data_whole,data_macro,how='left',left_index=True,right_index=True)

print(data_whole)
#检查缺失值
print(data_whole.isnull().sum())

TRD_Exchange_end=data_whole['USDCNH.FX']
TRD_Exchange_end.index.name='Date'
TRD_Exchange_end.rename('exchange',inplace=True)
VIX_end=data_whole['VIX.GI']
VIX_end.index.name='Date'
VIX_end.rename('vix',inplace=True)
macro_end=data_whole.drop(['USDCNH.FX','VIX.GI'],axis=1)
macro_end.index.name='Date'

TRD_Exchange_end.to_excel('训练数据集/TRD_Exchange.xlsx')
VIX_end.to_excel('训练数据集/VIX.GI.xlsx')
macro_end.to_excel('训练数据集/macro_data.xlsx')