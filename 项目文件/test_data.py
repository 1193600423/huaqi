import pandas as pd
import numpy as np
from datetime import datetime

test_data_start=datetime(2024,1,1)
test_data_end=datetime(2025,3,23)

#输入exchange数据
def create_tech_features(data,window_size=5):
    df=data.copy()
    df['returns'] = np.log(df['exchange']).diff()
    df.drop('exchange',axis=1,inplace=True)
    # 生成技术指标（MA, RSI等）
    # 滞后项
    for lag in [1, 3, 5]:
        df[f'lag_{lag}'] = df['returns'].shift(lag)
    # 滚动波动率
    df['volatility'] = df['returns'].rolling(window_size).std()
    # 均线差值
    df['ma_diff_5_20'] = df['returns'].rolling(5).mean() - df['returns'].rolling(20).mean()
    # RSI
    delta = df['returns'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window_size).mean()
    avg_loss = loss.rolling(window_size).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    return df

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
data_news=pd.read_excel('本地数据集/news_list_bs4.xlsx')[['time','text','sentiments']]
data_news.rename(columns={'time':'Date'},inplace=True)
data_news['Date']=pd.to_datetime(data_news['Date'])
data_news.set_index('Date',inplace=True)

print(data_news)
data_exchange=data_market[['USDCNH.FX']]
print(data_exchange)
data_exchange.rename(columns={'USDCNH.FX':'exchange'},inplace=True)
#做出特征
data_features=create_tech_features(data_exchange)
data_whole=pd.merge(data_exchange,data_features,left_index=True,right_index=True,how='left')



data_economic=pd.read_excel('本地数据集/经济数据集.xlsx',index_col=0)
data_processed=process_data(data_market,data_economic)
data_vix=pd.read_excel('本地数据集/VIX数据.xlsx',index_col=0)

data_whole=pd.merge(data_whole,data_processed,left_index=True,right_index=True,how='left')
data_whole=pd.merge(data_whole,data_vix,left_index=True,right_index=True,how='left')


data=data_whole.shift(1)

data=data[data.index>=test_data_start]
data=data[data.index<=test_data_end]

data.index.name='Date'
data.rename(columns={'VIX.GI':'vix'},inplace=True)
data.fillna(method='ffill',inplace=True)

#新闻数据对齐
data_news_back=pd.merge(data,data_news,left_index=True,right_index=True,how='left')
data_news_back=data_news_back[['text']]


print(data)

data.to_excel('回测数据集/back_pred_data.xlsx')
data_news_back.to_excel('回测数据集/back_news_data.xlsx')
