import pandas as pd
import numpy as np
#提取数据处理
from WindPy import w
import datetime
from bs4 import BeautifulSoup
import requests

from time import sleep, struct_time
import re


w.start()  # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
w.isconnected()  # 判断WindPy是否已经登录成功
#list:为提取代码序列,列表格式
#fields:提取数据种类，字符串格式
#start_time:开始时间，字符串格式
#end_time:结束时间，字符串格式
#name:数据集名称，字符串格式


def data_download_trade(list,fields,start_time,end_time):
    data = pd.DataFrame()
    for i in list:
        temp = w.wsd(i, fields, start_time, end_time, options="PriceAdj=F",usedf=True)[1]
        temp.rename(columns={'CLOSE': i}, inplace=True)
        data = pd.concat([data,temp], axis=1)
    return data


def data_download_edb(list,start_time,end_time):
    data = pd.DataFrame()
    for i in list:
        temp = (w.edb(i, start_time, end_time, "Fill=Previous",usedf=True))[1]
        temp.rename(columns={'CLOSE': i}, inplace=True)
        data = pd.concat([temp,data], axis=1)
    return data

#数据提取模板
# #在岸人民币、美元指数、离岸人民币、美元对人民币中间价、在岸人民币一年掉期、离岸人民币1月远期、离岸人民币1年远期
# ret_1=data_download_trade(["USDCNY.IB","USDX.FX","USDCNH.FX","USDCNY.EX","USDCNY1YS.IB","USDCNH1MF.FX","USDCNH1YF.FX"],"close", "2015-01-05", "2025-03-11")
# print(ret_1)
# #在岸人民币无本金交割1月远期、1年远期
# ret_2=data_download_trade(["CNYNDF1M.FX","CNYNDF1Y.FX",],"close", "2015-01-05", "2025-03-11")
# ret_2=ret_2[ret_2.index<=datetime.date(2024,12,31)]
# print(ret_2)
# #中国10年国债收益率、美国10年国债收益率
# ret_3=w.edb("M0325687,G0000891", "2015-01-05", "2025-03-11","Fill=Previous",usedf=True)[1]
# print(ret_3)
# 
# ret=pd.concat([ret_1,ret_2],axis=1)
# print(ret)
# ret.to_excel('市场数据集.xlsx')
# ret_3.to_excel("经济数据集.xlsx")
# vix_data=data_download_trade(['VIX.GI'],'CLOSE',"2015-01-05", "2025-03-11")
# vix_data.to_excel('本地数据集/VIX数据.xlsx')

# #中国10年国债，美国10年国债收益率
# #w.edb("M0325687,G0000891", "2024-03-11", "2025-03-11","Fill=Previous")

#最近一次更新时间
data_old_market=pd.read_excel('本地数据集/市场数据集.xlsx',index_col=0)
data_old_economic=pd.read_excel('本地数据集/经济数据集.xlsx',index_col=0)
data_old_vix=pd.read_excel('本地数据集/VIX数据.xlsx',index_col=0)

date_last_market=data_old_market.index[-1].strftime('%Y-%m-%d')
date_last_economic=data_old_economic.index[-1].strftime('%Y-%m-%d')
date_last_vix=data_old_vix.index[-1].strftime('%Y-%m-%d')
print(date_last_market)
print(date_last_economic)
print(date_last_vix)

market_assets=data_old_market.columns
economic_assets=data_old_economic.columns
vix_assets=data_old_vix.columns
print(market_assets)
print(economic_assets)
print(vix_assets)

# 获取昨天日期
base_date = datetime.datetime.now() - datetime.timedelta(days=1)
# 获取最近的交易日
previous_trading_day = pd.tseries.offsets.BDay().rollback(base_date)
# 将日期转换为'%Y-%m-%d'格式
date_latest = previous_trading_day.strftime('%Y-%m-%d')


if date_last_market<date_latest:
    print('需要更新日期')
    data_new_market=data_download_trade(market_assets, "CLOSE", date_last_market, date_latest)
    data_new_market.index = pd.to_datetime(data_new_market.index)
    # 更新新资产、新时间的资产序列
    data_whole_market = pd.concat([data_old_market.iloc[:-1, :], data_new_market], axis=0)
else:
    print('不需要更新日期')
    data_whole_market=data_old_market

if date_last_economic<date_latest:
    print('需要更新日期')
    data_new_econ=data_download_edb(economic_assets, date_last_economic, date_latest)
    data_new_econ.index = pd.to_datetime(data_new_econ.index)
    # 更新新资产、新时间的资产序列
    data_whole_econ = pd.concat([data_old_economic.iloc[:-1, :], data_new_econ], axis=0)
else:
    print('不需要更新日期')
    data_whole_econ=data_old_economic

if date_last_vix<date_latest:
    print('需要更新日期')
    data_new_vix=data_download_trade(vix_assets, "CLOSE", date_last_vix, date_latest)
    data_new_vix.index = pd.to_datetime(data_new_vix.index)
    # 更新新资产、新时间的资产序列
    data_whole_vix = pd.concat([data_old_vix.iloc[:-1, :], data_new_vix], axis=0)
else:
    print('不需要更新日期')
    data_whole_vix=data_old_vix


data_whole_market.sort_index(inplace=True)
data_whole_market.to_excel('本地数据集/市场数据集.xlsx')

data_whole_econ.sort_index(inplace=True)
data_whole_econ.to_excel('本地数据集/经济数据集.xlsx')

data_whole_vix.sort_index(inplace=True)
data_whole_vix.to_excel('本地数据集/VIX数据.xlsx')


#新闻数据更新
from datetime import datetime
data=pd.read_excel('本地数据集/news_list_bs4.xlsx')
print(data)
today = datetime.now()
today = today.strftime('%Y%m%d')


dic_url = {
    # "a": {
    #     "name": '华尔街见闻',
    #     "url": 'https://tophub.today/n/G2me3ndwjq'
    # },
    # "b": {
    #     "name": '第一财经',
    #     "url": 'https://tophub.today/n/0MdKam4ow1'
    # },
    # "c": {
    #     "name": '雪球',
    #     "url": 'https://tophub.today/n/X12owXzvNV'
    # },
    # "d": {
    #     "name": '新浪财经',
    #     "url": 'https://tophub.today/n/rx9ozj7oXb'
    # },
    "e": {
        "name": '和讯',
        "url": 'https://forex.hexun.com/fxobservation/'
    },
    "f": {
        "name": '新浪外汇',
        "url": 'https://finance.sina.com.cn/roll/index.d.html?cid=56982&page=1'
    }
}


def news_get_e(name, website):
    url = website
    req = requests.get(url)
    req.encoding = req.apparent_encoding
    html_text = req.text
    soup = BeautifulSoup(html_text, 'html.parser')
    paragraphs = soup.find_all('li')
    news_list = []
    for text in paragraphs:
        pattern = re.compile(r'\((\d{2}/\d{2} \d{2}:\d{2})\)(.*)')
        match = pattern.search(text.get_text())
        if match:
            news_list.append(text.get_text())
    strtime = []
    text = []
    for i in range(len(news_list)):
        temp_text = news_list[i]
        strtime.append((temp_text.split(')')[0] + ')').replace('(', '').replace(')', ''))
        text.append(temp_text.split(')')[1])
    links = soup.find_all('a')
    specific_urls = []
    for link in links:
        href = link.get('href')
        # print(href)
        if href and href.startswith('http://forex.hexun.com/'):
            parts = href.split('/')
            # print(parts)
            # print(parts[3].split('-'))
            if len(parts) == 5 and len(parts[3].split('-')) == 3:
                specific_urls.append(href)
    specific_urls = specific_urls[:(len(news_list))]
    df = pd.DataFrame({'time': strtime, 'text': text, 'url': specific_urls})
    return df


def simple_sentiment_analysis(text, positive_words, negative_words):
    positive_count = 0
    negative_count = 0
    # words = text.split()
    # print(text)
    for word in positive_words:
        if word in text:
            positive_count += 1
    for word in negative_words:
        if word in text:
            negative_count += 1
    # print(positive_count)
    # print(negative_count)
    if positive_count > 0 and negative_count == 0:
        return "积极"
    elif positive_count == 0 and negative_count > 0:
        return "消极"
    else:
        if positive_count > negative_count:
            return "积极"
        elif negative_count > positive_count:
            return "消极"
        else:
            return "中性"


df_all = pd.DataFrame(columns=['time', 'text'])
for outer_key, outer_value in dic_url.items():
    # print(outer_key, outer_value)
    if outer_key == 'e':
        df_all = pd.concat([df_all, news_get_e(name=outer_value['name'], website=outer_value['url'])], axis=0, ignore_index=True)
    # if outer_key == 'f':
    #     df_all = pd.concat([df_all, news_get_f(name=outer_value['name'], website=outer_value['url'])], axis=0, ignore_index=True)
# df_all.to_excel(f'df_all_{today}.xlsx')
# time.sleep(5)
# driver.quit()


dict_positive = list(pd.read_excel('dict_positive.xlsx')['Positive Word'])
dict_negative = list(pd.read_excel('dict_negative.xlsx')['Negative Word'])
df_news = df_all
# print(dict_positive)
# print(dict_negative)


sentiment = []
for i in range(len(list(df_news['text']))):sentiment.append(simple_sentiment_analysis(df_news['text'][i], dict_positive, dict_negative))
df_news['sentiments'] = sentiment
# print(sentiment)
# df_news.to_excel(f'news_list_{today}.xlsx')


def convert_date(time_list):
    new_list = []
    now = datetime.now()
    now_str = f"{now.month:02}/{now.day:02} {now.hour:02}:{now.minute:02}"
    for text in time_list:
        match1 = re.search(r'(\d{2})/(\d{2}) (\d{2}:\d{2})', text)
        if match1:
            new_text = f"{match1.group(1)}/{match1.group(2)} {match1.group(3)}"
            new_list.append(new_text)
            continue
        match2 = re.search(r'(\d{2})月(\d{2})日 (\d{2}:\d{2})', text)
        if match2:
            new_text = f"{match2.group(1).zfill(2)}/{match2.group(2).zfill(2)} {match2.group(3)}"
            new_list.append(new_text)
            continue
        match3 = re.search(r'\d{4}年(\d{1,2})月(\d{1,2})日', text)
        if match3:
            new_text = f"{match3.group(1).zfill(2)}/{match3.group(2).zfill(2)} 00:00"
            new_list.append(new_text)
            continue
        new_list.append(now_str)
    return new_list


df_news['time'] = convert_date(list(df_news['time']))
df_news['time'] = '2025-' + df_news['time']

# 转换为 datetime 格式
df_news['time'] = pd.to_datetime(df_news['time'], format='%Y-%m/%d %H:%M')

# 转换为指定格式 'yyyy-mm-dd'
df_news['time'] = df_news['time'].dt.strftime('%Y-%m-%d')

print(df_news)
print(data)
df_news = df_news.drop_duplicates(subset='text').sort_values(by='time',ascending=False).reset_index(drop=True)
# 使用 concat 将两个 DataFrame 拼接
date_end=df_news['time'].iloc[-1]
df_combined = pd.concat([df_news, data[data['time']<date_end]])
print(df_combined)
df_combined.to_excel('本地数据集/news_list_bs4.xlsx')

