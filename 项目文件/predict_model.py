import numpy as np
import pandas as pd
from arch import arch_model
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ruptures as rpt
import math
import pickle
import statsmodels.api as sm
from openai import OpenAI
from WindPy import w
w.start()  # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
w.isconnected()  # 判断WindPy是否已经登录成功
def data_download_trade(list,fields,start_time,end_time):
    data = pd.DataFrame()
    for i in list:
        temp = w.wsd(i, fields, start_time, end_time, options="PriceAdj=F",usedf=True)[1]
        temp.rename(columns={'CLOSE': i}, inplace=True)
        data = pd.concat([data,temp], axis=1)
    return data
#输入：

#train_data:dataframe， 列：Date，exchange
#pred_data:dataframe. 列：Date，exchange,.......有很多特征列
#vix_data:dataframe,列：Date，vix
#attention:预测数据需要基于lstm的time_step,多添加30天的预测步骤
class ForexPredictor:
    #参数:训练日期末尾，需要预测日期起始
    def __init__(self):
        #训练数据
        self.train_data =None
        self.train_lstm_data=None
        self.train_garch_data=None
        self.train_lightgbm_data=None
        self.train_regression_data=None

        #预测数据
        self.predict_data=None
        self.predict_lstm_data=None
        self.predict_garch_data=None
        self.predict_lightgbm_data=None
        self.predict_regression_data=None
        self.predict_news_data=None

        self.vix_data=None
        self.macro_data=None

        self.lstm_time_step=30
    # 数据获取与预处理（注意，数据的第一列应当是日期格式）
    def load_data(self, train_data, vix_data,macro_data,pred_data=pd.DataFrame(),news_data=pd.DataFrame()):
        # 读取vix数据
        self.vix_data = vix_data
        self.macro_data=macro_data

        #全部训练数据导入
        self.train_data=train_data#只有两列，Date和exchange

        #用来训练lstm的数据
        self.train_lstm_data=self.train_data[['Date','exchange']]

        #用来训练garch的数据
        self.train_garch_data=self.train_data[['Date','exchange']]
        #用来训练lightgbm的数据
        self.train_lightgbm_data=self.train_data[['Date','exchange']]
        #用来训练regression的数据
        self.train_regression_data=self.train_data[['Date','exchange']]

        if not pred_data.empty:
            #加载预测数据，可单独读取
            #创建预测lstm数据
            predict_data=pred_data #含有很多特征列

            self.predict_lstm_data = predict_data[['Date','exchange','vix']]
            #用来预测garch的数据
            self.predict_garch_data=predict_data[['Date','returns','exchange']]
            #用来预测lightgbm的数据
            self.predict_lightgbm_data=predict_data[['Date','lag_1','lag_3','lag_5','volatility','ma_diff_5_20','rsi']]

            # #用来预测regression的数据
            self.predict_regression_data=predict_data[['Date','中美利差', '美元指数', '汇差', '中间价价差', 'NDF1M', 'NDF1Y', 'CHDF1M', 'CHDF1Y', '掉期点1Y']]

            #用来预测news的数据
            self.predict_news_data=news_data


        pass

    # 技术指标生成
    def create_tech_features(self, window_size=5):
        prices=self.train_data['exchange'].rename('returns')
        # 计算对数收益率（平稳化处理）
        returns = np.log(prices).diff().dropna()

        df = pd.DataFrame(returns)
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

        self.tech_features=df
        #print(self.tech_features)
        pass

    #判断最近一段时间汇率波动情况，来决定是否需要重新训练模型
    def check_volatility(self):
        data_1=self.train_data

        pass

    #创建gbm的特征序列，其中输入series：pandas_series,列名为'returns'
    def create_gbm_features(self):
        train_df1= self.tech_features.shift(1)
        train_df=train_df1.copy()
        return train_df.dropna()


    # 训练模型
    def train_lstm(self):
        data = pd.merge(self.train_lstm_data, self.vix_data, on='Date', how='inner')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['exchange', 'vix']])

        # 创建时间序列数据：将数据分为输入（X）和目标（y）
        def create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), :])  # 使用前 timestep 天的汇率和VIX数据
                y.append(data[i + time_step, 0])  # 预测未来的汇率
            return np.array(X), np.array(y)

        time_step = self.lstm_time_step
        X, y = create_dataset(scaled_data, time_step)

        # Step 3: 导入训练数据
        X_train,y_train = X,y
        a=[]
        a.append(scaled_data[-time_step:, :])

        # 将数据转换为 PyTorch 的 tensor 格式
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # Step 4: 构建 LSTM 网络
        class LSTMModel(nn.Module):
            def __init__(self, input_size=2, hidden_layer_size=50, output_size=1):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
                self.fc = nn.Linear(hidden_layer_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                predictions = self.fc(lstm_out[:, -1])
                return predictions

        # Step 5: 训练模型
        model = LSTMModel(input_size=2, hidden_layer_size=50, output_size=1)  # 输入维度为2（汇率和VIX）
        criterion = nn.MSELoss()  # 均方误差损失
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        epochs = 200
        for epoch in range(epochs):
            model.train()

            # 前向传播
            y_pred = model(X_train)

            # 计算损失
            loss = criterion(y_pred, y_train.unsqueeze(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        #模型参数评估，观察是否训练效果良好
        model.eval()
        train_pred = model(X_train)
        train_pred = scaler.inverse_transform(
            np.hstack((train_pred.detach().numpy(), np.zeros_like(train_pred.detach().numpy()))))[:, 0]
        y_train = scaler.inverse_transform(
            np.hstack((y_train.unsqueeze(-1).detach().numpy(), np.zeros_like(y_train.unsqueeze(-1).detach().numpy()))))[
                  :, 0]
        mse = mean_squared_error(train_pred, y_train)
        mae = mean_absolute_error(train_pred, y_train)
        r2 = r2_score(train_pred, y_train)
        print(f"Learning Rate:  - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        #保存模型
        torch.save(model.state_dict(), 'model_save/lstm_model.pth')

        pass


    #输出未来一天百分比收益率波动的95%置信区间
    def train_garch(self):
        #数据导入
        data_use=self.train_garch_data.copy()
        data_use.set_index('Date', inplace=True)
        def pelt_model(data):
            # 创建PELT检测器
            model = "l1"  # 使用L2模型来计算变化点
            algo = rpt.Pelt(model=model).fit(data.values)
            # 检测变化点
            change_points = algo.predict(pen=12)  # pen是惩罚系数，调整可以影响段的数量

            # 返回最后一个变化点
            return data.iloc[change_points[-2]:]
        #数据切割
        data_pelt = pelt_model(data_use)
        # 计算百分比对数收益率
        data_pelt['Returns'] = 100 * np.log(data_pelt['exchange']).diff().dropna()
        data_pelt.dropna(inplace=True)
        # 拟合GARCH(1,1)模型
        model = arch_model(data_pelt['Returns'], mean='constant', lags=2, vol='GARCH', p=1, q=1)
        model_fit = model.fit(disp='off')

        # 使用 pickle 保存模型
        with open('model_save/arch_model.pkl', 'wb') as f:
            pickle.dump(model_fit, f)

        f.close()
        pass


    #输出未来一天的涨跌
    def train_lightgbm(self):
        train_features = self.create_gbm_features()
        X_train = train_features.drop(columns=['returns'])
        y_train= (train_features['returns'] > 0).astype(int)
        # 训练LightGBM
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        # 预测
        pred = model.predict(X_train)
        print(f'Accuracy: {accuracy_score(y_train, pred):.2%}')

        # 获取底层的 Booster 对象并保存模型
        model.booster_.save_model('model_save/lgb_model.txt')
        pass



    def train_regression(self):

        # 定义自变量和因变量
        data=self.macro_data
        lagged_columns = ['中美利差', '美元指数', '汇差', '中间价价差', 'NDF1M', 'NDF1Y', 'CHDF1M', 'CHDF1Y',
                          '掉期点1Y']
        for col in lagged_columns:
            data[col + '_lag1'] = data[col].shift(1)
        # 去除包含缺失值的行
        data = data.dropna()
        X = data[['中美利差_lag1', '美元指数_lag1', '汇差_lag1', '中间价价差_lag1', 'NDF1M_lag1', 'NDF1Y_lag1',
                  'CHDF1M_lag1', 'CHDF1Y_lag1', '掉期点1Y_lag1']]
        y = self.train_regression_data['exchange']
        y=y[y.index>=X.index[0]]
        y=y[y.index<=X.index[-1]]

        # 添加常数项
        X = sm.add_constant(X)

        # 进行线性回归
        model = sm.OLS(y, X).fit()
        print(model.summary())
        # 保存模型到文件
        with open('model_save/ols_model.pkl', 'wb') as f:
            pickle.dump(model, f)


        pass

    def train_news_model(self):
        pass


    #重新训练所有数据
    def train_all_models(self):
        self.train_lstm() #预测准确值
        self.train_garch() #预测波动区间
        self.train_lightgbm() #预测涨跌趋势
        self.train_regression() #调整预测值
        # self.train_news_model() #新闻分析
        pass
    # 预测模块
    def _lstm_predict(self):
        # 首先，定义模型架构
        # Step 4: 构建 LSTM 网络
        class LSTMModel(nn.Module):
            def __init__(self, input_size=2, hidden_layer_size=50, output_size=1):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
                self.fc = nn.Linear(hidden_layer_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                predictions = self.fc(lstm_out[:, -1])
                return predictions
        model = LSTMModel(input_size=2, hidden_layer_size=50, output_size=1)

        # 然后加载参数
        model.load_state_dict(torch.load('model_save/lstm_model.pth'))

        model.eval()

        #创建测试集
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.predict_lstm_data[['exchange', 'vix']])

        # 创建预测时间序列数据：将数据分为输入（X）和目标（y）
        def create_pred_dataset(data, time_step):
            X= []
            for i in range(len(data)-time_step):
                X.append(data[i:(i + time_step), :])  # 使用前 timestep 天的汇率和VIX数据
            return np.array(X)

        time_step = self.lstm_time_step
        X= create_pred_dataset(scaled_data, time_step)
        X = torch.tensor(X, dtype=torch.float32)

        test_pred = model(X)
        test_pred = scaler.inverse_transform(
            np.hstack((test_pred.detach().numpy(), np.zeros_like(test_pred.detach().numpy()))))[:, 0]
        test_pred_data=pd.DataFrame(([0]*self.lstm_time_step+test_pred.tolist()),index=self.predict_lstm_data['Date'],columns=['lstm_pred'])
        return test_pred_data

    def _garch_forecast(self):
        with open('model_save/arch_model.pkl', 'rb') as f:
             model_fit = pickle.load(f)

        # 提取模型参数
        mu = model_fit.params['mu']
        omega = model_fit.params['omega']
        alpha = model_fit.params['alpha[1]']
        beta = model_fit.params['beta[1]']

        # 初始化条件方差和残差
        last_sigma2 = model_fit.conditional_volatility[-1] ** 2
        last_epsilon = 0
        # 递推计算测试集波动率
        test_volatility = []
        for r in self.predict_garch_data['returns']:
            sigma2 = omega + alpha * (last_epsilon ** 2) + beta * last_sigma2
            test_volatility.append(np.sqrt(sigma2))
            current_epsilon = r - mu
            last_epsilon = current_epsilon
            last_sigma2 = sigma2

        # 转换为时间序列
        test_volatility = pd.Series(test_volatility, index=self.predict_garch_data.index)

        # 计算95%置信区间（基于正态假设）
        confidence_level = 1.96
        upper_band = mu + confidence_level * test_volatility
        lower_band = mu - confidence_level * test_volatility
        true_ret_upper = [math.exp(i / 100) - 1 for i in upper_band]
        true_ret_lower = [math.exp(i / 100) - 1 for i in lower_band]
        price_upper = [(1+true_ret_upper[i]) * self.predict_garch_data['exchange'].iloc[i] for i in range(len(true_ret_upper))]
        price_lower = [(1+true_ret_lower[i]) * self.predict_garch_data['exchange'].iloc[i] for i in range(len(true_ret_lower))]
        price_upper = pd.Series(price_upper, index=self.predict_garch_data['Date'])
        price_lower = pd.Series(price_lower, index=self.predict_garch_data['Date'])
        end_data=pd.DataFrame()
        end_data['garch_upper']=price_upper
        end_data['garch_lower']=price_lower
        #print('GARCH：未来一段时间的涨跌波动区间（95%置信区间）：',end_data)
        return end_data

    def _lgb_predict(self):
        lgb_model=lgb.Booster(model_file='model_save/lgb_model.txt')
        pred = lgb_model.predict(self.predict_lightgbm_data[['lag_1','lag_3','lag_5','volatility','ma_diff_5_20','rsi']])
        #pred:1为涨，0为跌
        pred = np.around(pred, 0).astype(int)
        pred_data=pd.DataFrame(pred,index=self.predict_lightgbm_data['Date'],columns=['lgb_pred'])
        return pred_data

    def _regression_adjust(self):
        # 加载模型
        with open('model_save/ols_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
            # 确保列顺序和训练数据一致
        columns = ['中美利差', '美元指数', '汇差', '中间价价差', 'NDF1M', 'NDF1Y',
                       'CHDF1M', 'CHDF1Y', '掉期点1Y']
        new_X=self.predict_regression_data[columns]
        # 检查列名是否包含'const'
        if 'const' not in new_X.columns:
            new_X['const'] = 1
        # 重新排列列顺序，让常数项在第一列
        new_X = new_X[['const'] + columns]
        res=loaded_model.predict(new_X)
        #print(res)
        #print('Regression：未来一段时间的汇率：',res)
        res_data=pd.DataFrame(res,columns=['ols_pred'])
        res_data.index=self.predict_regression_data['Date']
        #print(res_data)
        return res_data

    def _news_integration(self):
        news_data=self.predict_news_data.copy()
        def get_sentiment(text):
            # 在这里调用api
            client = OpenAI(api_key="sk-9f62b21b88da4158ab11a3724b9122ae", base_url="https://api.deepseek.com")

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你现在是金融行业的专家"},
                    {"role": "user",
                     "content": "请你根据下面的新闻内容给出对于离岸人民币兑美元汇率的市场情绪影响的评分，评分范围为-10到10，评分越高表示情感倾向越积极， 直接输出评分，不需要文字分析，下面是新闻内容：{}".format(
                         text)},
                ],
                stream=False
            )
            return int(response.choices[0].message.content)
        print(1)
        news_data['sentiment']=news_data['text'].apply(get_sentiment)
        print(2)
        news_data['sentiment']=news_data['sentiment'].astype(int)
        news_data.set_index('Date',inplace=True)
        news_data=news_data[['sentiment']]
        news_signal=news_data.groupby(news_data.index).sum()
        print(news_signal)
        return news_signal



        # 预测模块
    def generate_predictions(self):
            lstm_pred = self._lstm_predict()
            garch_pred=self._garch_forecast()
            lgb_signal = self._lgb_predict()
            ols_pred = self._regression_adjust()
            news_pred=self._news_integration()
            all_pred=pd.concat([lstm_pred,garch_pred,lgb_signal,ols_pred,news_pred],axis=1)
            print(all_pred)
            # grarch_lgb_combine=np.where((all_pred['lgb_pred']==1),all_pred['garch_upper'],all_pred['garch_lower'])
            weights=pd.DataFrame(index=all_pred.index,columns=['lstm','garch_upper','garch_lower','ols_pred'])
            weights['lstm_pred']=0.8
            weights['garch_upper']=np.where((lgb_signal['lgb_pred']==0),0.04,0.06)
            weights['garch_lower']=np.where((lgb_signal['lgb_pred']==0),0.06,0.04)
            weights['ols_pred']=0.1
            all_pred['adjusted_pred']=all_pred['lstm_pred']*weights['lstm_pred']+all_pred['garch_upper']*weights['garch_upper']+all_pred['garch_lower']*weights['garch_lower']+all_pred['ols_pred']*weights['ols_pred']
            all_pred['signal']=all_pred['adjusted_pred'].diff().shift(1)
            all_pred['news_add_signal'] = all_pred.apply(lambda row: -row['signal'] if (row['sentiment'] < -10) and (row['signal']>0) else row['signal'], axis=1)

            all_pred=all_pred.iloc[self.lstm_time_step:,:]
            data_start=all_pred.index[0].strftime('%Y-%m-%d')
            data_end=all_pred.index[-1].strftime('%Y-%m-%d')
            exchange_true=data_download_trade(['USDCNY.IB'],'CLOSE',data_start,data_end)
            all_pred['exchange_true']=exchange_true['USDCNY.IB']
            print(mean_squared_error(all_pred['adjusted_pred'],all_pred['exchange_true']))
            plt.plot(all_pred['adjusted_pred'],label='pred')
            plt.plot(all_pred['exchange_true'],label='true')
            plt.legend()
            plt.show()

            all_pred.to_excel('res_pred_data.xlsx')
            # final_signal = self._news_integration()

    # # 决策模块
    # def generate_trading_signal(self, predictions):
    #     # 多因子决策逻辑
    #     if (predictions['adjusted_pred'] > current_price + threshold) and  (predictions['trend_signal'] == 1):
    #         action = 'BUY'
    #
    #     elif ...:  # 其他条件组合
    #         action = 'SELL'
    #
    #     else:
    #         action = 'HOLD'
    #         return action




# 使用示例
predictor = ForexPredictor()
train_data=pd.read_excel('训练数据集/TRD_Exchange.xlsx')
vix_data=pd.read_excel('训练数据集/VIX.GI.xlsx')
macro_data=pd.read_excel('训练数据集/macro_data.xlsx')
news_data=pd.read_excel('回测数据集/back_news_data.xlsx')
pred_data=pd.read_excel('回测数据集/back_pred_data.xlsx')

predictor.load_data(train_data, vix_data,macro_data,pred_data,news_data)
predictor.create_tech_features()
# predictor.train_lstm()
# predictor.train_garch()
# predictor.train_lightgbm()
# predictor.train_regression()
#predictor.train_all_models()
# predictor._lstm_predict()
# predictor._garch_forecast()
# predictor._lgb_predict()
# predictor._regression_adjust()
predictor.generate_predictions()
# predictor.train_all_models()
# print(f"交易建议: {results['final_signal']}")