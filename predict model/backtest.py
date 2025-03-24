import backtrader as bt
from datetime import datetime
import pandas as pd
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        # 均线参数设置15天，15日均线
        ('maperiod', 20),
    )

    def log(self, txt, dt=None):
        # 记录策略的执行日志
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    def __init__(self):
        # 保存收盘价的引用
        self.dataclose = self.datas[0].close
        # 跟踪挂单
        self.order = None
        # 买入价格和手续费
        self.buyprice = None
        self.buycomm = None
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.maperiod)

        # 订单状态通知，买入卖出都是下单
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # broker 提交/接受了，买/卖订单则什么都不做
            return

        # 检查一个订单是否完成
        # 注意: 当资金不足时，broker会拒绝订单
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                '已买入, 价格: %.2f, 费用: %.2f, 佣金 %.2f' %
                (order.executed.price,
                order.executed.value,
                order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log('已卖出, 价格: %.2f, 费用: %.2f, 佣金 %.2f' %
                (order.executed.price,
                order.executed.value,
                order.executed.comm))
            # 记录当前交易数量
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')

        # 其他状态记录为：无挂起订单
        self.order = None


    # 交易状态通知，一买一卖算交易
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('交易利润, 毛利润 %.2f, 净利润 %.2f' %
        (trade.pnl, trade.pnlcomm))


    def next(self):
        # 记录收盘价
        self.log('Close, %.2f' % self.dataclose[0])

        # 如果有订单正在挂起，不操作
        if self.order:
            return

        # 如果没有持仓则买入
        if not self.position:
            # 今天的收盘价在均线价格之上
            if self.dataclose[0] > self.sma[0]:
                # 买入
                self.log('买入单, %.2f' % self.dataclose[0])
                # 跟踪订单避免重复
                self.order = self.buy()
        else:
            # 如果已经持仓，收盘价在均线价格之下
            if self.dataclose[0] < self.sma[0]:
                # 全部卖出
                self.log('卖出单, %.2f' % self.dataclose[0])
                # 跟踪订单避免重复
                self.order = self.sell()

class PandasData_more(bt.feeds.PandasData):
    lines = ('signal',) # 要添加的线
    # 设置 line 在数据源上的列位置
    params=(
        ('signal', -1),
           )


class MyStrategy(bt.Strategy):
    params = (
        # 均线参数设置15天，15日均线
        ('maperiod', 30),
    )
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    def __init__(self):
        print("--------- 打印 self.datas 第一个数据表格的 lines ----------")
        print(self.data0.lines.getlinealiases())
        self.dayclose=self.datas[0].lines.close
        self.dayopen=self.datas[0].lines.open
        self.signal=self.datas[0].lines.signal
        # 跟踪挂单
        self.order = None
        self.buy_price=0
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('已买入, 价格: %.5f, 费用: %.2f, 佣金 %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                self.buy_price=order.executed.price
            elif order.issell():
                self.log('已卖出, 价格: %.5f, 费用: %.2f, 佣金 %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # 其他状态记录为：无挂起订单
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('交易利润, 毛利润 %.2f, 净利润 %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        #记录每日收盘价
        self.log('Close, %.5f' % self.dayclose[0])

        # 如果有订单正在挂起，不操作
        if self.order:
            return
        if not self.position:
            if self.signal[0]>0 :
                # 买入
                self.log('信号买入, %.5f' % self.dayclose[0])
                self.order=self.buy()
        elif self.position:
            if self.signal[0]<0 :
                #卖出
                self.log('信号卖出, %.5f' % self.dayclose[0])
                self.order=self.sell()
            #设置止损
            elif  self.dayclose[0]<self.buy_price*0.995:
                #卖出
                self.log('止损卖出, %.2f' % self.dayclose[0])
                self.order = self.sell()
            #设置止盈
            elif  self.dayclose[0]>self.buy_price*1.05:
                #卖出
                self.log('止盈卖出, %.2f' % self.dayclose[0])
                self.order = self.sell()




if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # 增加一个策略
    cerebro.addstrategy(MyStrategy)

    # 获取数据
    exchange_df = pd.read_excel("回测数据集/back_test_data.xlsx",index_col='Date',parse_dates=True)
    print(exchange_df)
    start_date = datetime(2023, 5, 30)  # 回测开始时间
    end_date = datetime(2025, 3,19 )  # 回测结束时间
    data = PandasData_more(dataname=exchange_df, # 数据
        fromdate=start_date, # 读取的起始时间
        todate=end_date, # 读取的结束时间
        #nullvalue=0.0, # 缺失值填充
        open=0,
        signal=2,
        close=1,
        low=3,
        high=4,
        volume=5,)
        #openinterest=-1)  # 加载数据
    cerebro.adddata(data,name='exchange')  # 将数据传入回测系统

    # 设置初始资金
    cerebro.broker.setcash(800000.0)
    #设置费率
    cerebro.broker.setcommission(commission=0.00001)
    #设置每次买卖的股数量
    cerebro.addsizer(bt.sizers.FixedSize, stake=100000)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # # 创建并显示图表
    b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
    cerebro.plot(b)
