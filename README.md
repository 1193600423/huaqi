# 智能外汇交易分析平台



### 介绍

- 在全球经济一体化背景下，汇率波动对外贸企业、金融机构和个人投资者的影响日益显著。精准预测汇率趋势成为降低风险、优化收益的核心需求。

- 实际的模型实现了 自动化数据处理、智能化模型预测和一键式回测功能，但由于时间、人力限制，此处的平台网页只展示了原型。



### 项目结构

```shell
huaqi
├─back #平台后端
├─front #平台前端
├─项目文件 #完整的模型以及数据
└─README.md #说明文件
```



### 功能模块

##### 总览

展示 当前交易信号、当前预测外汇趋势、当前预测外汇价格，并给出实时操作建议。

右侧展示当日新闻，这些新闻同样也是模型中汇率预测的考虑因素。

![image-20250324091027230](README.assets/image-20250324091027230.png)



##### 回测

展示两张曲线图，分别是收益曲线图、预测曲线图。

预测曲线：基于融合LLM（大语言模型）与时间序列模型的智能分析体系。通过LLM处理新闻、市场情绪等非结构化数据，纳入多维度信息；结合短时间序列模型与LSTM模型，实现实时数据接收与动态预测。

根据预测汇率对 实际汇率的对比，计算模型的回测效果，展示收益曲线图。

![image-20250324091103270](README.assets/image-20250324091103270.png)



##### 历史行情

该模块分为两个子模块：历史行情图、重要历史事件解读。

![image-20250324092049346](README.assets/image-20250324092049346.png)



### 安装步骤

1. **克隆代码仓库**

   ```shell
   git clone https://github.com/1193600423/huaqi.git
   ```

2. **前端安装**

   ```shell
   cd front
   npm install
   npm run dev 
   ```

3. **后端启动**

   - 直接运行 BackApplication



