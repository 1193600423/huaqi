<template>
  <main>
    <div class="container">
      <div class="market-image">
        <img src="../assets/history4.jpg" alt="Market Image" style="width: 100%; height: 100%; object-fit: cover;">
      </div>
      <div class="event-interpretation">
        <h3>重要事件节点解读</h3>
        <h4>1. 2016 年：日本央行实施负利率政策</h4>
        <ul>
          <li>时间节点：2016 年 1 月 29 日</li>
          <li>事件背景：日本央行（BOJ）宣布实施负利率政策，将存款利率降至 -0.1%，以刺激经济增长并提高通胀率。</li>
          <li>影响：
            <ul>
              <li>日元短期大幅贬值，日元兑人民币汇率下跌。</li>
              <li>但随后市场对日本央行政策效果产生质疑，避险需求上升，日元在 2016 年下半年反弹。</li>
            </ul>
          </li>
        </ul>
        <hr>
        <h4>2. 2017-2018 年：中美贸易战及避险情绪影响</h4>
        <ul>
          <li>时间节点：2018 年 3 月起</li>
          <li>事件背景：
            <ul>
              <li>2017 年底至 2018 年，中美贸易摩擦加剧，美国对华商品加征关税。</li>
              <li>全球避险情绪上升，推动资金流入日元等避险资产。</li>
            </ul>
          </li>
          <li>影响：
            <ul>
              <li>2018 年期间，日元对人民币一度走强，因投资者寻求避险资产。</li>
              <li>但由于日本经济基本面较弱，日元升值空间受限。</li>
            </ul>
          </li>
        </ul>
        <hr>
        <h4>3. 2020 年：新冠疫情爆发及全球金融市场动荡</h4>
        <ul>
          <li>时间节点：2020 年 3 月</li>
          <li>事件背景：
            <ul>
              <li>新冠疫情全球扩散，全球市场大幅震荡，投资者寻求避险资产。</li>
              <li>美联储紧急降息，日本央行维持超宽松政策。</li>
            </ul>
          </li>
          <li>影响：
            <ul>
              <li>2020 年 3 月，全球避险资金流向日元，日元对人民币短期走强。</li>
              <li>但中国疫情控制较快，经济率先复苏，人民币在下半年逐步反弹，CNY/JPY 走强。</li>
            </ul>
          </li>
        </ul>
        <hr>
        <h4>4. 2022 年：日本央行宽松政策 vs. 美联储激进加息</h4>
        <ul>
          <li>时间节点：2022 年全年</li>
          <li>事件背景：
            <ul>
              <li>美联储自 2022 年 3 月起多次加息，日美利差扩大，日元大幅贬值。</li>
              <li>日本央行仍维持宽松政策，长期国债收益率被压低。</li>
            </ul>
          </li>
          <li>影响：
            <ul>
              <li>日元大幅贬值，对人民币汇率持续走低，2022 年 10 月日元兑人民币跌破 5。</li>
              <li>2022 年 12 月，日本央行调整 YCC（收益率曲线控制）政策，导致日元短期反弹。</li>
            </ul>
          </li>
        </ul>
        <hr>
        <h4>5. 2023-2024 年：日本央行政策转向及人民币波动</h4>
        <ul>
          <li>时间节点：2023 年 10 月 - 2024 年初</li>
          <li>事件背景：
            <ul>
              <li>日本央行暗示未来可能退出负利率政策，市场预期日元走强。</li>
              <li>2024 年 3 月，日本央行正式结束负利率政策，日元迎来阶段性升值。</li>
            </ul>
          </li>
          <li>影响：
            <ul>
              <li>2023 年底至 2024 年初，日元对人民币汇率有所反弹。</li>
              <li>但由于全球经济不确定性和中日利差仍然存在，日元升值幅度受限。</li>
            </ul>
          </li>
        </ul>
      </div>
    </div>
  </main>
</template>

<script setup lang="ts">
</script>

<style scoped>
.container {
  padding-top: 60px; /* Adjust this value based on the height of your NavBar */
  display: flex;
  flex-direction: column;
  align-items: center;
}

.market-image {
  width: 100%;
  max-width: 800px;
  height: 400px; /* Placeholder height */
  margin-bottom: 20px;
  background-color: rgba(200, 200, 200, 0.5); /* Light gray background as a placeholder */
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.event-interpretation {
  width: 100%;
  margin-bottom: 20px;
  max-width: 800px;
  background-color: rgba(255, 255, 255, 0.8);
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>