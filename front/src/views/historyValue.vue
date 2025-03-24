<template>
  <main>
    <div class="container">
      <div class="market-image">
        <img src="../assets/history4.jpg" alt="Market Image" style="width: 100%; height: 100%; object-fit: cover;">
      </div>
      <div class="event-interpretation">
        <h2>人民币汇率变动分析</h2>
        <div v-for="(section, index) in exchangeRateData" :key="index" class="section">
          <h3>{{ section.title }}</h3>
          <p v-html="section.content"></p>
        </div>
      </div>
    </div>
  </main>
</template>

<script setup lang="ts">
import { ref } from "vue";

const exchangeRateData = ref([
  {
    title: "2020年初至2020年中期",
    content: "汇率下降可能与全球疫情爆发有关，导致全球经济不确定性增加，投资者寻求避险资产，如美元，从而导致美元对其他货币的汇率上升。同时，中国疫情控制较好，经济逐步恢复，人民币相对稳定，汇率开始回升。",
  },
  {
    title: "2020年中期至2021年初",
    content: "汇率相对稳定，波动较小，这可能反映了全球经济逐步适应疫情带来的影响，市场情绪趋于稳定。",
  },
  {
    title: "2021年初至2022年初",
    content: "汇率再次出现下降趋势，这可能与美国经济复苏预期增强，美联储开始讨论缩减购债计划有关，导致美元走强。同时，中国经济增长放缓，可能也对人民币形成压力。",
  },
  {
    title: "2022年初至2022年中期",
    content: "汇率出现显著上升，这可能与全球供应链问题、能源价格上涨以及地缘政治紧张局势有关，导致市场对风险资产的需求增加，美元走强。",
  },
  {
    title: "2022年中期至2023年初",
    content: "汇率波动较大，这可能与美联储加息周期的开始有关，美元走强，同时中国经济增长面临挑战，人民币相对承压。同时，人民币汇率波动的主要因素，一是基本面方面，我国国际收支平衡尚处于筑底回升阶段，对市场情绪难完全对冲，令情绪面成为人民币汇率短期波动的主要因素。中美关系的不确定性、市场对非美国家降息速度快于美联储的预期发酵，成为了本次人民币汇率波动的导火索。",
  },
  {
    title: "2023年初至2024年",
    content: "汇率整体呈现上升趋势，这可能与美国经济数据强劲，美联储持续加息有关，美元保持强势。同时，中国经济增长可能有所恢复，但人民币仍面临一定压力。",
  },
  {
    title: "2024年至今",
    content: "汇率再次出现波动，这可能与全球经济不确定性增加，包括地缘政治紧张局势、全球贸易摩擦等因素有关，导致市场对美元的需求增加。同时，汇率的波动将令货币政策操作更趋谨慎，而人民币对美元的波动仍能维持在7.2-7.3之间，后续走势有赖于美联储以及其他央行货币政策操作路径的清晰化。",
  },
]);
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
.section {
  margin-bottom: 20px;
}

h3 {
  color: #2c3e50;
}
</style>
