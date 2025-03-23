<template>
  <div class="back-test-container">
    <div class="charts">
      <div class="chart">
        <h3>收益曲线图</h3>
        <!-- Add your profit curve chart here -->
        <iframe src="http://localhost:8080/bt_bokeh_plot_0.html" width="100%" height="800px"></iframe>
      </div>
      <div class="chart">
        <h3>预测曲线图</h3>
        <!-- Add your prediction curve chart here -->
        <img src="../assets/predict.png" alt="Prediction Chart" style="width: 100%; height: 500px; object-fit: contain;">
      </div>
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped>
.back-test-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding-top: 60px; /* Adjust this value based on the height of your NavBar */
}

.charts {
  display: flex;
  justify-content: space-between;
  flex: 1;
  margin-bottom: 20px;
}

.chart {
  flex: 1;
  margin: 0 10px;
  padding: 20px;
  background-color: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 10px;
  border: 1px solid #ddd;
  text-align: left;
}

th {
  background-color: #f4f4f4;
}
</style>
