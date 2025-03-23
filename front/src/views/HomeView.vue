<template>
  <div class="home-container">
    <div class="background-overlay"></div>
    <div class="content">
      <div class="left-column">
        <div class="block current">
          <h2>当前</h2>
          <div class="sub-block">
            <h3>当前交易信号</h3>
            <div class="status-badge good">优</div>
          </div>
          <div class="sub-block">
            <h3>当前预测外汇趋势</h3>
            <div class="status-badge up">涨</div>
          </div>
          <div class="sub-block">
            <h3>当前预测外汇价格</h3>
            <div class="price">5.99</div>
          </div>
        </div>
        <div class="block real-time">
          <h2>实时操作</h2>
          <div class="suggestion-box">建议买入</div>
        </div>
      </div>
      <div class="block news">
        <h2>新闻</h2>
        <ul>
          <li v-for="news in newsList" :key="news.url">
            <a :href="news.url" target="_blank">{{ news.time }}</a>
            <b :href="news.url" target="_blank">{{ news.text }}</b>
            <br>
            <c>Sentiments: </c>
            <c :href="news.url" target="_blank" :style="{ color: getSentimentColor(news.sentiments) }">
              {{ news.sentiments }}
            </c>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';

const newsList = ref([]);

const fetchNews = async () => {
  try {
    console.log('Fetching news...');
    const response = await axios.get('http://localhost:8080/api/news'); // Update the URL to match your backend API endpoint
    console.log('News:', response.data);
    newsList.value = response.data;
  } catch (error) {
    console.error('Error fetching news:', error);
  }
};

const getSentimentColor = (sentiment) => {
  switch (sentiment) {
    case '积极':
      return 'green';
    case '中性':
      return 'gray';
    case '消极':
      return 'red';
    default:
      return 'black';
  }
};

onMounted(() => {
  fetchNews();
});
</script>

<style scoped>
html, body {
  height: 100%;
  margin: 0;
}

.home-container {
  height: 100%;
  padding-top: 60px; /* Adjust this value based on the height of your NavBar */
  position: relative;
  background-size: cover;
  background-position: center;
  background-attachment: fixed; /* Prevent background from moving */
  display: flex;
  justify-content: center;
  align-items: center;
}

.background-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
}

.content {
  display: flex;
  width: 100%;
  height: 100%;
  z-index: 2;
}

.left-column {
  display: flex;
  flex-direction: column;
  width: 50%;
}

.block {
  height: calc(100vh - 80px);
  margin: 10px;
  padding: 20px;
  background-color: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.current {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.current h3 {
  align-self: flex-start;
  margin-top: 20px;
}

.real-time {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.real-time h2 {
  align-self: flex-start;
}

.news {
  width: 50%;
  max-height: 100%;
  overflow-y: auto;
}

.suggestion-box {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background-color: #007bff;
  color: white;
  font-size: 18px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 0 auto;
}

h2 {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 20px;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  margin-bottom: 10px;
  padding: 10px;
  background-color: rgba(255, 255, 255, 0.9);
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

a {
  font-size: 16px;
  text-decoration: none;
  color: #007bff;
}

b {
  margin-left: 10px;
  font-size: 18px;
}

c {
  font-size: 16px;
}

.status-badge {
  margin: 10px 20px;
  display: inline-block;
  padding: 5px 10px;
  border-radius: 4px;
  color: white;
  font-size: 16px;
  font-weight: bold;
}

.status-badge.good {
  background-color: green;
}

.status-badge.up {
  background-color: blue;
}

.price {
  margin: 10px 20px;
  font-size: 24px;
  font-weight: bold;
  color: #333;
}
</style>