<template>
  <el-header class="navbar">
    <div class="navbar-content">
      <h1 class="app-name">HUAQI</h1>

      <div class="nav-buttons">
        <el-button type="primary" @click="goToHome" class="nav-button">
          总览
        </el-button>
        <el-button
          type="primary"
          @click="goToBackTest"
          class="nav-button"
        >
          回测
        </el-button>
        <el-button
          type="primary"
          @click="goToHistoryValue"
          class="nav-button"
        >
          历史行情
        </el-button>
      </div>

      <el-button @click="logout" class="logout-button">
        <img src="/images/logout.svg" alt="Logout" class="logout-icon" />
      </el-button>
    </div>
  </el-header>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const goToHome = () => {
  router.push({ name: 'Home' })
}

const goToBackTest = () => {
  router.push({ name: 'BackTest' })
}

const goToHistoryValue = () => {
  router.push({ name: 'HistoryValue' })
}

const logout = () => {
  console.log('Logging out')
  router.push({ name: 'Login' })
}
</script>

<style scoped>
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  background-color: #f4f4f4;
  color: #000;
  position: fixed;
  top: 0;
  width: 100%;
  z-index: 1000;
}

.navbar-content {
  display: flex;
  justify-content: center; /* Center align the content horizontally */
  align-items: center;
  gap: 30px;
  width: 100%;
}

.app-name {
  font-size: 30px;
  margin-right: 20px;
}

.nav-buttons {
  display: flex;
  gap: 50px;
}

.nav-button {
  background-color: transparent;
  color: #000;
  font-size: 18px;
  font-family: "Microsoft YaHei UI", serif;
  border-radius: 8px;
  padding-left: 15px; /* Add left padding */
  padding-right: 15px; /* Add right padding */
  height: 100%;
}

.logout-button {
  background-color: transparent;
  border: none;
  margin-right: 10px; /* Adjust this value to set the desired distance from the right edge */
  margin-left: auto; /* This will push the button to the far right */
}

.logout-icon {
  width: 20px;
  height: 20px;
}
</style>
