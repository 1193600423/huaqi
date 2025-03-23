import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import historyValue from '@/views/historyValue.vue'
import backTest from '@/views/backTest.vue'

const routes = [
  {
    path: '/',
    redirect: '/home',
  },
  {
    path: '/home',
    name: 'Home',
    component: HomeView,
  },
  {
    path: '/historyValue',
    name: 'HistoryValue',
    component: historyValue,
  },
  {
    path: '/backTest',
    name: 'BackTest',
    component: backTest,
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router
