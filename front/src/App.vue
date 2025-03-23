<template>
  <div id="app">
    <NavBar v-if="showNavBar" />
    <router-view />
  </div>
</template>

<script setup>
import { useRoute } from 'vue-router'
import { computed } from 'vue'
import NavBar from './components/NavBar.vue'

const route = useRoute()
const showNavBar = computed(() => !['/login', '/register'].includes(route.path))
</script>

<style>
/* Add your styles here */
</style>
