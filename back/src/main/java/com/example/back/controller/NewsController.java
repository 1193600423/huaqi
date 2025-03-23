package com.example.back.controller;

import com.example.back.service.NewsService;
import com.example.back.vo.Article;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.List;

/**
 * @author simba@onlying.cn
 * @date 2025/3/8 15:31
 */
@RestController
@RequestMapping("/api/news")
public class NewsController {
    @Autowired
    private NewsService newsService;

    @GetMapping
    public List<Article> getNews() {
        return newsService.getTopHeadlines();
    }

}
