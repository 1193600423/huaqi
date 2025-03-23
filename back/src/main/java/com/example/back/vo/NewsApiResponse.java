package com.example.back.vo;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;


import java.util.List;

/**
 * @author simba@onlying.cn
 * @date 2025/3/8 15:34
 */

@Getter
@Setter
@NoArgsConstructor
public class NewsApiResponse {
    private String status;
    private int totalResults;
    private List<Article> articles;

    // Getters & Setters
}
