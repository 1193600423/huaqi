package com.example.back.vo;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * @author simba@onlying.cn
 * @date 2025/3/8 15:34
 */
@Getter
@Setter
@NoArgsConstructor
public class Article {
    //	time	text	url	sentiments
    private String time;
    private String text;
    private String url;
    private String sentiments;

    // Getters & Setters
    public String getTime() {
        return time;
    }
    public void setTime(String time) {
        this.time = time;
    }
    public String getText() {
        return text;
    }
    public void setText(String text) {
        this.text = text;
    }

    public String getUrl() {
        return url;
    }
    public void setUrl(String url) {
        this.url = url;
    }
    public String getSentiments() {
        return sentiments;
    }
    public void setSentiments(String sentiments) {
        this.sentiments = sentiments;
    }

}
