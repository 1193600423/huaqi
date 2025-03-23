package com.example.back.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author simba@onlying.cn
 * @date 2025/3/8 15:12
 */
@RestController
@RequestMapping("/api/hello")
public class HelloController {

    @GetMapping
    public String hello(){
        return "hello";
    }
}
