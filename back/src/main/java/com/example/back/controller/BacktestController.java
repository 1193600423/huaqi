package com.example.back.controller;
import org.springframework.web.bind.annotation.*;
import java.io.*;

/**
 * @author simba@onlying.cn
 * @date 2025/3/11 17:33
 */
@RestController
@RequestMapping("/backtest")
public class BacktestController {

//    @GetMapping("/run")
//    public String runBacktest() {
//        try {
//            Process process = Runtime.getRuntime().exec("python3 python-scripts/backtest.py");
//            process.waitFor();
//
//            // 返回 HTML 文件路径
//            return "/static/backtest_result.html";
//        } catch (Exception e) {
//            return "Error: " + e.getMessage();
//        }
//    }

    @GetMapping("/get")
    // 获取回测结果的静态html
    public String getBacktestResult() {
        // 返回 HTML 文件路径
        return "/static/backtest_result.html";
    }
}
