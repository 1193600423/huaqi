package com.example.back.Exception;

/**
 * @author simba@onlying.cn
 * @date 2025/3/16 22:12
 */
public class Exception extends RuntimeException{
    public Exception(String message){
        super(message);
    }

    // 爬取失败，错误码
    public static Exception getException(int code){
        return new Exception("爬取失败，错误码：" + code);
    }

    // 运行 Python 爬虫失败
    public static Exception getException(){
        return new Exception("运行 Python 爬虫失败");
    }

}
