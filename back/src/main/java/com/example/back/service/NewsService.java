package com.example.back.service;

import com.example.back.vo.Article;
import com.example.back.Exception.Exception;
import com.example.back.vo.NewsApiResponse;
import org.apache.poi.ss.usermodel.Workbook;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Date;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

/**
 * @author simba@onlying.cn
 * @date 2025/3/8 15:32
 */
@Service
public class NewsService {
    private static final String API_KEY = "b304cc0ee3a84b7f884862e7f6e6b2c5";
    private static final String API_URL = "https://newsapi.org/v2/top-headlines?country=us&apiKey=" + API_KEY;

    @Autowired
    private RestTemplate restTemplate;

    /**
     * 读取 XLSX 文件并返回 Article 对象的列表
     */
    public List<Article> getTopHeadlines() {
        List<Article> articles = new ArrayList<>();
        // 请将文件路径替换为相对的 XLSX 文件路径
        String filePath = "src/main/resources/data/news.xlsx";

        try (FileInputStream fis = new FileInputStream(new File(filePath));
             XSSFWorkbook workbook = new XSSFWorkbook(fis)) {
            Sheet sheet = workbook.getSheetAt(0); // 默认读取第一个 sheet

            Iterator<Row> rowIterator = sheet.iterator();
            // 跳过标题行
            if (rowIterator.hasNext()) {
                rowIterator.next();
            }

            while (rowIterator.hasNext()) {
                Row row = rowIterator.next();
                Article article = new Article();
                // 根据 XLSX 中各列顺序赋值到 Article 对象
                article.setTime(getCellValue(row.getCell(1)));
                article.setText(getCellValue(row.getCell(2)));
                article.setUrl(getCellValue(row.getCell(3)));
                article.setSentiments(getCellValue(row.getCell(4)));
                articles.add(article);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return articles;
    }

    /**
     * 根据单元格类型获取单元格的 String 值
     */
    private String getCellValue(Cell cell) {
        if (cell == null) {
            return "";
        }
        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                if (DateUtil.isCellDateFormatted(cell)) {
                    // 若是日期格式，按自定义格式输出
                    // 2025-03-22
                    Date date = cell.getDateCellValue();
                    System.out.println(date);
                    DateFormat df = new SimpleDateFormat("yyyy-MM-dd");
                    return df.format(date);
                } else {
                    return String.valueOf(cell.getNumericCellValue());
                }
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            case FORMULA:
                // 若需要根据公式计算值，可根据需要调整
                return cell.getCellFormula();
            default:
                return "";
        }
    }
}
