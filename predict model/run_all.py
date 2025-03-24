# run_all.py
import os
import runpy

list=['数据提取.py','train_data.py','test_data.py','predict_model.py','backtest_data.py','backtest.py']
# 执行每个文件
for file in list:
    runpy.run_path(file)