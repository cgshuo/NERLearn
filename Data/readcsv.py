# -*- coding:utf-8 -*-
# @Time: 2020/9/5 3:39 PM
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: readcsv.py

import pandas as pd
import numpy as np


df_data = pd.read_csv('./test.csv')
print(df_data)
df_train = df_data.head(24000)
df_test = df_data.tail(6000)
df_train.to_csv('./train_data', sep = ' ', index = False, header = False)
df_test.to_csv('./test_data', sep = ' ', index = False, header = False )



