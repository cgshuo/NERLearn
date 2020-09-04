# -*- coding:utf-8 -*-
# @Time: 2020/9/4 8:15 PM
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: read_data.py

import os
import pandas as pd

def read_data():
    filepath =[]
    datas = list()
    labels = list()
    linedata = list()
    linelabel = list()
    tags = set()
    filenames = os.listdir('../ACE2005')
    for filename in filenames:
        print(filename)
        with open(os.path.join('../ACE2005', filename), 'r') as f:
            for line in f.readlines():
                line = line.split()
                linedata = []
                linelabel = []
                numNotO = 0
                for word in line:
                    word = word.split('/')
                    linedata.append(word[0])
                    linelabel.append(word[1])
                    tags.add(word[1])
                    if word[1] != 'O':
                        numNotO += 1
                if numNotO != 0:
                    datas.append(linedata)
                    labels.append(linelabel)



    return filepath



