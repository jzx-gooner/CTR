#coding = utf-8
# jzx

import pandas as pd 

df = pd.read_excel("./框架数据明细.xlsx")

print("输出值\n",df["电梯间照片编号1"].values)