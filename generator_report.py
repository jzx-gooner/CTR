# -*- coding:utf-8 -*-
# 生成报告 
import os
import xlwt
import glob
def get_dir():
    return os.getcwd()

def write_excel(sku):
    # 按位置添加数据，col表示列的意思
    dir_col=0
    file_col=1
    row_init=0
    dir = "./"+sku
    print(dir)
    pic_list = glob.glob(dir+"/*")
    #print(pic_list)
    #写表头
    header = ["产品主题","时间","识别结果"]
    i=0
    for each_header in header:
            sh.write(0, i, each_header)
            i += 1
    #设置列宽
    sh.col(2).width = 6000  #列  宽度
    for row, pic in enumerate(pic_list):
        url = pic
        sku_name = pic.split("/")[-2]
        sku_time = pic.split("/")[-1].split(".")[0]
        sh.write(row+1,0,sku_name)  
        sh.write(row+1,1,sku_time)
        sh.write(row+1,2,xlwt.Formula('HYPERLINK("%s")'%url))    

skus_list = list(set(glob.glob("*"))-set(["generator_report.py"]))
print(skus_list)    
wb = xlwt.Workbook()
for sku in skus_list:
    sh = wb.add_sheet(sku)
    write_excel(sku)
wb.save("报告.xls")
