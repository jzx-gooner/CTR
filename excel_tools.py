# coding=utf-8
import xlrd
import xlsxwriter
import sys
reload(sys)
sys.setdefaultencoding('utf8')
if sys.getdefaultencoding() != 'gbk':
    reload(sys)
    sys.setdefaultencoding('gbk')

class ExcelTool(object):
    def read_excel(self,file):
        data = xlrd.open_workbook(filename=file)  # 打开文件#此时data相当于指向该文件的指针
        table = data.sheet_by_index(0)  # 通过索引获取表格
        names = data.sheet_names()
        print(names[0].encode('utf-8'))
        row = int(raw_input("请输入图片所在的行: "))
        column = int(raw_input("请输入图片所在的列: "))
        col_images = table.col_values(column)
        img_list = []  # 图片索引
        for item_col_images in col_images:
            name = item_col_images.encode("utf-8")
            path = "./images/" + name + ".jpg"
            img_list.append(path)
        # print row_3.index("实际广告内容").encode('utf-8') #获得索引
        print(img_list[row:])
        return img_list[row:]


    def write_excel(self,results, excel_path):
        workbook = xlsxwriter.Workbook(excel_path)
        worksheet = workbook.add_worksheet()
        worksheet.write_column('U4', excel_path)
        workbook.close()
