#coding=utf-8
from excel_tools import ExcelTool
from cnn_solution import CnnSolution
from sift_solution import SiftSolution
import text_ocr
import shutil

if __name__ == '__main__':
    excel_path = "./框架数据明细.xlsx"
    source = "./source/"
    excle_tool = ExcelTool()
    img_list = excle_tool.read_excel(excel_path)
    cnn_predict = CnnSolution()
    sift_soulution = SiftSolution()
    # assert len(img_list)==len(set(img_list))
    for query in img_list:
        # solution 1: sift
        best_match_img_name, is_pass_min_match = sift_soulution.find_match_index(source, query)
        if is_pass_min_match:
	    sift_soulution.crop_img(best_match_img_name + '.jpg', query)
            print best_match_img_name.split("/")[-1]
        else:
            shutil.copy(query, "./debug/temp/")
	#solution 2 ：cnn
        result = cnn_predict.cnn_predict(query)
        print (result)

	#text ocr


	#write excel
