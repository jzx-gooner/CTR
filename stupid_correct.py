#coding=utf-8
from __future__ import print_function, unicode_literals
import jieba
jieba.load_userdict("./userdict.txt")
import jieba.posseg as pseg

ocr_list=[
    [["威宝","0318","5156"],"衡水威宝 0318-5156666"],
     [["经济","交叉口","西南角"],"地址：衡水市经济开发区振华路与冀衡路交叉口西南角"]]

def find_match_words(test_sent):
    words = jieba.cut(test_sent)
    for word in words:
        print(word)
        if word in ocr_list[0][0]:
            print(("关键词<<{}>>匹配上了").format(word))
            return True,ocr_list[0][1]
        elif word in ocr_list[1][0]:
            print(("关键词<<{}>>匹配上了").format(word))
            return True,ocr_list[1][1]
        else:
            print("can not do it")
            
    else:
        print("没有能匹配上的")
        return False,None
if __name__ == '__main__':
    test_sent = ("衡水分割水天气威dd宝交f叉口西f南角 03dfd18-51we56a666")
    flag,result = find_match_words(test_sent)
    print(result)
