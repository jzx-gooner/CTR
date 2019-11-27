#coding=utf-8
from __future__ import print_function, unicode_literals
import sys
sys.path.append("../")
import jieba
jieba.load_userdict("userdict.txt")
import jieba.posseg as pseg

ocr_list=[
    [("衡水","老白干"),"衡水老白干真好喝"],
     [("济南","趵突泉"),"济南趵突泉真好喝"]]

def find_match_words(test_sent):
    words = jieba.cut(test_sent)
    for word in words:
        if word in ocr_list[0][0]:
            print (ocr_list[0][1])
            break
        elif word in ocr_list[1][0]:
            print (ocr_list[1][1])
            break
        else:
            print("can not do it")

if __name__ == '__main__':
    test_sent = ("衡水水水打了老老白干干一拳头")
    find_match_words(test_sent)
    test_sent = ("济南趵突泉打了干干一拳头")
    find_match_words(test_sent)
    
