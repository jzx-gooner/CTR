# CTR

### 流程
#### 图片匹配
![图片匹配 ](https://github.com/jzx-gooner/CTR/blob/master/docx/match_image.jpg) 

#### 图片裁剪
![图片裁剪 ](https://github.com/jzx-gooner/CTR/blob/master/docx/crop_image.jpg)   

#### 文本定位
![文本定位 ](https://github.com/jzx-gooner/CTR/blob/master/docx/text_detection.jpg)



#### 1. 将所提供图片放入source文件夹，生成所提供原始图片的特征

``` Bash
python generate_descriptors.py
```


#### 2. 运行find_you.py，输入链接所在的行列，eg（3,19）获取类别，裁剪校正图片,sift,cnn两种方法判断类别

``` Bash
python find_you.py
```

#### 3. ocr

``` Bash
python text_ocr.py
```
