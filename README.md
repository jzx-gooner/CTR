# CTR

### 流程

#### 1. 将所提供图片放入source文件夹，生成所提供原始图片的特征

``` Bash
python generate_descriptors.py
```


#### 2. 运行find_you.py，输入链接所在的行列，eg（3,19）获取类别，裁剪校正图片

``` Bash
python find_you.py
```

#### 3. ocr

#### 4. 校准