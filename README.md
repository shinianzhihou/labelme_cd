

# LABELME_CD


> 这个作者很懒，名字都没改，代码没咋变，只增加了个变化检测的标注功能。


https://user-images.githubusercontent.com/29772895/207556216-cb4934b4-f9ed-42c4-a2a2-d8fa9ad1b63e.mp4



基于[labelme](https://github.com/wkentaro/labelme/tree/v5.1.0)构建的用于变化检测的标注工具。



## 安装

```bash
conda create -n labelme_cd python=3.9 -y
conda activate labelme_cd
conda install pyqt -y

cd labelme_cd
python setup.py install

```

## 使用

1. 首先将待标注数据以 `assets/examples/change_detection/` 的方式放好（支持软连接）




2. `A/` 和 `B/` 表示待标注的两个时相目录，下面的命名保持一致， `label/` 用来存储标注结果

3. 通过如下方式启动软件进行标注：

    ```bash
    # option-1
    labelme_cd
    # then chose the specific dir (A/ or B/)


    # option-2
    labelme_cd ./assets/examples/change_detection/A/
    # or
    # labelme_cd ./assets/examples/change_detection/B/

    ```

## 特性

> 这个作者很懒，下面的东西下次再整理放上来 

- 支持两张图分别进行亮度/对比度调节、频域对齐和直方图校正。

- 支持[lazy_cd](https://github.com/shinianzhihou/lazy_cd)中目标/像素级算法辅助标注

## 声明

- This repo is the fork of [wkentaro/labelme](https://github.com/wkentaro/labelme).

- This project is part of the [shinianzhihou/lazy_cd](https://github.com/shinianzhihou/lazy_cd)
