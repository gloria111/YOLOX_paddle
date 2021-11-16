# YOLOX_paddle
【飞桨论文复现挑战赛（第四期）】论文序号9 复现

论文名称: YOLOX: Exceeding YOLO Series in 2021
===========================
测试结果

![Lady (1)](https://user-images.githubusercontent.com/26295563/133543628-95c3cdb1-7f0e-4aec-bfc1-835ffeb0adcf.jpg)

![image](https://user-images.githubusercontent.com/26295563/133545014-2afcfa60-d994-48da-8ff1-536fc7346b27.png)

YOLOX-s 模型评估
-----------
![image](https://user-images.githubusercontent.com/26295563/138433867-f6e6d4e5-2a25-40e2-8b70-dc10684506c1.png)
在cpu下的评估结果如上图所示

需要把eval的coco数据集val2017放在/paddle版本/YOLOX/datasets/COCO/目录下，与annotataions同一级。详见datasets。

    pip install paddle
    pip install pycocotools
    pip install loguru
    export YOLOX_DATADIR=/home/aistudio/paddle版本/YOLOX/datasets
    python tools/eval.py -n  yolox-s -c yolox_s.pdparams -b 64 -d 1 --conf 0.001 [--fp16] [--fuse]

######   fuse: fuse conv and bn
######   d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
######   b: total batch size across on all GPUs

[yolox_s的paddle权重文件链接:链接: https://pan.baidu.com/s/1QHLmOVMyBxXA3YSicbyYng 提取码: u2uy]( https://pan.baidu.com/s/1QHLmOVMyBxXA3YSicbyYng)。

复现中评估部分出现问题，分类为背景，得分过低。
转换参数参考https://zhuanlan.zhihu.com/p/188744602

评估部分问题：

######   1.在darknet层特征图出现较大错误
训练部分问题：

######   1.显存溢出





