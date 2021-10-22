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

    pip install paddleslim
    pip install pycocotools
    pip install loguru
    export YOLOX_DATADIR=/home/aistudio/paddle版本/YOLOX/datasets
    python tools/eval.py -n  yolox-s -c yolox_s.pdparams -b 64 -d 1 --conf 0.001 [--fp16] [--fuse]

######   fuse: fuse conv and bn
######   d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
######   b: total batch size across on all GPUs

[yolox_s的paddle权重文件链接:https://pan.baidu.com/s/1F7VjGPOlg6pLvM8AE62vxw,提取码: nuri ](https://pan.baidu.com/s/1F7VjGPOlg6pLvM8AE62vxw)。

复现中评估部分出现转换参数问题，读参后模型输出：

![截屏2021-10-13 10 04 15](https://user-images.githubusercontent.com/26295563/137054658-465ebb58-4ecb-4b5b-a1f6-f6453900005a.png)

-----------

问题部分：

######   1.评分都为0

######   2.检测目标框只能分类为第一类且置信度过低

######   3.output score设置为nms输出[:,0]，应该是output[:,5]*[:,6]



