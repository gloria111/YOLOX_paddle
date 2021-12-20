# YOLOX_paddle
【飞桨论文复现挑战赛（第五期）】论文序号22 复现

论文名称: YOLOX: Exceeding YOLO Series in 2021
===========================
测试结果


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

######   在darknet层特征图出现较大错误
训练部分问题：

######   依赖库原因,将mem暂设为0
######  CUDA_LAUNCH_BLOCKING=1 python tools/train.py -n yolox-s -d 1 -b 1 --fp16


![image](https://user-images.githubusercontent.com/26295563/142130686-e8871c34-a5b7-4009-8e61-932ba13a2b14.png)

训练几个iter后出现
![image](https://user-images.githubusercontent.com/26295563/143725770-924ccd0e-450a-4cbe-90df-c78685057ffd.png)

issue链接https://github.com/PaddlePaddle/Paddle/issues/37665
