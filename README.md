# YOLOX_paddle
【飞桨论文复现挑战赛（第四期）】论文序号9 复现

论文名称: YOLOX: Exceeding YOLO Series in 2021

测试结果

![Lady (1)](https://user-images.githubusercontent.com/26295563/133543628-95c3cdb1-7f0e-4aec-bfc1-835ffeb0adcf.jpg)

![image](https://user-images.githubusercontent.com/26295563/133545014-2afcfa60-d994-48da-8ff1-536fc7346b27.png)

cd paddle版本/YOLOX

YOLOX-s 模型评估

需要把eval的coco数据集val2017放在/paddle版本/YOLOX/datasets/COCO/目录下，与annotataions同一级。详见datasets。


pip install paddleslim

pip install pycocotools

pip install loguru


export YOLOX_DATADIR=/home/aistudio/paddle版本/YOLOX/datasets

python tools/eval.py -n  yolox-s -c yolox_s.pdparams -b 64 -d 1 --conf 0.001 [--fp16] [--fuse]

--fuse: fuse conv and bn
-d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
-b: total batch size across on all GPUs


复现中评估部分出现转换参数问题，读参后模型输出：
![image](https://user-images.githubusercontent.com/26295563/136894032-78a64a7e-1819-4680-b1a6-b4a422c68ec2.png)



