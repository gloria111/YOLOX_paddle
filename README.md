# YOLOX_paddle
【飞桨论文复现挑战赛（第四期）】论文序号9 复现

论文名称: YOLOX: Exceeding YOLO Series in 2021

测试结果

![Lady (1)](https://user-images.githubusercontent.com/26295563/133543628-95c3cdb1-7f0e-4aec-bfc1-835ffeb0adcf.jpg)

![image](https://user-images.githubusercontent.com/26295563/133545014-2afcfa60-d994-48da-8ff1-536fc7346b27.png)

复现没有实现train部分，复现失败

YOLOX-s 模型评估
python tools/eval.py -n  yolox-s -c yolox_s.pth -b 64 -d 1 --conf 0.001 [--fp16] [--fuse]

--fuse: fuse conv and bn
-d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
-b: total batch size across on all GPUs
