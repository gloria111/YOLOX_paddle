# YOLOX_paddle
【飞桨论文复现挑战赛（第四期）】论文序号9 复现

论文名称: YOLOX: Exceeding YOLO Series in 2021

测试结果

![Lady (1)](https://user-images.githubusercontent.com/26295563/133543628-95c3cdb1-7f0e-4aec-bfc1-835ffeb0adcf.jpg)

![image](https://user-images.githubusercontent.com/26295563/133545014-2afcfa60-d994-48da-8ff1-536fc7346b27.png)

复现没有实现train部分，复现失败

2021-09-30 16:03:14 | INFO     | __main__:151 - Model Structure:
YOLOX(
  (backbone): YOLOPAFPN(
    (backbone): CSPDarknet(
      (stem): Focus(
        (conv): BaseConv(
          (conv): Conv2D(12, 32, kernel_size=[3, 3], padding=1, data_format=NCHW)
          (bn): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
          (act): Silu()
        )
      )
      (dark2): Sequential(
        (0): BaseConv(
          (conv): Conv2D(32, 64, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
          (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
          (act): Silu()
        )
        (1): CSPLayer(
          (conv1): BaseConv(
            (conv): Conv2D(64, 32, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(64, 32, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv3): BaseConv(
            (conv): Conv2D(64, 64, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (m): Sequential(
            (0): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(32, 32, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(32, 32, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
          )
        )
      )
      (dark3): Sequential(
        (0): BaseConv(
          (conv): Conv2D(64, 128, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
          (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          (act): Silu()
        )
        (1): CSPLayer(
          (conv1): BaseConv(
            (conv): Conv2D(128, 64, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(128, 64, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv3): BaseConv(
            (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (m): Sequential(
            (0): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(64, 64, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
            (1): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(64, 64, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
            (2): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(64, 64, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
          )
        )
      )
      (dark4): Sequential(
        (0): BaseConv(
          (conv): Conv2D(128, 256, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
          (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
          (act): Silu()
        )
        (1): CSPLayer(
          (conv1): BaseConv(
            (conv): Conv2D(256, 128, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(256, 128, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv3): BaseConv(
            (conv): Conv2D(256, 256, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (m): Sequential(
            (0): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
            (1): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
            (2): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
          )
        )
      )
      (dark5): Sequential(
        (0): BaseConv(
          (conv): Conv2D(256, 512, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
          (bn): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
          (act): Silu()
        )
        (1): SPPBottleneck(
          (conv1): BaseConv(
            (conv): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (m): Sequential(
            (0): MaxPool2D(kernel_size=1, stride=1, padding=0)
            (1): MaxPool2D(kernel_size=2, stride=1, padding=1)
            (2): MaxPool2D(kernel_size=3, stride=1, padding=1)
          )
          (conv2): BaseConv(
            (conv): Conv2D(1024, 512, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
        )
        (2): CSPLayer(
          (conv1): BaseConv(
            (conv): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv3): BaseConv(
            (conv): Conv2D(512, 512, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (m): Sequential(
            (0): Bottleneck(
              (conv1): BaseConv(
                (conv): Conv2D(256, 256, kernel_size=[1, 1], data_format=NCHW)
                (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
              (conv2): BaseConv(
                (conv): Conv2D(256, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
                (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
                (act): Silu()
              )
            )
          )
        )
      )
    )
    (upsample): Upsample(scale_factor=2, mode=nearest, align_corners=False, align_mode=0, data_format=NCHW)
    (lateral_conv0): BaseConv(
      (conv): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
      (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
      (act): Silu()
    )
    (C3_p4): CSPLayer(
      (conv1): BaseConv(
        (conv): Conv2D(512, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv2): BaseConv(
        (conv): Conv2D(512, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv3): BaseConv(
        (conv): Conv2D(256, 256, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (m): Sequential(
        (0): Bottleneck(
          (conv1): BaseConv(
            (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
            (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
        )
      )
    )
    (reduce_conv1): BaseConv(
      (conv): Conv2D(256, 128, kernel_size=[1, 1], data_format=NCHW)
      (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      (act): Silu()
    )
    (C3_p3): CSPLayer(
      (conv1): BaseConv(
        (conv): Conv2D(256, 64, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv2): BaseConv(
        (conv): Conv2D(256, 64, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv3): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (m): Sequential(
        (0): Bottleneck(
          (conv1): BaseConv(
            (conv): Conv2D(64, 64, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
            (bn): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
        )
      )
    )
    (bu_conv2): BaseConv(
      (conv): Conv2D(128, 128, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
      (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      (act): Silu()
    )
    (C3_n3): CSPLayer(
      (conv1): BaseConv(
        (conv): Conv2D(256, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv2): BaseConv(
        (conv): Conv2D(256, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv3): BaseConv(
        (conv): Conv2D(256, 256, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (m): Sequential(
        (0): Bottleneck(
          (conv1): BaseConv(
            (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
            (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
        )
      )
    )
    (bu_conv1): BaseConv(
      (conv): Conv2D(256, 256, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
      (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
      (act): Silu()
    )
    (C3_n4): CSPLayer(
      (conv1): BaseConv(
        (conv): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv2): BaseConv(
        (conv): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (conv3): BaseConv(
        (conv): Conv2D(512, 512, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=512, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (m): Sequential(
        (0): Bottleneck(
          (conv1): BaseConv(
            (conv): Conv2D(256, 256, kernel_size=[1, 1], data_format=NCHW)
            (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
          (conv2): BaseConv(
            (conv): Conv2D(256, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
            (bn): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
            (act): Silu()
          )
        )
      )
    )
  )
  (head): YOLOXHead(
    (stems): Sequential(
      (0): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (1): BaseConv(
        (conv): Conv2D(256, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (2): BaseConv(
        (conv): Conv2D(512, 128, kernel_size=[1, 1], data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
    )
    (cls_convs): Sequential(
      (0): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (1): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (2): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (3): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (4): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (5): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
    )
    (reg_convs): Sequential(
      (0): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (1): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (2): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (3): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (4): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
      (5): BaseConv(
        (conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (act): Silu()
      )
    )
    (cls_preds): Sequential(
      (0): Conv2D(128, 80, kernel_size=[1, 1], data_format=NCHW)
      (1): Conv2D(128, 80, kernel_size=[1, 1], data_format=NCHW)
      (2): Conv2D(128, 80, kernel_size=[1, 1], data_format=NCHW)
    )
    (reg_preds): Sequential(
      (0): Conv2D(128, 4, kernel_size=[1, 1], data_format=NCHW)
      (1): Conv2D(128, 4, kernel_size=[1, 1], data_format=NCHW)
      (2): Conv2D(128, 4, kernel_size=[1, 1], data_format=NCHW)
    )
    (obj_preds): Sequential(
      (0): Conv2D(128, 1, kernel_size=[1, 1], data_format=NCHW)
      (1): Conv2D(128, 1, kernel_size=[1, 1], data_format=NCHW)
      (2): Conv2D(128, 1, kernel_size=[1, 1], data_format=NCHW)
    )
    (l1_loss): L1Loss()
    (bcewithlog_loss): BCEWithLogitsLoss()
    (iou_loss): IOUloss()
  )
)
2021-09-30 16:03:14 | ERROR    | yolox.core.launch:100 - An error has been caught in function 'launch', process 'MainProcess' (26655), thread 'MainThread' (140684119906048):
Traceback (most recent call last):

  File "tools/eval.py", line 225, in <module>
    args=(exp, args, num_gpu),
          │    │     └ 1
          │    └ Namespace(batch_size=64, ckpt='yolox_s.pth', conf=0.001, devices=1, dist_backend='nccl', dist_url=None, exp_file=None, experi...
          └ ╒══════════════════╤═════════════════════════════════════════════════════════════════════════════════════════════════════════...

> File "/home/aistudio/paddle版本/YOLOX/yolox/core/launch.py", line 100, in launch
    main_func(*args)
    │          └ (╒══════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════...
    └ <function main at 0x7ff391a7ef80>

  File "tools/eval.py", line 153, in main
    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
                │   │             │    │           │               │    │     │    └ False
                │   │             │    │           │               │    │     └ Namespace(batch_size=64, ckpt='yolox_s.pth', conf=0.001, devices=1, dist_backend='nccl', dist_url=None, exp_file=None, experi...
                │   │             │    │           │               │    └ False
                │   │             │    │           │               └ Namespace(batch_size=64, ckpt='yolox_s.pth', conf=0.001, devices=1, dist_backend='nccl', dist_url=None, exp_file=None, experi...
                │   │             │    │           └ False
                │   │             │    └ 64
                │   │             └ Namespace(batch_size=64, ckpt='yolox_s.pth', conf=0.001, devices=1, dist_backend='nccl', dist_url=None, exp_file=None, experi...
                │   └ <function Exp.get_evaluator at 0x7ff2cc932320>
                └ ╒══════════════════╤═════════════════════════════════════════════════════════════════════════════════════════════════════════...

  File "/home/aistudio/paddle版本/YOLOX/yolox/exp/yolox_base.py", line 339, in get_evaluator
    val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
                 │    │               │           │               │        └ False
                 │    │               │           │               └ False
                 │    │               │           └ False
                 │    │               └ 64
                 │    └ <function Exp.get_eval_loader at 0x7ff2cc932290>
                 └ ╒══════════════════╤═════════════════════════════════════════════════════════════════════════════════════════════════════════...

  File "/home/aistudio/paddle版本/YOLOX/yolox/exp/yolox_base.py", line 312, in get_eval_loader
    preproc=ValTransform(legacy=legacy),
            │                   └ False
            └ <class 'yolox.data.data_augment.ValTransform'>

  File "/home/aistudio/paddle版本/YOLOX/yolox/data/datasets/coco.py", line 41, in __init__
    data_dir = os.path.join(get_yolox_datadir(), "COCO")
               │  │    │    └ <function get_yolox_datadir at 0x7ff389236440>
               │  │    └ <function join at 0x7ff392ebf560>
               │  └ <module 'posixpath' from '/opt/conda/envs/python35-paddle120-env/lib/python3.7/posixpath.py'>
               └ <module 'os' from '/opt/conda/envs/python35-paddle120-env/lib/python3.7/os.py'>

  File "/home/aistudio/paddle版本/YOLOX/yolox/data/dataloading.py", line 63, in get_yolox_datadir
    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
                 │  │    │       │  │    │       │     └ None
                 │  │    │       │  │    │       └ <module 'yolox' (namespace)>
                 │  │    │       │  │    └ <function dirname at 0x7ff392ebf830>
                 │  │    │       │  └ <module 'posixpath' from '/opt/conda/envs/python35-paddle120-env/lib/python3.7/posixpath.py'>
                 │  │    │       └ <module 'os' from '/opt/conda/envs/python35-paddle120-env/lib/python3.7/os.py'>
                 │  │    └ <function dirname at 0x7ff392ebf830>
                 │  └ <module 'posixpath' from '/opt/conda/envs/python35-paddle120-env/lib/python3.7/posixpath.py'>
                 └ <module 'os' from '/opt/conda/envs/python35-paddle120-env/lib/python3.7/os.py'>

  File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/posixpath.py", line 156, in dirname
    p = os.fspath(p)
        │  │      └ None
        │  └ <built-in function fspath>
        └ <module 'os' from '/opt/conda/envs/python35-paddle120-env/lib/python3.7/os.py'>

TypeError: expected str, bytes or os.PathLike object, not NoneType
Exception ignored in: <function COCODataset.__del__ at 0x7ff2cc998290>
Traceback (most recent call last):
  File "/home/aistudio/paddle版本/YOLOX/yolox/data/datasets/coco.py", line 62, in __del__
    del self.imgs
AttributeError: imgs
