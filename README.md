<!--
 * @Author: wuwenbo
 * @Date: 2021-06-02 10:14:54
 * @LastEditTime: 2021-06-20 16:14:49
 * @FilePath: /ESCCAttention/README.md
-->
# Channel Attention Research for Environment Sound Classification

## Dataset
1. [ESC-10](https://github.com/karolpiczak/ESC-50)
2. [ESC-50](https://github.com/karolpiczak/ESC-50)
3. [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

## File
```
ESCCAttention
├── README.md
├── attention
│   ├── cbam.py
│   ├── escc.py
│   ├── se.py
│   └── ts.py
├── dataloaders.py
├── environment.py
├── flops.py
├── models
│   ├── __init__.py
│   ├── cbam_resnet.py
│   ├── densenet.py
│   ├── inception.py
│   ├── resnet.py
│   ├── se_resnet.py
│   └── vit.py
├── preprocessing
│   ├── __init__.py
│   ├── preprocessingESC10.py
│   ├── preprocessingESC50.py
│   └── preprocessingUSC.py
├── test.py
├── train.py
├── utils.py
└── validate.py
```

在我的机器上训练的模型 ResNet 的结果是 [79.83963%, 88.51335%, 75.89189%, 84.94949%, 88.67521%, 80.80194%, 84.00954%, 759,704%, 759,704%] %.39787] 10 倍的平均准确率为 83.70240，略低于论文中的结果（84.76%）。每个折叠的最后一次迭代的平均准确度为 79.1663%。

# Base
## USC
Accuracy of each fold  [0.77090493 0.8018018  0.72324324 0.82525253 0.88461538 0.85783718
 0.83890215 0.77419355 0.84191176 0.84229391]
Mean accuracy of the dataset:  0.8160956429376389

## ESC10
Accuracy of each fold  [0.925  0.95   0.9125 0.975  0.9625]
Accuracy of each fold  [0.925  0.95   0.9125 0.975  0.9625]
Mean accuracy of the dataset:  0.9450000000000001

## ESC50
Accuracy of each fold  [0.83 0.85 0.83 0.86 0.79]
Accuracy of each fold  [0.83 0.85 0.83 0.86 0.79]
Mean accuracy of the dataset:  0.8320000000000001

# SE
## ESC10
Accuracy of each fold  [0.9625 0.95   0.95   0.975  0.9625]
Mean accuracy of the dataset:  0.96

## ESC50
Accuracy of each fold  [0.8825 0.885  0.8725 0.9175 0.8275]
Mean accuracy of the dataset:  0.877