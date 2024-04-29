# 使用onnxruntime推理自定义模型

模型为GRFB-UNET，原论文即作者源代码连接如下：

论文：[GRFB-UNet: A new multi-scale attention network with group receptive field block for tactile paving segmentation - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417423026118))

Github：https://github.com/Chon2020/GRFB-Unet.git

**论文摘要：**

触觉铺装在视障人士的出行中起着至关重要的作用，帮助他们找到前进的道路。因此，识别触觉铺装的区域和趋势对于支持视障人士的独立行走是非常有意义的。视觉分割技术显示出对触觉铺装区域进行分割的潜力，这些区域的形状可以用来进一步检查道路趋势。为了有效地提高触觉铺装分割的准确性和鲁棒性，本文提出了一种结合UNet网络和多尺度特征提取的新型触觉铺装分割方法。在基本UNet网络中嵌入了组感受野块（GRFB）的结构，以获得触觉铺装的多个尺度的感受野。为了提高计算效率，采用组卷积策略与GRFB模块相结合。同时，每个组卷积之后使用小规模卷积来实现跨通道的信息交互和整合，旨在提取更丰富的高层特征。本文构建了各种场景下的触觉铺装数据集，并进行了实验评估。此外，本文还详细地展示了与典型网络和结构模块的比较分析。实验结果表明，在触觉铺装分割的比较网络中，所提出网络在整体性能方面表现最佳，为触觉铺装分割提供了有价值的参考。

**文件结构如下：**

./
│  grfb-unet.pth	由原论文代码训练1200轮得到的pytorch权重文件
│  grfb_unet.onnx	由论文模型依据权重文件导出的onnx格式的模型
│  model2onnx.py	将模型导出为onnx格式的python代码
│  onnxruntime.cpp	基于onnx格式运行的C++代码
|  onnx_info.cpp	查看onnx模型的输入输出
│  README.md	本文档
│ 
└─images	测试图像
        001.jpg
        002.jpg
        ... ...

**C++代码环境要求**

- 安装opencv （我的版本4.8.1）
- 安装onnxruntime （我的版本1.17.3）

**Python环境要求**

- 参看论文源代码requirement.txt
- pip安装onnx （我的版本1.15.0）