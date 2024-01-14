import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 加载预训练的骨干网络
backbone = torchvision.models.mobilenet_v2(pretrained=True).features

# 定义RPN的anchor generator
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# 定义ROI Pooling层的特征金字塔层
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# 定义Faster R-CNN模型
model = FasterRCNN(backbone=backbone,
                   num_classes=21,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

# 模型推理
inputs = torch.randn(1, 3, 800, 800)
outputs = model(inputs)

# 可以访问输出的“boxes”、“labels”和“scores”
boxes = outputs[0]['boxes']
labels = outputs[0]['labels']
scores = outputs[0]['scores']