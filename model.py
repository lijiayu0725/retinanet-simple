import math
import torch
from torch import nn

from anchor import generate_anchors, snap_to_anchors
from fpn import ResNet50FPN
from loss import FocalLoss, SmoothL1Loss


class RetinaNet(nn.Module):
    def __init__(self, classes=80, state_dict_path='/Users/nick/.cache/torch/checkpoints/resnet50-19c8e357.pth'):
        super(RetinaNet, self).__init__()
        self.backbone = ResNet50FPN(state_dict_path=state_dict_path)
        self.name = 'RetinaNet'
        self.ratios = [0.5, 1.0, 2.0]
        self.scales = [4 * 2 ** (i / 3) for i in range(3)]
        self.anchors = {}
        self.classes = classes

        self.threshold = 0.05
        self.top_n = 1000
        self.nms = 0.5
        self.detections = 100

        self.stride = self.backbone.stride

        def make_head(out_size):
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(256, out_size, 3, padding=1)]
            return nn.Sequential(*layers)

        num_anchors = len(self.ratios) * len(self.scales)
        self.cls_head = make_head(classes * num_anchors)
        self.box_head = make_head(4 * num_anchors)
        self.initialize()

        self.cls_criterion = FocalLoss()
        self.box_criterion = SmoothL1Loss()

    def initialize(self):
        self.backbone.initialize()

        def initialize_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        self.cls_head.apply(initialize_layer)
        self.box_head.apply(initialize_layer)

        def initialize_prior(layer):
            pi = 0.01
            b = - math.log((1 - pi) / pi)
            nn.init.constant_(layer.bias, b)
            nn.init.normal_(layer.weight, std=0.01)

        self.cls_head[-1].apply(initialize_prior)

    def forward(self, x):
        if self.training:
            x, targets = x

        features = []
        features.extend(self.backbone(x))

        cls_heads = [self.cls_head(t) for t in features]
        box_heads = [self.box_head(t) for t in features]

        if self.training:
            return self.loss(x, cls_heads, box_heads, targets.float())

    def loss(self, x, cls_heads, box_heads, targets):
        cls_losses, box_losses, fg_targets = [], [], []
        for cls_head, box_head in zip(cls_heads, box_heads):
            # compute each feature's loss in fpn
            size = cls_head.shape[-2:]
            stride = x.shape[-1] / cls_head.shape[-1]

            cls_target, box_target, depth = self.extract_targets(targets, stride, size)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))

            cls_head = cls_head.view_as(cls_target).float()
            cls_mask = (depth >= 0).expand_as(cls_target).float()
            cls_loss = self.cls_criterion(cls_head, cls_target)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())

            box_head = box_head.view_as(box_target).float()
            box_mask = (depth > 0).expand_as(box_target).float()
            box_loss = self.box_criterion(box_head, box_target)
            box_loss = box_mask * box_loss
            box_losses.append(box_loss.sum())

        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        box_loss = torch.stack(box_losses).sum() / fg_targets
        return cls_loss, box_loss

    def extract_targets(self, targets, stride, size):
        cls_target, box_target, depth = [], [], []
        for target in targets:
            target = target[target[:, -1] > -1]  # ignore the padding target in dataset
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)
            snapped = snap_to_anchors(target,
                                      [s * stride for s in size[::-1]],
                                      stride, self.anchors[stride].to(targets.device),
                                      self.classes)
            for l, s in zip((cls_target, box_target, depth), snapped):
                l.append(s)
        return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)

    def fix_bn(self):
        def fix_batchnorm_param(layer):
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        self.apply(fix_batchnorm_param)

    def train(self, mode=True):
        super(RetinaNet, self).train(mode)
        self.fix_bn()
        return self
