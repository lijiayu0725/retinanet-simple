import torch


def generate_anchors(stride, ratios_vals, scales_vals):
    scales = torch.FloatTensor(scales_vals).repeat(len(ratios_vals), 1)
    scales = scales.transpose(0, 1).reshape(-1, 1)
    ratios = torch.FloatTensor(ratios_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.round(torch.sqrt(wh[:, 0] * wh[:, 1] / ratios))
    dwh = torch.stack([ws, torch.round(ws * ratios)], dim=1)

    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales) - 1
    return torch.cat([xy1, xy2], dim=1)


def snap_to_anchors(boxes, size, stride, anchors, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_anchors = anchors.shape[0] if anchors is not None else 1
    width, height = int(size[0] / stride), int(size[1] / stride)

    if boxes.nelement() == 0:
        return (torch.zeros([num_anchors, num_classes, height, width], device=device),
                torch.zeros([num_anchors, 4, height, width], device=device),
                torch.zeros([num_anchors, 1, height, width], device=device))

    boxes, classes = boxes.split(4, dim=1)

    x_arange = torch.arange(0, size[0], stride, device=device, dtype=classes.dtype)
    y_arange = torch.arange(0, size[1], stride, device=device, dtype=classes.dtype)
    x, y = torch.meshgrid([x_arange, y_arange])
    xyxy = torch.stack((x, y, x, y), dim=2).unsqueeze(0)
    anchors = anchors.view(-1, 1, 1, 4).to(dtype=classes.dtype)
    anchors = (xyxy + anchors).reshape(-1, 4)

    # 1.compute IoU between boxes and anchors
    boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:] - 1], dim=1)  # xywh to xyxy -> (n, 4)
    xy1 = torch.max(anchors[:, None, :2], boxes[:, :2])  # -> (num_anchors, n, 2)
    xy2 = torch.min(anchors[:, None, 2:], boxes[:, 2:])  # -> (num_anchors, n, 2)
    inter = torch.prod((xy2 - xy1).clamp(0), dim=2)  # -> (num_anchors, n)
    boxes_area = torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, dim=1)  # -> (n,)
    anchors_area = torch.prod(anchors[:, 2:] - anchors[:, :2] + 1, dim=1)  # -> (num_anchors,)
    overlap = inter / (anchors_area[:, None] + boxes_area[None, :] - inter)  # -> (num_anchors, n)

    # 2.keep best box per anchor
    overlap, indices = overlap.max(
        dim=1)  # -> overlap(num_anchors,), indices(num_anchors,) max_overlap and max_overlap_indices
    box_target = box2delta(boxes[indices], anchors)  # -> (num_anchors, n) compute regression target
    box_target = box_target.view(num_anchors, 1, width, height, 4)  # -> reshape box_target to (anchors, 1, W, H, 4)
    box_target = box_target.permute(0, 4, 3, 2, 1)  # -> (anchors, 4, H, W, 1)
    box_target = box_target.squeeze().contiguous()  # -> (anchors, 4, H, W)

    depth = torch.ones_like(overlap) * -1  # -> (num_anchors,): -1
    depth[overlap < 0.4] = 0  # background
    depth[overlap >= 0.5] = classes[indices][overlap >= 0.5].squeeze() + 1  # objects, foreground
    depth = depth.view(num_anchors, width, height).transpose(1,
                                                             2).contiguous()  # depth: tag to distinguish foreground and background -> (anchors, H, W)

    # 3.generate target classes
    cls_target = torch.zeros((anchors.shape[0], num_classes + 1), device=device,
                             dtype=boxes.dtype)  # init cls_target -> (num_anchors, 81)
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()  # -> (num_anchors, 1)
    classes.view(-1, 1)
    classes[overlap < 0.4] = num_classes  # background
    cls_target.scatter_(1, classes, 1)  # classes to one-hot eg. 3 -> 0, 0, 0, 1, 0, 0, ...
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)  # -> (anchors, 1, W, H, 80)
    cls_target = cls_target.permute(0, 4, 3, 2, 1)  # -> (anchors, 80, H, W, 1)
    cls_target = cls_target.squeeze().contiguous()  # -> (anchors, 80, H, W)

    return (cls_target.view(num_anchors, num_classes, height, width),
            box_target.view(num_anchors, 4, height, width),
            depth.view(num_anchors, 1, height, width))


def box2delta(boxes, anchors):
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    delta_ctr = (boxes_ctr - anchors_ctr) / anchors_wh
    delta_wh = torch.log(boxes_wh / anchors_wh)

    return torch.cat([delta_ctr, delta_wh], dim=1)
