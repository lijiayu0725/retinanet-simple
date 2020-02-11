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


def delta2box(deltas, anchors, size, stride):
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1
    clamp = lambda t: torch.max(m, torch.min(t, M))

    xy1 = clamp(pred_ctr - 0.5 * pred_wh)
    xy2 = clamp(pred_ctr + 0.5 * pred_wh - 1)

    return torch.cat([xy1, xy2], dim=1)


def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None):
    device = all_cls_head.device
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.shape[0] if anchors is not None else 1  # 9
    num_classes = all_cls_head.shape[1] // num_anchors  # 80
    height, width = all_cls_head.shape[-2:]  # H, W

    batch_size = all_cls_head.shape[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, 4), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].reshape(-1)
        box_head = all_box_head[batch, :, :, :].reshape(-1, 4)

        # keep score over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)  # cls score >= thresh indices
        if keep.nelement() == 0:
            continue

        # gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.shape[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices / width / height) % num_classes
        classes = classes.type(all_cls_head.type())

        x = indices % width
        y = (indices / width) % height
        a = indices / num_classes / height / width
        box_head = box_head.view(num_anchors, 4, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = torch.stack([x, y, x, y], dim=1).type(all_cls_head.type()) * stride + anchors[a, :]
            boxes = delta2box(boxes, grid, [width, height], stride)

        out_scores[batch, :scores.shape[0]] = scores
        out_boxes[batch, :boxes.shape[0], :] = boxes
        out_classes[batch, :classes.shape[0]] = classes

    return out_scores, out_boxes, out_classes


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 4), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    for batch in range(batch_size):
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():  # ???
                i -= 1
                break
            # compute iou over boxes
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), dim=1)
            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1  # criterion -> no useful any more

            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes
