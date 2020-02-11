import random

import cv2
import torch
import torch.nn.functional as F


def random_horizontal_flip(image, boxes):
    if random.randint(0, 1):
        image = cv2.flip(image, 1)
        boxes[:, 0] = image.shape[0] - boxes[:, 0] - boxes[:, 2]
    return image, boxes


def totensor(image, boxes=None):
    data = torch.from_numpy(image.transpose(2, 0, 1))
    data = data.float() / 255
    if boxes is not None:
        return data, boxes
    else:
        return data

def normalize(data, boxes=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.FloatTensor(mean)
    std = torch.FloatTensor(std)
    data = (data - mean[:, None, None]) / std[:, None, None]
    if boxes is not None:
        return data, boxes
    else:
        return data

def pad(data, boxes=None, stride=128):
    ph, pw = (stride - d % stride for d in data.shape[1:])
    data = F.pad(data, [0, pw, 0, ph])
    if boxes is not None:
        return data, boxes
    else:
        return data


if __name__ == '__main__':
    pass