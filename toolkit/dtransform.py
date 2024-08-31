from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.transforms.functional as F


# 添加jpeg压缩
class JPEGCompression:
    def __init__(self, quality=10, p=0.3):
        self.quality = quality
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            img_np = np.array(img)
            _, buffer = cv2.imencode('.jpg', img_np[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            jpeg_img = cv2.imdecode(buffer, 1)
            return Image.fromarray(jpeg_img[:, :, ::-1])
        return img


# 原始数据增强
def transforms_imagenet_train(
        img_size=(224, 224),
        scale=(0.08, 1.0),
        ratio=(3./4., 4./3.),
        hflip=0.5,
        vflip=0.5,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='random',
        mean=(0.485, 0.456, 0.406),
        jpeg_compression = 0,
):
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range

    primary_tfl = [
        RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)

        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size

        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    if jpeg_compression == 1:
        secondary_tfl += [JPEGCompression(quality=10, p=0.3)]

    final_tfl = [transforms.ToTensor()]

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


# 推理（测试）使用
def create_transforms_inference(h=256, w=256):
    transformer = transforms.Compose([
            transforms.Resize(size=(h, w)),
            transforms.ToTensor(),
        ])
    
    return transformer


def create_transforms_inference1(h=256, w=256):
    transformer = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle=90)),
        transforms.Resize(size=(h, w)),
        transforms.ToTensor(),
    ])

    return transformer


def create_transforms_inference2(h=256, w=256):
    transformer = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle=180)),
        transforms.Resize(size=(h, w)),
        transforms.ToTensor(),
    ])

    return transformer


def create_transforms_inference3(h=256, w=256):
    transformer = transforms.Compose([
        transforms.Lambda(lambda img: F.rotate(img, angle=270)),
        transforms.Resize(size=(h, w)),
        transforms.ToTensor(),
    ])

    return transformer


def create_transforms_inference4(h=256, w=256):
    transformer = transforms.Compose([
        transforms.Lambda(lambda img: F.hflip(img)),
        transforms.Resize(size=(h, w)),
        transforms.ToTensor(),
    ])

    return transformer


def create_transforms_inference5(h=256, w=256):
    transformer = transforms.Compose([
        transforms.Lambda(lambda img: F.vflip(img)),
        transforms.Resize(size=(h, w)),
        transforms.ToTensor(),
    ])

    return transformer
