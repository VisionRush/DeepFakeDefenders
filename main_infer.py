from PIL import Image
import numpy as np
import timm
import einops
import torch
from torch import nn
from toolkit.dtransform import create_transforms_inference, create_transforms_inference1,\
                    create_transforms_inference2,\
                    create_transforms_inference3,\
                    create_transforms_inference4,\
                    create_transforms_inference5
from toolkit.chelper import load_model
import torch.nn.functional as F


def extract_model_from_pth(params_path, net_model):
    checkpoint = torch.load(params_path)
    state_dict = checkpoint['state_dict']

    net_model.load_state_dict(state_dict, strict=True)

    return net_model


class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        self.kernel = torch.from_numpy(self._build_kernel(inc)).float()

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        return filters


class INFER_API:

    _instance = None
        
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(INFER_API, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self.transformer_ = [create_transforms_inference(h=512, w=512),
                        create_transforms_inference1(h=512, w=512),
                        create_transforms_inference2(h=512, w=512),
                        create_transforms_inference3(h=512, w=512),
                        create_transforms_inference4(h=512, w=512),
                        create_transforms_inference5(h=512, w=512)]
        self.srm = SRMConv2d_simple()

        # model init
        self.model = load_model('all', 2)
        model_path = './final_model_csv/final_model.pth'
        self.model = extract_model_from_pth(model_path, self.model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        self.model.eval()

    def _add_new_channels_worker(self, image):
        new_channels = []

        image = einops.rearrange(image, "h w c -> c h w")
        image = (image - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN).view(-1, 1, 1)) / torch.as_tensor(
            timm.data.constants.IMAGENET_DEFAULT_STD).view(-1, 1, 1)
        srm = self.srm(image.unsqueeze(0)).squeeze(0)
        new_channels.append(einops.rearrange(srm, "c h w -> h w c").numpy())

        new_channels = np.concatenate(new_channels, axis=2)
        return torch.from_numpy(new_channels).float()

    def add_new_channels(self, images):
        images_copied = einops.rearrange(images, "c h w -> h w c")
        new_channels = self._add_new_channels_worker(images_copied)
        images_copied = torch.concatenate([images_copied, new_channels], dim=-1)
        images_copied = einops.rearrange(images_copied, "h w c -> c h w")

        return images_copied

    def test(self, img_path):
        # img load
        img_data = Image.open(img_path).convert('RGB')

        # transform
        all_data = []
        for transform in self.transformer_:
            current_data = transform(img_data)
            current_data = self.add_new_channels(current_data)
            all_data.append(current_data)
        img_tensor = torch.stack(all_data, dim=0).unsqueeze(0).cuda()

        preds = self.model(img_tensor)

        return round(float(preds), 20)


def main():
    img = '51aa9b8d0da890cd1d0c5029e3d89e3c.jpg'
    infer_api = INFER_API()
    print(infer_api.test(img))


if __name__ == '__main__':
    main()