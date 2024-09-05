import torch
import torch.nn as nn
from model.convnext import convnext_base
import timm
from model.replknet import create_RepLKNet31B


class augment_inputs_network(nn.Module):
    def __init__(self, model):
        super(augment_inputs_network, self).__init__()
        self.model = model
        self.adapter = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.adapter(x)
        x = (x - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=x.get_device()).view(1, -1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD, device=x.get_device()).view(1, -1, 1, 1)

        return self.model(x)


class final_model(nn.Module):  # Total parameters: 158.64741325378418 MB
    def __init__(self):
        super(final_model, self).__init__()
        
        self.convnext = convnext_base(num_classes=2)
        self.convnext = augment_inputs_network(self.convnext)

        self.replknet = create_RepLKNet31B(num_classes=2)
        self.replknet = augment_inputs_network(self.replknet)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)

        pred1 = self.convnext(x)
        pred2 = self.replknet(x)

        outputs_score1 = nn.functional.softmax(pred1, dim=1)
        outputs_score2 = nn.functional.softmax(pred2, dim=1)

        predict_score1 = outputs_score1[:, 1]
        predict_score2 = outputs_score2[:, 1]

        predict_score1 = predict_score1.view(B, N).mean(dim=-1)
        predict_score2 = predict_score2.view(B, N).mean(dim=-1)

        return torch.stack((predict_score1, predict_score2), dim=-1).mean(dim=-1)


def load_model(model_name, ctg_num, use_sync_bn):
    """Load standard model, like vgg16, resnet18,

    Args:
        model_name: e.g., vgg16, inception, resnet18, ...
        ctg_num: e.g., 1000
        use_sync_bn: True/False
    """
    if model_name == 'convnext':
        model = convnext_base(num_classes=ctg_num)
        model_path = 'pre_model/convnext_base_1k_384.pth'
        check_point = torch.load(model_path, map_location='cpu')['model']
        check_point.pop('head.weight')
        check_point.pop('head.bias')
        model.load_state_dict(check_point, strict=False)

        model = augment_inputs_network(model)
        
    elif model_name == 'replknet':
        model = create_RepLKNet31B(num_classes=ctg_num, use_sync_bn=use_sync_bn)
        model_path = 'pre_model/RepLKNet-31B_ImageNet-1K_384.pth'
        check_point = torch.load(model_path)
        check_point.pop('head.weight')
        check_point.pop('head.bias')
        model.load_state_dict(check_point, strict=False)

        model = augment_inputs_network(model)

    elif model_name == 'all':
        model = final_model()

    print("model_name", model_name)

    return model

