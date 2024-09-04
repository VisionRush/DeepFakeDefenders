from toolkit.chelper import final_model
import torch

# Trained ConvNeXt and RepLKNet paths (for reference)
convnext_path = './final_model_csv/convnext_end.pth'
replknet_path = './final_model_csv/replk_end.pth'

model = final_model()
model.convnext.load_state_dict(torch.load(convnext_path, map_location='cpu')['state_dict'], strict=True)
model.replknet.load_state_dict(torch.load(replknet_path, map_location='cpu')['state_dict'], strict=True)

torch.save({'state_dict': model.state_dict()}, './final_model_csv/final_model.pth')
