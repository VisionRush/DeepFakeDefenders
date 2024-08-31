from toolkit.chelper import final_model
from thop import profile
import torch


model = final_model().cuda()
input = torch.randn(1, 1, 6, 512, 512).cuda()
flops, params = profile(model, inputs=(input, ))
total_megabytes = params / (1024 * 1024)
print("Total parameters in MB:", total_megabytes)