import torch
import torchvision
from models import networks
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2
import torchvision.transforms as transforms

import torch.nn as nn
from PIL import Image
import util.util

def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

path = './latest_net_G.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(path, map_location=device)

#net = networks.define_G(3, 3, 64, 'resnet_9blocks', norm='instance')
net = networks.ResnetGenerator(3,3, norm_layer=nn.InstanceNorm2d,n_blocks=9)


if hasattr(state_dict, '_metadata'):
    del state_dict._metadata
for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    __patch_instance_norm_state_dict(state_dict, net, key.split('.'))

net.load_state_dict(state_dict)
net.eval()

img = cv2.imread('/Users/zhangyu/python/cyclegan/datasets/bitmoji/testOne/epoch020_real_A.png')
transform1 = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0] and convert [H,W,C] to [C,H,W]
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# 这个代码没用！！！！！！！！！！！！！！！！！！！！！！！！！！！
#traced_model = torch.jit.trace(net, example_input)
#traced_model.save("model2.pt")


#optimized_traced_model = optimize_for_mobile(traced_model)
#optimized_traced_model._save_for_lite_interpreter("model.pt")
