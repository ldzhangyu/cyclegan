import torch
import torchvision
from models import networks
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2
import torchvision.transforms as transforms

import coremltools as ct
import torch.nn as nn
from PIL import Image
import util.util
from coremltools.models.neural_network.quantization_utils import *


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

path = './model_for_coreml.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scriptMode = torch.jit.load(path)



scriptMode.eval()


example_input = torch.rand(1, 3, 256, 256) # after test, will get 'size mismatch' error message with size 256x256


model = ct.convert(
    scriptMode, source='pytorch',
    inputs=[ct.ImageType(bias=[-1, -1, -1], scale=2.0/255, shape=example_input.shape)],
)


line_quant_model_8 = quantize_weights(model, 8, "linear")


line_quant_model_8.save('Cyclegan.mlmodel')

#model.save("Cyclegan.mlmodel")

# #imgPath = '/Users/zhangyu/python/cyclegan/datasets/bitmoji/testOne/epoch020_real_A.png'
# img = Image.open(imgPath).convert('RGB')
# transform1 = transforms.Compose([
#     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0] and convert [H,W,C] to [C,H,W]
#     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# ])
# img = transform1(img)  # 归一化到 [-1.0,1.0],并转成[C,H,W]
# tup1 = (img,)
# img = torch.stack(tup1, 0, out=None)
# img = scriptMode.forward(img)
# img = util.util.tensor2im(img)
# img = Image.fromarray(img)
# img.show()
#traced_model = torch.jit.trace(net, example_input)
#traced_model.save("model2.pt")


#optimized_traced_model = optimize_for_mobile(traced_model)
#optimized_traced_model._save_for_lite_interpreter("model.pt")
# input_name = scriptMode.inputs[0].name.split(':')[0]
# print(input_name) #Check input_name.
# keras_output_node_name = scriptMode.outputs[0].name.split(':')[0]
# graph_output_node_name = keras_output_node_name.split('/')[-1]
# print(graph_output_node_name) #Check input_name.
#
#
# mlmodel = ct.convert(scriptMode,
#                      input_name_shape_dict={input_name: (1, 256, 256, 3)},
#                      output_feature_names=[graph_output_node_name],
#                      minimum_ios_deployment_target='13',
#                      image_input_names=input_name,
#                      image_scale=2/ 255.0,
#                      red_bias=-1,
#                      green_bias=-1,
#                      blue_bias=-1
#                      )
# mlmodel.save('./cyclegan.mlmodel')
