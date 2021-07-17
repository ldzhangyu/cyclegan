
import torch
import torchvision
import coremltools as ct
from models import networks
import io
# Convert to Core ML using the Unified Conversion API


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

# Load a pre-trained version of MobileNetV2
# path = './models/photo2cartoon_weights.pt'
path = './latest_net_G.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(path, map_location=device)

net = networks.define_G(3, 3, 64, 'resnet_9blocks', norm='instance')

if hasattr(state_dict, '_metadata'):
    del state_dict._metadata

# patch InstanceNorm checkpoints prior to 0.4
for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
#.

# def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
# net = networks(ngf=32, img_size=256, light=True).to(device)
net.load_state_dict(state_dict)
net.eval()


example_input = torch.rand(1, 3, 256, 256) # after test, will get 'size mismatch' error message with size 256x256
# traced_model = torch.jit.load(f, map_location="cpu")
traced_model = torch.jit.trace(net, example_input)


# with open(path, 'rb') as f:
#     buffer = io.BytesIO(f.read())
#     buffer.seek(0)
#     loaded = torch.jit.load(buffer,map_location='cpu')
print(type(traced_model))



model = ct.convert(
    traced_model, source='pytorch',
    inputs = [ct.ImageType(name="input_1", shape=example_input.shape)],  # name "input_1" is used in 'quickstart'
)
model.save("MobileNetV2.mlmodel")
