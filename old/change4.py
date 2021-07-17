
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
import torch
import torchvision

# spec = coremltools.utils.load_spec("MobileNetV2.mlmodel")
#
#
# output = spec.description.output[0]
#
# import coremltools.proto.FeatureTypes_pb2 as ft
# output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
# output.type.imageType.height = 256
# output.type.imageType.width = 256
#
# coremltools.utils.save_spec(spec, "newFace.mlmodel")

model = torchvision.models.mobilenet_v2()
model.eval()
example_input = torch.rand(1, 3, 256, 256)
traced_model = torch.jit.trace(model, example_input)

input = ct.TensorType(name='input_name', shape=(1, 3, 256, 256))
mlmodel = ct.convert(traced_model, inputs=[input])
results = mlmodel.predict({"input": example_input.numpy()})
print(results['1651']) # 1651 is the node name given by PyTorch's JIT
