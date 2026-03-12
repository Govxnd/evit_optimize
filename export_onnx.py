import torch
from ref import EViT

model = EViT(num_classes=10)

model.load_state_dict(torch.load("pruned_model.pth"))

model.eval()

dummy_input = torch.randn(1,3,512,512)

torch.onnx.export(
    model,
    dummy_input,
    "evit_model.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print("ONNX model exported")