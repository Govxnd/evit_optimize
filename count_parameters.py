import torch
from ref import EViT

# load model
model = EViT(num_classes=10)

# load pruned weights
model.load_state_dict(torch.load("pruned_model.pth"))

# count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total Parameters:", total_params)
print("Trainable Parameters:", trainable_params)
print("Model Size (MB):", total_params * 4 / (1024**2))