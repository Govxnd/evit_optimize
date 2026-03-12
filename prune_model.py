import torch
import torch.nn.utils.prune as prune
from ref import EViT

model = EViT(num_classes=10)

# Apply pruning
for name, module in model.named_modules():

    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.3)

    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.3)

# 🔴 IMPORTANT: make pruning permanent
for module in model.modules():

    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        try:
            prune.remove(module, 'weight')
        except:
            pass

print("Pruning applied and finalized")

torch.save(model.state_dict(), "pruned_model.pth")