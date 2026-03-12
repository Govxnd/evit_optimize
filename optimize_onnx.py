import onnx
import onnxoptimizer

model = onnx.load("evit_simplified.onnx")

passes = [
    "eliminate_identity",
    "eliminate_nop_transpose",
    "fuse_consecutive_transposes",
    "fuse_bn_into_conv",
]

optimized_model = onnxoptimizer.optimize(model, passes)

onnx.save(optimized_model, "evit_optimized.onnx")

print("ONNX optimization complete")