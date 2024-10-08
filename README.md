
# TensorRT: ONNX to TensorRT Conversion Guide

This guide provides step-by-step instructions on converting ONNX models to TensorRT using both **GPU** and **NVIDIA Deep Learning Accelerator (DLA)**, as well as verifying the TensorRT engine. It also includes important restrictions when using DLA.

---

## Conversion Instructions

### 1. Convert ONNX to TensorRT Using GPU

To convert an ONNX model to TensorRT using the **GPU**:

```bash
/usr/src/tensorrt/bin/trtexec --device=0 --verbose --saveEngine=<output_engine.trt> --onnx=<model.onnx>
```

- `--device=0`: Specifies the GPU device to use.
- `--saveEngine`: The path to save the converted TensorRT engine.
- `--onnx`: The path to your ONNX model.

---

### 2. Convert ONNX to TensorRT Using DLA

To convert an ONNX model to TensorRT using **NVIDIA DLA**:

```bash
/usr/src/tensorrt/bin/trtexec --verbose --saveEngine=<output_engine.trt> --onnx=<model.onnx> --useDLACore=0 --fp16 --allowGPUFallback
```

- `--useDLACore=0`: Specifies DLA core 0 for inference.
- `--fp16`: Enables FP16 precision, which DLA supports.
- `--allowGPUFallback`: Fallback to GPU if DLA encounters unsupported layers.

---

### 3. Verify a TensorRT Engine

To load and test an existing TensorRT engine:

```bash
/usr/src/tensorrt/bin/trtexec [--device=0 | --useDLACore=0] --loadEngine=<output_engine.trt>
```

- `--loadEngine`: The path to your pre-built TensorRT engine file.

---

## DLA Constraints

**Important DLA Limitations**:
- **Maximum Batch Size**: 4096.
- **Maximum Non-Batch Dimension Size**: 8192.
- **No Dynamic Dimensions**: DLA does not support dynamic dimensions. The `min`, `max`, and `opt` values in profiles must be identical.
- **Static Dimensions at Runtime**: The runtime dimensions must match the dimensions used during engine building.
- **Layer Partitioning**: TensorRT may split the network into multiple DLA loadable sections if some layers cannot be processed by DLA. If `GPUFallback` is enabled, these layers will fall back to GPU execution; otherwise, TensorRT may emit an error.
- **Memory Limits**: Only 16 DLA loadable layers can be loaded concurrently per DLA core due to hardware and software memory constraints.
- **Consistent Batch Size**: All layers in a DLA loadable must have the same batch size. Layers with different batch sizes will be partitioned into separate DLA graphs.

---

For more details on **GPU Fallback Mode**, refer to the official TensorRT documentation.

---

This guide will help you efficiently convert and verify models using TensorRT, leveraging both GPU and DLA for optimized inference!
