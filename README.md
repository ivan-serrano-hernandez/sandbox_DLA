Convertir onxx a trt usando GPU
	/usr/src/tensorrt/bin/trtexec --device=0 --verbose --saveEngine= --onnx=
Convertir onnx a trt usando DLA
	/usr/src/tensorrt/bin/trtexec --verbose --saveEngine= --onnx= --useDLACore=0 --fp16 --allowGPUFallback
Comprobar un modelo con TensorRT
	/usr/src/tensorrt/bin/trtexec [--device=0 | --useDLACore=0] --loadEngine=
---
Restricciones de DLA
***

    The maximum supported batch size is 4096.
    The maximum supported size for non-batch dimensions is 8192.
    DLA does not support dynamic dimensions. Thus, for wildcard dimensions, the profile's min, max, and opt values must be equal.
    The runtime dimensions must be the same as the dimensions used for building.
    TensorRT may split a network into multiple DLA loadables if any intermediate layers cannot run on DLA and GPUFallback is enabled. Otherwise, TensorRT can emit an error and fallback. For more information, refer to GPU Fallback Mode.
    Due to hardware and software memory limitations, only 16 DLA loadable can be loaded concurrently per core.
    Each layer must have the same batch size within a single DLA loadable. Layers with different batch sizes will be partitioned into separate DLA graphs.

***
