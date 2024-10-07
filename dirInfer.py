import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import os
import sys

from utils.general import  non_max_suppression

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)




class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Inference():

    def __init__(self):
        self.batch = 1
        '''
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        '''
        self.batch = 1

        self.host_inputs  = []
        self.cuda_inputs = []

        self.host_outputs = []
        self.cuda_outputs = []

        self.bindings = []
        

    def PrepareEngine(self):

        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                      'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                      'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                      'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                      'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                      'hair drier', 'toothbrush']

        with open('/app/weights/yolov9-c.trt', 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        self.output_shape = [self.engine.get_tensor_shape(binding) for binding in self.engine][1]

        # create buffer
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding)) * self.batch
            host_mem = cuda.pagelocked_empty(shape=[size],dtype=np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if self.engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

        self.frame_counter = 0
        self.source_path="images_COCO"
        l = os.listdir(self.source_path)
        self.input = sorted(l, key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.numObjs = []
    
    def DoInference(self):
        for i in range(len(self.input)):
            img_path = self.input[i]
            bgr_img = cv2.imread(os.path.join(self.source_path, img_path))
        
            # bgr_img = cv2.flip(bgr_img, 1)

            # Format the frame
            h, w, _ = bgr_img.shape
            scale = min(640/w, 640/h)
            inp = np.zeros((640, 640, 3), dtype = np.float32)
            nh = int(scale * h)
            nw = int(scale * w)
            inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), (nw, nh))
            inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
            inp = np.expand_dims(inp.transpose(2, 0, 1), 0)

            # Compute the prediction
            np.copyto(self.host_inputs[0], inp.ravel())
            context = self.engine.create_execution_context()

            self.stream = cuda.Stream()

            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            context.execute_v2(self.bindings)
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)

            self.stream.synchronize()

            output = self.host_outputs[0]

            # convert to output shape
            preds = torch.Tensor(output.reshape(self.output_shape))

            # apply non max supression
            preds = non_max_suppression(preds,
                                        0.5,
                                        0.5,
                                        labels=[],
                                        multi_label=True,
                                        agnostic=False,
                                        max_det=300)

            # print number of objects
            self.numObjs.append(len(preds[0]))

        print("Number of objects detected among all frames:")
        print(self.numObjs)

if __name__ == "__main__":

    inferObj = Inference()
    inferObj.PrepareEngine()
    inferObj.DoInference()

