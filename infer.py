import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch

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

        with open('yolov9-s-fp32-nms.trt', 'rb') as f:
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

    
    
    def DoInference(self):
        image = cv2.imread("./data/images/horses.jpg")
        image = cv2.resize(image, (320,320))
        image = (2.0 / 255.0) * image.transpose((2, 0, 1)) - 1.0

        np.copyto(self.host_inputs[0], image.ravel())
        context = self.engine.create_execution_context()

        self.stream = cuda.Stream()

        start_time = time.time()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        context.execute_v2(self.bindings)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)

        self.stream.synchronize()
        print("execute times "+str(time.time()-start_time))

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
        
        # extract the classes
        class_indices = [int(row[-1]) for row in preds[0]]
        # class_names_list = [class_names.get(idx, 'Unknown') for idx in class_indices]

        for cls in class_indices:
            print(self.names[cls])
        print("-------------------")

        print(preds)
    
        print("-------------------")

        confs = [float(row[4]) for row in preds[0]]
        for conf in confs:
            print(conf) 


        print("-------------------")

        topLeft = [(float(row[0]),float(row[1])) for row in preds[0]]
        bottomRight = [(float(row[2]),float(row[3])) for row in preds[0]]

        print(topLeft)
        print(bottomRight)


    
    """def __del__(self):
        self.host_inputs  = []
        self.cuda_inputs = []

        self.host_outputs = []
        self.cuda_inputs = []

        self.bindings = []
        self.stream = None"""

if __name__ == "__main__":

    inferObj = Inference()
    inferObj.PrepareEngine()
    inferObj.DoInference()
