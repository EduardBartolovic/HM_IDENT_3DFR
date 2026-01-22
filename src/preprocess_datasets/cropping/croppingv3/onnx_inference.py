from enum import Enum
from abc import ABC, abstractmethod
import onnxruntime as ort

class InferenceType(Enum):
    WINDOWS = 1
    Android = 2


class ONNX_Inference(ABC):
    
    def __init__(self, model_path):
        pass

    @abstractmethod
    def perform_inference(self, input):
        pass
        

class ONNX_Inference_Windows(ONNX_Inference):
    
    def __init__(self, model_path, device="cpu", output_name_is_none=True):
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        if output_name_is_none: 
            self.output_name = None
        else:
            self.output_name = self.session.get_outputs()[0].name

    def perform_inference(self, input):
        if self.output_name is None:
            output = self.session.run(None, {self.input_name: input})
        else: 
            output = self.session.run([self.output_name], {self.input_name: input})
        return output


class ONNX_Inference_Android(ONNX_Inference):

    def __init__(self, model_path, device="cpu"):
        pass

    def perform_inference(self, input):
        pass