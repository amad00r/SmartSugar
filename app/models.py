from jnius import autoclass
import os
import numpy as np
import storage

File = autoclass('java.io.File')
Interpreter = autoclass('org.tensorflow.lite.Interpreter')
InterpreterOptions = autoclass('org.tensorflow.lite.Interpreter$Options')
Tensor = autoclass('org.tensorflow.lite.Tensor')
DataType = autoclass('org.tensorflow.lite.DataType')
TensorBuffer = autoclass('org.tensorflow.lite.support.tensorbuffer.TensorBuffer')
ByteBuffer = autoclass('java.nio.ByteBuffer')

class Diabetes:
    def __init__(self):
        self.load()
    
    def load(self):
        self.blood_sugar_model = self.build_blood_sugar_model()
        self.tendency_model = self.build_tendency_model()

    def build_blood_sugar_model(self):
        class BloodSugar:
            def __init__(self):
                self.load()

            def load(self, num_threads=None):
                model = File(os.path.join(storage.model_dir_path, 'blood_sugar_model.tflite'))
                options = InterpreterOptions()
                if num_threads is not None:
                    options.setNumThreads(num_threads)
                self.interpreter = Interpreter(model, options)
                self.allocate_tensors()

            def allocate_tensors(self):
                self.interpreter.allocateTensors()
                self.input_shape = self.interpreter.getInputTensor(0).shape()
                self.output_shape = self.interpreter.getOutputTensor(0).shape()
                self.output_type = self.interpreter.getOutputTensor(0).dataType()

            def predict(self, values):
                input = ByteBuffer.wrap(values.tobytes())
                output = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
                self.interpreter.run(input, output.getBuffer().rewind())
                return np.reshape(np.array(output.getFloatArray()), self.output_shape)

        return BloodSugar()

    def build_tendency_model(self):
        class Tendency:
            def __init__(self):
                self.load()

            def load(self, num_threads=None):
                model = File(os.path.join(storage.model_dir_path, 'tendency_model.tflite'))
                options = InterpreterOptions()
                if num_threads is not None:
                    options.setNumThreads(num_threads)
                self.interpreter = Interpreter(model, options)
                self.allocate_tensors()

            def allocate_tensors(self):
                self.interpreter.allocateTensors()
                self.input_shape = self.interpreter.getInputTensor(0).shape()
                self.output_shape = self.interpreter.getOutputTensor(0).shape()
                self.output_type = self.interpreter.getOutputTensor(0).dataType()

            def predict(self, values):
                input = ByteBuffer.wrap(values.tobytes())
                output = TensorBuffer.createFixedSize(self.output_shape, self.output_type)
                self.interpreter.run(input, output.getBuffer().rewind())
                return np.reshape(np.array(output.getFloatArray()), self.output_shape)

        return Tendency()

    def normalize_input(self, input):
        input[0] = input[0] / 400 
        input[2] = input[2] / 20
        input[3] = input[3] / 10
        input = np.expand_dims(np.array(input, dtype=np.float32), axis=0)
        return input

    def normalize_output(self, output):
        output *= 500
        return round(output)
    
    def predict(self, values):
        normalized_input = self.normalize_input(values)

        predicted_blood_sugar = self.normalize_output(self.blood_sugar_model.predict(normalized_input[:]).item())

        tendency_output = self.tendency_model.predict(normalized_input[:]).tolist()[0]

        if tendency_output.index(max(tendency_output)) == 0:
            predicted_tendency = -1
        elif tendency_output.index(max(tendency_output)) == 1:
            predicted_tendency = -0.5
        elif tendency_output.index(max(tendency_output)) == 2:
            predicted_tendency = 0
        elif tendency_output.index(max(tendency_output)) == 3:
            predicted_tendency = 0.5
        elif tendency_output.index(max(tendency_output)) == 4:
            predicted_tendency = 1

        return [predicted_blood_sugar, predicted_tendency]   