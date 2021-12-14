import tensorflow as tf
import numpy as np

class Diabetes:
    def __init__(self):
        self.blood_sugar_model = self.build_blood_sugar_model()
        self.tendency_model = self.build_tendency_model()
    
    def build_blood_sugar_model(self):
        class BloodSugar:
            def __init__(self):
                self.interpreter = tf.lite.Interpreter(model_path='app/blood_sugar_model.tflite')
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.interpreter.allocate_tensors()

            def predict(self, values):
                self.interpreter.set_tensor(self.input_details[0]['index'], Diabetes.normalize_input(self, values))
                self.interpreter.invoke()
                result = Diabetes.normalize_output(self, self.interpreter.get_tensor(self.output_details[0]['index']))
                return result

        return BloodSugar()

    def build_tendency_model(self):
        class Tendency:
            def __init__(self):
                self.interpreter = tf.lite.Interpreter(model_path='app/tendency_model.tflite')
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.interpreter.allocate_tensors()

            def predict(self, values):
                self.interpreter.set_tensor(self.input_details[0]['index'], Diabetes.normalize_input(self, values))
                self.interpreter.invoke()
                result = self.interpreter.get_tensor(self.output_details[0]['index'])
                return result

        return Tendency()

    def normalize_input(self, input):
        input[0] = input[0] / 400
        input[1] = input[1]
        input[2] = input[2] / 20
        input[3] = input[3] / 10
        input = np.expand_dims(np.array(input, dtype=np.float32), axis=0)
        return input

    def normalize_output(self, output):
        output *= 500
        return output
    
    def predict(self, values):
        predicted_blood_sugar = round(self.blood_sugar_model.predict(values).item())
        predicted_tendency = self.tendency_model.predict(values).tolist()[0]

        return [predicted_blood_sugar, predicted_tendency]





    

    