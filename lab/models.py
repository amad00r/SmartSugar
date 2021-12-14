import numpy as np
import pandas as pd

import csv, os

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self):
        self.name = 'C:/Users/amade/Desktop/TR/lab/dataset/dataset.csv'

        self.X_train_blood_sugar = None
        self.X_test_blood_sugar = None

        self.y_train_blood_sugar = None
        self.y_test_blood_sugar = None


        self.X_train_tendency = None
        self.X_test_tendency = None

        self.y_train_tendency = None
        self.y_test_tendency = None


    def print_all(self):
        print(self.X_train_blood_sugar)
        print(self.X_test_blood_sugar)
        print(self.y_train_blood_sugar)
        print(self.y_test_blood_sugar)
        print(self.X_train_tendency)
        print(self.X_test_tendency)
        print(self.y_train_tendency)
        print(self.y_test_tendency)


    def normalize_data(self, data):
        i = 0
        for element in data:
            if i == 0:
                data[i] = float(element) / 500
            elif i == 1:
                data[i] = element
            elif i == 2:
                data[i] = float(element) / 20
            elif i == 3:
                data[i] = float(element) / 10
            elif i == 4:
                data[i] = float(element) / 500
            elif i == 5:
                data[i] = [0] * 5
                if element == '-1':
                    data[i][0] = 1
                elif element == '-0.5':
                    data[i][1] = 1
                elif element == '0':
                    data[i][2] = 1
                elif element == '0.5':
                    data[i][3] = 1
                elif element == '1':
                    data[i][4] = 1
            i += 1
        
        return data

    def import_dataset(self):
        with open(self.name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            element_counter = 0
            
            X_dataset = None
            y_blood_sugar_dataset = None
            y_tendency_dataset = None
            
            for row in csv_reader:
                if element_counter != 0:
                    del row[0]
                    values = self.normalize_data(row)
                    
                    if element_counter == 1:
                        y_tendency_dataset = np.array(values[-1], dtype=np.single)
                        del values[-1]
                        y_blood_sugar_dataset = np.array(values[-1], dtype=np.single)
                        del values[-1]
                        X_dataset = np.array(values, dtype=np.single)
                    else:
                        y_tendency_dataset = np.vstack((y_tendency_dataset, values[-1])).astype(np.single)
                        del values[-1]
                        y_blood_sugar_dataset = np.vstack((y_blood_sugar_dataset, values[-1])).astype(np.single)
                        del values[-1]
                        X_dataset = np.vstack((X_dataset, values)).astype(np.single)
                element_counter += 1
            print(f'''\nNombre d'elements al dataset: {element_counter - 1}''')

        self.X_train_blood_sugar, self.X_test_blood_sugar, self.y_train_blood_sugar, self.y_test_blood_sugar = train_test_split(X_dataset, y_blood_sugar_dataset, test_size=0.2)
        self.X_train_tendency, self.X_test_tendency, self.y_train_tendency, self.y_test_tendency = train_test_split(X_dataset, y_tendency_dataset, test_size=0.2)


class BloodSugarModel:
    def __init__(self):
        self.name_blood_sugar = 'C:/Users/amade/Desktop/TR/lab/model/blood_sugar_model.h5'
        self.name_tendency = 'C:/Users/amade/Desktop/TR/lab/model/tendency_model.h5'

        self.optimal_sugar_range = (70, 130)

        self.blood_sugar_model = self.build_blood_sugar_model()
        self.tendency_model = self.build_tendency_model()

        """ if os.path.isfile(self.name_blood_sugar):
            self.load_blood_sugar()
        if os.path.isfile(self.name_tendency):
            self.load_tendency() """

    def normalize_input(self, input):
        i = 0
        for element in input:
            if i == 0:
                input[i] = float(element) / 400
            elif i == 1:
                input[i] = float(element)
            elif i == 2:
                input[i] = float(element) / 20
            elif i == 3:
                input[i] = float(element) / 10
            i += 1

        return input
    
    def build_blood_sugar_model(self):
        model = Sequential()
        model.add(Dense(4, input_shape=(4,)))
        model.add(LeakyReLU())
        model.add(Dropout(0.1))
        model.add(Dense(2))
        model.add(LeakyReLU())
        model.add(Dense(1))
        model.add(LeakyReLU())
        model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def predict_blood_sugar(self, input):
        data = self.normalize_input(input[:])
        predicted_sugar = round(self.blood_sugar_model.predict([data]).item() * 500)
        return predicted_sugar

    def train_blood_sugar(self, x_train, y_train, x_val, y_val):
        self.blood_sugar_model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=1, validation_data=(x_val, y_val))

    def save_blood_sugar(self):
        self.blood_sugar_model.save(self.name_blood_sugar)
        lite_model = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(self.name_blood_sugar)).convert()
        with open('C:/Users/amade/Desktop/TR/lab/blood_sugar_model.tflite', 'wb') as f:
            f.write(lite_model)

    def build_tendency_model(self):
        model = Sequential()
        model.add(Dense(4, input_shape=(4,)))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))
        model.add(Dense(4))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        model.add(Dense(5, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=0.08), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict_tendency(self, input):
        data = self.normalize_input(input[:])
        predicted_tendency = self.tendency_model.predict([data])
        return predicted_tendency

    def train_tendency(self, x_train, y_train, x_val, y_val):
        self.tendency_model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=1, validation_data=(x_val, y_val))
    
    def save_tendency(self):
        self.tendency_model.save(self.name_tendency)
        lite_model = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(self.name_tendency)).convert()
        with open('C:/Users/amade/Desktop/TR/lab/tendency_model.tflite', 'wb') as f:
            f.write(lite_model)