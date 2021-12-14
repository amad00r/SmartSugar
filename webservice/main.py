from flask import Flask, request, send_file
import zipfile, os, csv
import tensorflow as tf
import numpy as np
from io import TextIOWrapper, BytesIO
from time import sleep

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from sklearn.model_selection import train_test_split

app_dir = os.getcwd()
data_dir = os.path.join(app_dir, 'data')

class Training:
   def __init__(self):
      self.restart()

   def restart(self):
      self.blood_sugar_model = self.build_blood_sugar_model()
      self.tendency_model = self.build_tendency_model()

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

   def train_blood_sugar(self):
      self.blood_sugar_model.fit(self.X_train_blood_sugar, self.y_train_blood_sugar, epochs=300, verbose=1, batch_size=1, validation_data=(self.X_test_blood_sugar, self.y_test_blood_sugar))

   def save_blood_sugar(self):
      lite_model = tf.lite.TFLiteConverter.from_keras_model(self.blood_sugar_model).convert()
      return lite_model

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

   def train_tendency(self):
      self.tendency_model.fit(self.X_train_tendency, self.y_train_tendency, epochs=300, verbose=1, batch_size=1, validation_data=(self.X_test_tendency, self.y_test_tendency))

   def save_tendency(self):
      lite_model = tf.lite.TFLiteConverter.from_keras_model(self.tendency_model).convert()
      return lite_model

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

   def import_dataset(self, data):
      csv_file = csv.reader(data, delimiter=',')

      X_dataset = None
      y_blood_sugar_dataset = None
      y_tendency_dataset = None
      
      for row in csv_file:
         if csv_file.line_num != 1:
            values = self.normalize_data(row)
            
            if X_dataset is None:
               y_tendency_dataset = np.array(values[5], dtype=np.single)
               y_blood_sugar_dataset = np.array(values[4], dtype=np.single)
               X_dataset = np.array([values[0], values[1], values[2], values[3]], dtype=np.single)
            else:
               y_tendency_dataset = np.vstack((y_tendency_dataset, np.array(values[5], dtype=np.single))).astype(np.single)
               y_blood_sugar_dataset = np.vstack((y_blood_sugar_dataset, np.array(values[4], dtype=np.single))).astype(np.single)
               X_dataset = np.vstack((X_dataset, np.array([values[0], values[1], values[2], values[3]], dtype=np.single))).astype(np.single)
               
      self.X_train_blood_sugar, self.X_test_blood_sugar, self.y_train_blood_sugar, self.y_test_blood_sugar = train_test_split(X_dataset, y_blood_sugar_dataset, test_size=0.2)
      self.X_train_tendency, self.X_test_tendency, self.y_train_tendency, self.y_test_tendency = train_test_split(X_dataset, y_tendency_dataset, test_size=0.2)

   def train(self):
      self.train_blood_sugar()
      self.train_tendency()
      blood_sugar_file = self.save_blood_sugar()
      tendency_file = self.save_tendency()
      return [blood_sugar_file, tendency_file]

training = Training()

app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_files():
   try:
      dataset_file = TextIOWrapper(request.files['dataset'], encoding='utf-8')
      training.import_dataset(dataset_file)
      files = training.train()

      zip_buffer = BytesIO()
      zipfolder = zipfile.ZipFile(zip_buffer, 'w', compression = zipfile.ZIP_STORED)
      zipfolder.writestr('blood_sugar_model.tflite', files[0])
      zipfolder.writestr('tendency_model.tflite', files[1])
      zipfolder.close()
      zip_buffer.seek(0)

      training.restart()

      return send_file(zip_buffer, mimetype='zip', download_name='model.zip', as_attachment=False)
   except Exception as e:
      print(e)
 
if __name__ == '__main__':
   app.run(debug = True, host = '0.0.0.0', port = '4631')