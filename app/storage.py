from android.storage import app_storage_path
import os, csv, requests, zipfile, shutil, android_scoped_storage
from kivy.storage.jsonstore import JsonStore
from kivymd.toast import toast
from io import BytesIO

app_path = app_storage_path()
model_dir_path = os.path.join(app_path, 'model')

store = JsonStore('data.json')

if not os.path.isdir(model_dir_path):
    os.mkdir(model_dir_path)
    os.rename(os.path.join(os.getcwd(), 'blood_sugar_model.tflite'), os.path.join(model_dir_path, 'blood_sugar_model.tflite'))
    os.rename(os.path.join(os.getcwd(), 'tendency_model.tflite'), os.path.join(model_dir_path, 'tendency_model.tflite'))

class Record:
    def __init__(self):
        self.record_path = os.path.join(app_path, 'record.csv')

        if not os.path.isfile(self.record_path):
            with open(self.record_path, 'w') as file:
                file.write('Glucosa en sang,Tendència de la glucosa en sang,Insulina administrada,Hidrats de carboni ingerits,Predicció de la glucosa en sang en 40 minuts,Predicció de la tendència de la glucosa en sang en 40 minuts,Hora,Data\n')
                file.close()

    def number_elements(self):
        with open(self.record_path, 'r') as file:
            return len(list(csv.reader(file, delimiter=','))) - 1

    def read(self, index):
        with open(self.record_path, 'r') as file:
            element = list(csv.reader(file, delimiter=','))[index]
            file.close
        return element

    def write(self, row):
        with open(self.record_path, 'a') as file:
            csv.writer(file, delimiter=',').writerow(row)
            file.close()

    def remove(self, item):
        item = list(map(str, item))
        with open(self.record_path, 'r+') as file:
            rows = list(csv.reader(file, delimiter=','))
            file.close()
        rows.remove(item)
        with open(self.record_path, 'w') as file:
            csv.writer(file, delimiter=',').writerows(rows)
            file.close()

    def remove_all(self):
        with open(self.record_path, 'w') as file:
            file.write('Glucosa en sang,Tendència de la glucosa en sang,Insulina administrada,Hidrats de carboni ingerits,Predicció de la glucosa en sang en 40 minuts,Predicció de la tendència de la glucosa en sang en 40 minuts,Hora,Data\n')
            file.close()

    def download_record(self):
        return android_scoped_storage.SharedStorage().insert(self.record_path, 'Downloads')

class Dataset:
    def __init__(self):
        self.dataset_template_path = os.path.join(os.getcwd(), 'dataset_template.csv')
        self.dataset_path = os.path.join(app_path, 'dataset.csv')

        if store.exists('dataset'):
            self.dataset_name = store.get('dataset')['filename']
            self.number_elements = store.get('dataset')['number_elements']
            self.dataset_path = os.path.join(app_path, self.dataset_name)
        else:
            self.dataset_name = None
            self.number_elements = None
            self.dataset_path = None

    def selected(self):
        return store.exists('dataset')

    def save_path(self, path):
        shutil.copy(path, app_path)
        self.dataset_name = os.path.basename(path)
        self.dataset_path = os.path.join(app_path, self.dataset_name)
        with open(path, 'r') as file:
            self.number_elements = len(list(csv.reader(file, delimiter=','))) - 1
            file.close()
        store.put('dataset', filename=self.dataset_name, number_elements=self.number_elements)

    def remove(self):
        if store.exists('dataset'):
            self.dataset_name = None
            self.number_elements = None
            os.remove(self.dataset_path)
            store.delete('dataset')

    def get_dataset_file(self):
        return open(self.dataset_path, 'r')

    def download_dataset_template(self):
        return android_scoped_storage.SharedStorage().insert(self.dataset_template_path, 'Downloads')

class WebService:
    def __init__(self):
        self.api_url = 'http://smartsugar.sytes.net:4631/'
        
    def api_connection(self, dataset):
        try:
            response = BytesIO(requests.post(self.api_url, files={'dataset': dataset}).content)
            zip_file = zipfile.ZipFile(response, 'r', compression=zipfile.ZIP_STORED)
            os.remove(os.path.join(model_dir_path, 'blood_sugar_model.tflite'))
            os.remove(os.path.join(model_dir_path, 'tendency_model.tflite'))
            zip_file.extractall(model_dir_path)

        except Exception as e:
            print(e)
            toast(repr(e))
