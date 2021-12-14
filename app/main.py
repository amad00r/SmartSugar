from android.permissions import request_permissions, check_permission, Permission
from android.storage import primary_external_storage_path
from android import api_version

request_permissions([Permission.INTERNET, Permission.READ_EXTERNAL_STORAGE])

from kivy.config import Config
from kivy.core.window import Window
from kivy.clock import Clock

Window.keyboard_anim_args = {'d': .2, 't': 'in_out_expo'}
Window.softinput_mode = 'below_target'

from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.toast import toast
from kivy.metrics import dp
from kivy.uix.anchorlayout import AnchorLayout

from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.list import TwoLineAvatarListItem, IconLeftWidget, OneLineIconListItem
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.dropdownitem import MDDropDownItem
from kivy.properties import ObjectProperty, StringProperty
from kivy.utils import get_color_from_hex
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.bottomsheet import MDListBottomSheet

import numpy as np
import os

""" import matplotlib.pyplot as plt
import PIL as pil
from io import BytesIO """

from datetime import datetime

import models, storage, picker, android_scoped_storage
model = models.Diabetes()
record = storage.Record()
dataset = storage.Dataset()
web_service = storage.WebService()

class Manager(ScreenManager):
    def __init__(self, **kwargs):
        super(Manager, self).__init__(**kwargs)


class ListElement(TwoLineAvatarListItem):
    def __init__(self, data, **kwargs):
        super(ListElement, self).__init__(**kwargs)
        self.font_style='H5'
        self.data = data

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.selected_item = None
        if api_version < 29:
            self.file_manager = MDFileManager(
                exit_manager=self.exit_manager,
                select_path=self.select_path,
                preview=False,
            )
        self.called = False

    def callback(self, instance):
        if instance.icon == 'file-plus-outline':
            self.parent.current = 'input_screen'
        else:
            self.parent.current = 'recommend_screen'

        self.ids.speed_dial.close_stack()

    def load_dataset_label(self):
        if dataset.dataset_path == None:
            self.ids.dataset_item.secondary_text = 'Cap arxiu seleccionat'
            self.ids.dataset_item.tertiary_text = 'Cap arxiu seleccionat'
        else:
            self.ids.dataset_item.secondary_text = dataset.dataset_name
            self.ids.dataset_item.tertiary_text = f"Nombre d'elements: {str(dataset.number_elements)}"

    def load_record_label(self):
        if record.number_elements() == 0:
            self.ids.record_item.secondary_text = 'Encara no hi ha cap element al registre'
        else:
            self.ids.record_item.secondary_text = f"Nombre d'elements: {str(record.number_elements())}"

    def open_dataset_actions(self):
        actions_menu = MDListBottomSheet()
        actions_menu.add_item(
            text='Importar dataset',
            callback=lambda x: self.open_filechooser(),
            icon='database-import-outline',
        )
        actions_menu.add_item(
            text='Descarregar plantilla del dataset',
            callback=lambda x: self.download_template(),
            icon='download',
        )
        actions_menu.open()

    def open_model_actions(self):
        actions_menu = MDListBottomSheet()
        actions_menu.add_item(
            text='Entrenar model',
            callback=lambda x: self.train_model(),
            icon='dumbbell',
        )
        actions_menu.open()

    def open_record_actions(self):
        actions_menu = MDListBottomSheet()
        actions_menu.add_item(
            text='Esborrar registre',
            callback=lambda x: self.remove_record(),
            icon='table-large-remove',
        )
        actions_menu.add_item(
            text='Descarregar registre',
            callback=lambda x: self.download_record(),
            icon='download',
        )
        actions_menu.open()

    def download_template(self):
        if dataset.download_dataset_template():
            toast('Plantilla de dataset descarregada correctament!')
        else:
            toast('ERROR')

    def download_record(self):
        if record.number_elements() > 0:
            if record.download_record():
                toast('Registre descarregat correctament!')
            else:
                toast('ERROR')
        else:
            toast('No hi ha cap element al registre')

    def remove_record(self):
        record.remove_all()
        self.ids.container.clear_widgets()
        self.ids.record_item.secondary_text = 'Encara no hi ha cap element al registre'

    def open_filechooser(self):
        if check_permission(Permission.READ_EXTERNAL_STORAGE):
            try:
                if api_version > 28:
                    picker.Picker(callback=self.get_csv_path).pick_file()
                else:
                    self.file_manager.show('/sdcard')
            except Exception as e:
                print(e)
                toast('ERROR', gravity=80, length_long=1)
        else:
            toast("Otorga accés a l'emmagatzament a l'aplicació", gravity=80, length_long=1)
            request_permissions([Permission.READ_EXTERNAL_STORAGE])

    def select_path(self, path):
        self.exit_manager()
        self.get_csv_path(path)
    
    def exit_manager(self, *args):
        self.file_manager.close()

    def get_csv_path(self, selected_file):
        try:
            dataset.remove()
            file_path = android_scoped_storage.SharedStorage().retrieveUri(selected_file)
            if file_path[:].lower().endswith('.csv'):
                dataset.save_path(file_path)
                self.ids.dataset_item.secondary_text = dataset.dataset_name
                self.ids.dataset_item.tertiary_text = f"nombre d'elements: {str(dataset.number_elements)}"
                toast('Arxiu seleccionat correctament!', gravity=80, length_long=1)
            else:
                os.remove(file_path)
                self.ids.dataset_item.secondary_text = 'Ningun arxiu seleccionat'
                self.ids.dataset_item.tertiary_text = 'Ningun arxiu seleccionat'
                toast('Selecciona un arxiu .csv', gravity=80, length_long=1)
        except Exception as e:
            print(e)
            toast('ERROR', gravity=80, length_long=1)

    def load_record(self):
        num_elements = record.number_elements()
        if num_elements != 0:
            for index in range(num_elements):
                element = record.read(index + 1)
                item = ListElement(data=element, text=str(element[0]), secondary_text=element[6], on_release=self.open_info_dialog)
                if element[1] == str(1.0):
                    icon = IconLeftWidget(icon='arrow-up')
                elif element[1] == str(0.5):
                    icon = IconLeftWidget(icon='arrow-top-right')
                elif element[1] == str(0.0):
                    icon = IconLeftWidget(icon='arrow-right')
                elif element[1] == str(-0.5):
                    icon = IconLeftWidget(icon='arrow-bottom-right')
                elif element[1] == str(-1.0):
                    icon = IconLeftWidget(icon='arrow-down')
                item.add_widget(icon)
                self.ids.container.add_widget(item)

    def update_record(self, values):
        record.write(values)
        item = ListElement(data=values, text=str(values[0]), secondary_text=values[6], on_release=self.open_info_dialog)
        if values[1] == 1:
            icon = IconLeftWidget(icon='arrow-up')
        elif values[1] == 0.5:
            icon = IconLeftWidget(icon='arrow-top-right')
        elif values[1] == 0:
            icon = IconLeftWidget(icon='arrow-right')
        elif values[1] == -0.5:
            icon = IconLeftWidget(icon='arrow-bottom-right')
        elif values[1] == -1:
            icon = IconLeftWidget(icon='arrow-down')
        item.add_widget(icon)
        self.ids.container.add_widget(item)
        self.load_record_label()
        # self.ids.chart.update_chart((values[0], sum(x * int(t) for x, t in zip([1, 1/60], values[6].split(':')))), (values[5], sum(x * int(t) for x, t in zip([1, 1/60], values[6].split(':'))) + 2/3))
        

    def train_model(self):
        if dataset.selected():
            web_service.api_connection(dataset.get_dataset_file())
            model.load()
        else:
            toast('No hi ha ningun arxiu .csv seleccionat', gravity=80, length_long=1)

    def open_info_dialog(self, item):
        self.selected_item = item
        self.dialog = MDDialog(
                title='Informació',
                text=f"Glucosa en sang (mg/dL): {item.data[0]}\n\nTendència: {item.data[1]}\n\nUnitats d'insulina administrada (0,01 x mL): {item.data[2]}\n\nRacions d'hidrats de carboni ingerits (100 x g): {item.data[3]}\n\nPredicció de glucosa en sang en 40 min (mg/dL): {item.data[4]}\n\nPredicció de tendència en 40 min: {item.data[5]}\n\nHora: {item.data[6]}\n\nData: {item.data[7]}\n\n",
                buttons=[
                    MDFlatButton(
                        text='ESBORRA',
                        font_style='Button',
                        on_release=self.remove_item,
                        theme_text_color='Custom'
                    ),
                    MDRaisedButton(
                        text='OK',
                        font_style='Button',
                        on_release=self.close_dialog,
                        theme_text_color='Custom'
                    )
                ]
            )
        self.dialog.open()

    def remove_item(self, *args):
        self.ids.container.remove_widget(self.selected_item)
        record.remove(self.selected_item.data)
        self.ids.record_item.secondary_text = 'Encara no hi ha cap element al registre'
        self.close_dialog()

    def close_dialog(self, *args):
        self.dialog.dismiss()

class IconListItem(OneLineIconListItem):
    icon = StringProperty()

class InputScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.values = [None] * 4   
        self.tendency_menu = MDDropdownMenu(
            position='bottom',
            width_mult=4
        )
        
    def build_menu(self):   
        menu_items = [
            {
                'viewclass': 'IconListItem',
                'text': '1',
                'height': dp(56),
                'on_release': lambda x = '1': self.set_item(x),
                'left_icon': 'arrow-up'
            },
            {
                'viewclass': 'IconListItem',
                'text': '0.5',
                'height': dp(56),
                'on_release': lambda x = '0.5': self.set_item(x),
                'left_icon': 'arrow-top-right'
            },
            {
                'viewclass': 'IconListItem',
                'text': '0',
                'height': dp(56),
                'on_release': lambda x = '0': self.set_item(x),
                'left_icon': 'arrow-right'
            },
            {
                'viewclass': 'IconListItem',
                'text': '-0.5',
                'height': dp(56),
                'on_release': lambda x = '-0.5': self.set_item(x),
                'left_icon': 'arrow-bottom-right'
            },
            {
                'viewclass': 'IconListItem',
                'text': '-1',
                'height': dp(56),
                'on_release': lambda x = '-1': self.set_item(x),
                'left_icon': 'arrow-down'
            }
        ]
        
        self.tendency_menu.items = menu_items
        self.tendency_menu.caller = self.ids.field2
        
        self.tendency_menu.bind()

    def open_tendency_menu(self):
        self.tendency_menu.open()

    def set_item(self, selection):
        self.values[1] = float(selection)
        self.ids.field2.text = selection
        self.tendency_menu.dismiss()

    def get_values(self, value, index):
        try:
            if index == 0:
                self.values[index] = int(value)
            else:
                self.values[index] = float(value)
        except:
            self.values[index] = None   

    def send_input(self):
        if self.values[0] != None and self.values[1] != None and self.values[2] != None and self.values[3] != None:
            input = self.values[:]
            time = datetime.now().strftime('%H:%M')
            date = datetime.now().strftime('%d/%m/%Y')
            prediction = model.predict(input[:])
            input.append(prediction[0])
            input.append(prediction[1])
            input.append(time)
            input.append(date)
            self.manager.get_screen('main_screen').update_record(input[:])
            self.remove()
        else:
            toast('Omple tots els camps', gravity=80, length_long=1)

    def remove(self):
        self.parent.current = 'main_screen'
        self.values = [None] * 4
        self.ids.field1.text = ''
        self.ids.field2.text = ''
        self.ids.field3.text = ''
        self.ids.field4.text = ''

    def open_help_dialog(self):
        help_text = '''Glucosa en sang: introdueix els nivells de glucosa en sang en mil·ligrams de glucosa per decilitre de sang. \n\nTendència: introdueix la tendència dels nivells de glucosa en sang. 1 si pujen molt ràpid, 0.5 si pujen, 0 si estan estables, -0.5 si baixen i -1 si baixen en picat. \n\nUnitats: introdueix les unitats administrades d'insulina ràpida. Una unitat equival a una centèssima part de mil·lilitre d'insulina ràpida U-100. \n\nRacions: introdueix el nombre de racions ingerides d'hidrats de carboni. Una ració equival a 100 grams d'hidrats de carboni.'''

        self.dialog = MDDialog(
                title='Ajuda',
                text=help_text,
                buttons=[
                    MDFlatButton(
                        text='OK',
                        font_style='Button',
                        on_release=self.close_dialog,
                        theme_text_color='Custom'
                    )
                ]
            )
        
        self.dialog.open()

    def close_dialog(self, *args):
        self.dialog.dismiss()


class RecommendScreen(Screen):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.values = [None] * 4
        self.processing = False
        self.tendency_menu = MDDropdownMenu(
            position='bottom',
            width_mult=4,
        )

    def build_menu(self):
        menu_items = [
            {
                'viewclass': 'IconListItem',
                'text': '1',
                'height': dp(56),
                'on_release': lambda x = '1': self.set_item(x),
                'left_icon': 'arrow-up'
            },
            {
                'viewclass': 'IconListItem',
                'text': '0.5',
                'height': dp(56),
                'on_release': lambda x = '0.5': self.set_item(x),
                'left_icon': 'arrow-top-right'
            },
            {
                'viewclass': 'IconListItem',
                'text': '0',
                'height': dp(56),
                'on_release': lambda x = '0': self.set_item(x),
                'left_icon': 'arrow-right'
            },
            {
                'viewclass': 'IconListItem',
                'text': '-0.5',
                'height': dp(56),
                'on_release': lambda x = '-0.5': self.set_item(x),
                'left_icon': 'arrow-bottom-right'
            },
            {
                'viewclass': 'IconListItem',
                'text': '-1',
                'height': dp(56),
                'on_release': lambda x = '-1': self.set_item(x),
                'left_icon': 'arrow-down'
            }
        ]

        self.tendency_menu.items = menu_items
        self.tendency_menu.caller = self.ids.recom_field2
        self.tendency_menu.bind()

    def open_tendency_menu(self):
        self.tendency_menu.open()

    def set_item(self, selection):
        self.values[1] = float(selection)
        self.ids.recom_field2.text = selection
        self.tendency_menu.dismiss()
    
    def get_values(self, value, index):
        try:
            if index == 0:
                self.values[index] = int(value)
            else:
                self.values[index] = float(value)
        except:
            self.values[index] = None
        
    def recommend(self):
        optimal_values = []
        last = None
        for insulin in np.arange(0, 21, 0.5).tolist():
            for portions in np.arange(0, 11, 0.5).tolist():
                data = [self.values[0], self.values[1], insulin, portions]
                prediction = model.predict(data[:])
                if optimal_values == []:
                    optimal_values = [insulin, portions]
                else:
                    if abs(prediction[0] - 95) + insulin + portions < abs(last[0] - 95) + optimal_values[0] + optimal_values[1]:
                        optimal_values = [insulin, portions]
                last = prediction[:]

        self.values[2] = optimal_values[0]
        self.values[3] = optimal_values[1]


    def send_input(self):
        if self.values[0] != None and self.values[1] != None:
            self.processing = True
            self.recommend()
            input = self.values[:]
            time = datetime.now().strftime('%H:%M')
            date = datetime.now().strftime('%d/%m/%Y')
            prediction = model.predict(input[:])
            input.append(prediction[0])
            input.append(prediction[1])
            input.append(time)
            input.append(date)
            self.manager.get_screen('main_screen').update_record(input[:])
            self.processing = False
            self.remove()
        else:
            toast('Omple tots els camps', gravity=80, length_long=1)

    def remove(self):
        self.parent.current = 'main_screen'
        self.values = [None] * 4
        self.ids.recom_field1.text = ''
        self.ids.recom_field2.text = ''

    def open_help_dialog(self):
        help_text = '''Glucosa en sang: introdueix els nivells de glucosa en sang en mil·ligrams de glucosa per decilitre de sang. \n\nTendència: introdueix la tendència dels nivells de glucosa en sang. 1 si pujen molt ràpid, 0.5 si pujen, 0 si estan estables, -0.5 si baixen i -1 si baixen en picat.'''

        self.dialog = MDDialog(
                title='Ajuda',
                text=help_text,
                buttons=[
                    MDFlatButton(
                        text='OK',
                        font_style='Button',
                        on_release=self.close_dialog,
                        theme_text_color='Custom'
                    )
                ]
            )

        self.dialog.open()

    def close_dialog(self, *args):
        self.dialog.dismiss()



'''
class Chart(Image):
    image = ObjectProperty()

    def __init__(self, **kwargs):
        super(Image, self).__init__(**kwargs)
        #reversed(plt.rcParams['figure.figsize'])
        plt.rcParams['figure.dpi'] = 90
        self.x_values = []
        self.y_values = []
        self.x_prediction = []
        self.y_prediction = []
        self.image = ObjectProperty()
        self.update_chart()
        
    def update_chart(self, value=None, prediction=None):
        if value != None:
            self.x_values.append(value[1])
            self.y_values.append(value[0])

            self.x_prediction.append(prediction[1])
            self.y_prediction.append(prediction[0])
            
        buf = BytesIO()
        plt.style.use('ggplot')
        plt.xlim(0, 24)
        plt.ylim(0, 500)
        plt.plot(self.x_values, self.y_values, marker='o')
        plt.plot(self.x_prediction, self.y_prediction, '--', marker='o')
        plt.autoscale()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = pil.Image.open(buf).convert('RGB')
        pixels = np.array(img)
        pixels[(pixels == (255, 255, 255)).all(axis = -1)] = (250, 250, 250)
        img2 = pil.Image.fromarray(pixels, mode='RGB')
        buf = BytesIO()
        img2.save(buf, format='png')
        buf.seek(0)
        self.image = CoreImage(buf, ext='png').texture
'''




class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.icon = 'icon.png'
        self.kv = Builder.load_file('layout.kv')

    def build(self):
        self.theme_cls.material_style = 'M3'
        self.theme_cls.theme_style = 'Light'
        self.theme_cls.primary_palette = 'Blue'
        self.theme_cls.accent_palette = 'Blue'
        self.title = 'SmartSugar'
        
        return self.kv

if __name__ == '__main__':
    MainApp().run()