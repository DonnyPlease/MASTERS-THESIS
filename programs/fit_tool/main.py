import sys, os

import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog, QTableWidgetItem
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Histogram import Histogram
from MainWindow import Ui_MainWindow
from MplCanvas import MplCanvas
from Dataset import DatasetUtils, DatasetRecord
               

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # # Replace list view with custom list view
        # self.listViewHistograms = ListWidgetHistograms()
        # self.layout.replaceWidget(self.listViewHistograms, self.listWidgetHistograms)
        # self.deleteLater(self.listViewHistograms)
         
        # Connects
        self.pushButtonSetRange.clicked.connect(self.set_range)
        self.pushButtonRemoveRange.clicked.connect(self.clear_range)
        self.chooseFolderButton.clicked.connect(self.choose_folder)
        self.listViewHistograms.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.pushButtonFit.clicked.connect(self.fit_selected_range)
        self.checkBoxShowOriginal.clicked.connect(self.histogram_settings_update)
        self.checkBoxShowCustomFit.clicked.connect(self.histogram_settings_update)
        self.pushButtonChooseDataset.clicked.connect(self.choose_dataset)
        self.pushButtonFit.clicked.connect(self.perform_fit)
        self.pushButtonSaveCustom.clicked.connect(self.save_custom_to_dict)
        self.pushButtonSaveOriginal.clicked.connect(self.save_original_fit_to_final)
        self.actionSave.triggered.connect(self.save_final_dataset)
        
        # Create canvas for plotting histograms and bind it into the frame
        # in the middle of the window
        self.mpl_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.frameLayout = QVBoxLayout(self.framePlot)
        self.frameLayout.addWidget(self.mpl_canvas)
        
        
        self.set_columns_in_table()
        
        # Check check check the box, check the other too
        self.checkBoxShowOriginal.setChecked(True) 
        self.checkBoxShowCustomFit.setChecked(True) 

        self.histograms_folder = ''
        self.selected_histogram_path = ''
        
        
        self.autofit_dataset = {}
        self.final_dataset = {}
        
        self.current_autofit = None
        self.current_custom_fit = None
        
        self.initialize_folders()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Check if QListWidget has focus
            if self.listViewHistograms.hasFocus():
                selected_item = self.listViewHistograms.currentItem()
                self.try_select_item(selected_item)
        else:
            super().keyPressEvent(event)
    
    def choose_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select a folder")
        self.histograms_folder = folder_path
        print(folder_path)
        if folder_path:
            self.populate_list(folder_path)
            self.set_text_label_folder_path()
        
    def choose_dataset(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select a folder")
        self.dataset_folder = folder_path
        if folder_path:
            print("loading dataset")
            self.load_dataset()
            print("populating table")
            try:
                self.populate_table()
                print("Table was successfuly populated.")
            except:
                print("There were errors while trying to populate the table.")
        self.mpl_canvas.draw_histogram()

    def set_text_label_folder_path(self):
        path_checkpoints = self.histograms_folder.split('/')
        
        path = ''
        for i in range(max(len(path_checkpoints)-3,0),len(path_checkpoints)):
            path += path_checkpoints[i]+'/'
        self.labelFolderPath.setText("Path: "+ path)
    
    def set_columns_in_table(self):
        self.tableWidget.setColumnWidth(0,120)
        self.tableWidget.setColumnWidth(1,120)
    
    def populate_list(self, folder_path):
        self.listViewHistograms.clear()
        folders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
        self.listViewHistograms.addItems(folders)

    def on_item_double_clicked(self, item):
        self.try_select_item(item)
        
    def try_select_item(self, item):
        try:
           self.new_item_chosen(item) 
        except:
            print("""Item you clicked has some problems. Check, whether its 
                  name is in the correct format \n and whether it contains 
                  files: bins.txt. counts.txt""")

    def new_item_chosen(self, item):
        self.clear_range()
        self.load_histogram_from_selected_item(item)
        self.update_current_fit_data()
        if self.current_autofit is not None:
            self.mpl_canvas.autofit_result = self.current_autofit
        else:
            self.mpl_canvas.autofit_result = None
            
        if self.current_custom_fit is not None:
            self.mpl_canvas.custom_fit_result = self.current_custom_fit
        else:
            self.mpl_canvas.custom_fit_result = None
            
        self.populate_table()
        self.mpl_canvas.draw_histogram()

    def clear_range(self):
        self.mpl_canvas.clear_range()
    
    def load_histogram_from_selected_item(self, item):
        folder_name = item.text()
        folder_path = self.histograms_folder + '/' + folder_name
        self.mpl_canvas.histogram.load_from_folder(folder_path)
    
    def initialize_folders(self):
        self.histograms_folder = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/old_data/trimmed_histograms'
        self.populate_list(self.histograms_folder)
        
        self.dataset_folder = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/dataset'
        self.load_dataset()
        
        
    def set_range(self):
        self.mpl_canvas.draw_histogram()
        self.mpl_canvas.setting_range = 1
        print("setting range")

    def fit_selected_range(self):
        self.mpl_canvas.fit_selected_range()
        
    def histogram_settings_update(self):
        self.mpl_canvas.show_original_fit = self.checkBoxShowOriginal.isChecked()
        self.mpl_canvas.show_custom_fit = self.checkBoxShowCustomFit.isChecked()
        self.mpl_canvas.draw_histogram()
        
    def load_dataset(self):
        self.final_dataset, self.autofit_dataset = DatasetUtils.load_datasets_to_dicts(self.dataset_folder)
    
    def populate_table(self):
        self.update_current_fit_data()
        self.populate_table_column_with_data(self.current_autofit,0)
        self.populate_table_column_with_data(self.current_custom_fit,1)
    
    def update_current_fit_data(self):
        if len(self.autofit_dataset.keys()) == 0: return
        hist = self.mpl_canvas.histogram
        if hist is None: return
        I = '1e' + str(hist.I)
        L = "{:.2f}".format(hist.L)
        alpha = str(hist.alpha)
        key = (I,L,alpha)
        print(key)
        if self.autofit_dataset.get(key) is not None:
            self.current_autofit = self.autofit_dataset[key]
        else:
            self.current_autofit = None
            
        if self.final_dataset.get(key) is not None:
            self.current_custom_fit = self.final_dataset[key]
        else:
            self.current_custom_fit = None
    
    def populate_table_column_with_data(self, data, col):
        if data is None: data = DatasetRecord()
        self.tableWidget.setItem(0,col,QTableWidgetItem(data.I)) 
        self.tableWidget.setItem(1,col,QTableWidgetItem(data.L)) 
        self.tableWidget.setItem(2,col,QTableWidgetItem(data.alpha)) 
        self.tableWidget.setItem(3,col,QTableWidgetItem("{:.4f}".format(float(data.t_hot))+' ± '+"{:.4f}".format(float(data.t_hot_stdev)))) 
        self.tableWidget.setItem(4,col,QTableWidgetItem(data.type)) 
        self.tableWidget.setItem(5,col,QTableWidgetItem("{:.4f}".format(float(data.min_energy))))
        self.tableWidget.setItem(6,col,QTableWidgetItem("{:.4f}".format(float(data.max_energy))))
        self.tableWidget.setItem(7,col,QTableWidgetItem("{:.4f}".format(float(data.a))))
        self.tableWidget.setItem(8,col,QTableWidgetItem("{:.4f}".format(float(data.b))))
        self.tableWidget.setItem(9,col,QTableWidgetItem("{:.4f}".format(float(data.c))))
        self.tableWidget.setItem(10,col,QTableWidgetItem("{:.4f}".format(float(data.d))))
        self.tableWidget.setItem(11,col,QTableWidgetItem("{:.4f}".format(float(data.e))))
        self.tableWidget.setItem(12,col,QTableWidgetItem("{:.4f}".format(float(data.f))))
        self.tableWidget.setItem(13,col,QTableWidgetItem("{:.4f}".format(float(data.g))))
    
    def perform_fit(self):
        self.custom_fit_result = self.mpl_canvas.fit_selected_range()
        self.populate_table_column_with_data(self.custom_fit_result, 1)
        
    def save_custom_to_dict(self):
        print("Saving ", end='')
        try:
            key = self.custom_fit_result.make_key()
            print("with key:")
            print(key)
        
            self.final_dataset[key] = self.custom_fit_result
            self.save_final_dataset()
            print("Save successful")
        except:
            print("failed")
        
    def save_original_fit_to_final(self):
        key = self.current_autofit.make_key()
        self.final_dataset[key] = self.current_autofit
        self.save_final_dataset()
    
    def save_final_dataset(self):
        DatasetUtils.overwrite_final(self.dataset_folder, self.final_dataset)
    
if __name__ == "__main__": 
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()