import sys, os

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import logging

import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog, QTableWidgetItem
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from MainWindow import Ui_MainWindow
from graph_settings import GraphSettings
from models_settings import ModelsOptions
from graphCanvas import GraphCanvas
from models.prediction_grid import PredictionGrid

from sklearn.svm import SVR
from models.svr import Svr
from models.gp3d_gpy import Gp3dGpy as GP
from models.gp3d_gpy import Gp3dGpy
from transformer import Transformer

logging.basicConfig(filename='models_tool/logs.log', encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a logger for matplotlib.font_manager
font_manager_logger = logging.getLogger('matplotlib.font_manager')
tick_manager_logger = logging.getLogger('matplotlib.ticker')
font_manager_logger.setLevel(logging.WARNING)
tick_manager_logger.setLevel(logging.WARNING)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        
        self.graph_settings = GraphSettings(self)
        self.models_settings = ModelsOptions(self)
        # Connect the menu actions
        self.actionOpen_folder.triggered.connect(self.choose_folder)
        self.actionModels.triggered.connect(self.models_settings.show_dialog)
        self.comboBoxSelectModel.currentIndexChanged.connect(self.load_selected_model)
        
        self.drawPredictionButton.clicked.connect(self.draw_prediction_button_clicked)
        
        self.selected_model = None
        self.Models = []
        
        self.graph_canvas = GraphCanvas(self, width=5, height=4, dpi=100)
        self.frameLayout = QVBoxLayout(self.frame)
        self.frameLayout.addWidget(self.graph_canvas)
        
        # INITIALIZE FOLDER WITH MODELS
        self.folder_with_models = PATH_TO_PROJECT + 'models/models'
        self.load_models(self.folder_with_models)
         
        logger.debug("MainWindow initialized")
        
    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        return folder
    
    def choose_folder(self):
        folder = self.pick_folder()
        if folder:
            self.Models = []
            self.folder_with_models = folder
            logger.debug(f"Folder {folder} chosen")
            self.load_models(folder)
        else:
            logger.debug("No folder chosen")
            
    def load_models(self, folder=None):
        if folder is None:
            folder = self.folder_with_models
        self.Models = []
        for file in os.listdir(folder):
            if file.endswith('.pkl'):
                if file.split('_')[0] == 'gp' and self.models_settings.gp_option:
                    self.add_model(file.split('_')[0])
                elif file.split('_')[0] == 'svr' and self.models_settings.svr_option:
                    self.add_model(file.split('_')[0])
                elif file.split('_')[0] == 'nn' and self.models_settings.nn_option:
                    self.add_model(file.split('_')[0])
                logger.debug(f"Model {file} added")
        
        self.fill_combo_box()
        self.load_selected_model()
           
    def fill_combo_box(self):
        self.comboBoxSelectModel.clear()
        self.comboBoxSelectModel.addItems(self.Models)
        return
             
    def add_model(self, model_name):
        self.Models.append(model_name)
        
    def load_selected_model(self):
        model_name = self.comboBoxSelectModel.currentText()
        if model_name == "svr":
            model = Svr()
        elif model_name == "gp":
            model = GP()
        try:
            self.selected_model = model.load(self.folder_with_models+"/"+model_name+"_model.pkl")
            logger.debug(f"Model {model_name} loaded")
        except:
            logger.debug(f"Model {model_name} not loaded")
            
    def draw_prediction(self):
        if self.selected_model is None:
            logger.debug("No model selected")
            return
        i_grid, l_grid, a_grid = self.prediction_grid.custom_grid_for_plotting(self.graph_settings)
        self.graph_canvas.draw_slice(i_grid, l_grid, a_grid, self.prediction, axes=self.graph_settings.get_axes())
        
        logger.debug("Prediction drawn")
        
    def predict(self):
        if self.selected_model is None:
            logger.debug("No model selected - cannot predict")
            return
        prediction_grid = PredictionGrid(transformer=self.selected_model.transformer)
        x_pred = prediction_grid.custom_grid_for_prediction(self.graph_settings)
        prediction = self.selected_model.predict(x_pred)
        self.prediction = prediction.reshape((prediction_grid.X.shape))
        self.prediction_grid = prediction_grid
        logger.debug("Prediction done")
        return self.prediction

    def draw_prediction_button_clicked(self):
        self.predict()
        self.draw_prediction()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())