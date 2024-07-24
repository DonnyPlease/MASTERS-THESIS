import sys, os

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

from PyQt6.QtWidgets import QDialog
from PyQt6 import uic

from ModelsSettings import Ui_modelsOptionsDialog

import logging

class ModelsOptionsDialog(QDialog, Ui_modelsOptionsDialog):
    def __init__(self, parent=None, options=None):
        super(ModelsOptionsDialog, self).__init__(parent)
        self.setupUi(self)
        self.main_window = parent
        self.confimModelOptionsButton.clicked.connect(self.confirm_button_clicked)
        self.options = options
        
        self.nn_option_checkbox.setChecked(options.nn_option)
        self.gp_option_checkbox.setChecked(options.gp_option)
        self.svr_option_checkbox.setChecked(options.svr_option)
    
        
    def update_options(self):
        self.options.nn_option = self.nn_option_checkbox.isChecked()
        self.options.gp_option = self.gp_option_checkbox.isChecked()
        self.options.svr_option = self.svr_option_checkbox.isChecked()
        logging.debug(f"NN: {self.options.nn_option}, GP: {self.options.gp_option}, SVR: {self.options.svr_option}")
        
    def confirm_button_clicked(self):
        self.update_options()
        self.main_window.load_models()
        self.close()
        
class ModelsOptions:
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.nn_option = True
        self.gp_option = True
        self.svr_option = True
        
    def show_dialog(self):
        dialog = ModelsOptionsDialog(self.main_window, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            pass