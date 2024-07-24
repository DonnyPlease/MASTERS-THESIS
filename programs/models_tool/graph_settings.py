import sys, os

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)

import logging

class GraphSettings:
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.main_window.i_min_spinbox.valueChanged.connect(self.set_i_min)
        self.main_window.i_max_spinbox.valueChanged.connect(self.set_i_max)
        self.main_window.i_count_spinbox.valueChanged.connect(self.set_i_count)
        self.main_window.fix_i_radio.toggled.connect(self.set_fix_i)
        
        self.main_window.l_min_spinbox.valueChanged.connect(self.set_l_min)
        self.main_window.l_max_spinbox.valueChanged.connect(self.set_l_max)
        self.main_window.l_count_spinbox.valueChanged.connect(self.set_l_count)
        self.main_window.fix_l_radio.toggled.connect(self.set_fix_l)
        self.main_window.fix_a_radio.toggled.connect(self.set_fix_a)
        
        self.main_window.a_min_spinbox.valueChanged.connect(self.set_a_min)
        self.main_window.a_max_spinbox.valueChanged.connect(self.set_a_max)
        self.main_window.a_count_spinbox.valueChanged.connect(self.set_a_count)
        
        self.fix_i = True
        self.main_window.fix_i_radio.setChecked(True)
        self.main_window.i_count_spinbox.setEnabled(False)
        self.fix_l = False
        self.fix_a = False
        
        self.i_min = 17
        self.i_max = 19
        self.i_count = 1
        self.l_min = 0.01
        self.l_max = 5.0
        self.l_count = 50
        self.a_min = 0
        self.a_max = 60
        self.a_count = 50
       
    def get_axes(self):
        axes = ["l","a"]
        if self.fix_i:
            axes = ["l","a"]
        elif self.fix_l:
            axes = ["i","a"]
        elif self.fix_a:
            axes = ["i","l"]
        return axes
        
    def set_i_min(self):
        self.i_min = self.main_window.i_min_spinbox.value()
        logging.debug(f"i_min: {self.i_min}")
        
    def set_i_max(self):
        self.i_max = self.main_window.i_max_spinbox.value()
        logging.debug(f"i_max: {self.i_max}")
        
    def set_i_count(self):
        self.i_count = self.main_window.i_count_spinbox.value()
        logging.debug(f"i_count: {self.i_count}")
        
    def set_fix_i(self):
        self.fix_i = self.main_window.fix_i_radio.isChecked()
        if self.fix_i:
            self.main_window.i_count_spinbox.setValue(1)
            self.main_window.i_count_spinbox.setEnabled(False)
        else:
            self.main_window.i_count_spinbox.setEnabled(True) 
        logging.debug(f"fix_i: {self.fix_i}")

    def set_l_min(self):
        self.l_min = self.main_window.l_min_spinbox.value()
        logging.debug(f"l_min: {self.l_min}")
        
    def set_l_max(self):
        self.l_max = self.main_window.l_max_spinbox.value()
        logging.debug(f"l_max: {self.l_max}")
    
    def set_l_count(self):
        self.l_count = self.main_window.l_count_spinbox.value()
        logging.debug(f"l_count: {self.l_count}")
    
    def set_fix_l(self):
        self.fix_l = self.main_window.fix_l_radio.isChecked()
        if self.fix_l:
            self.main_window.l_count_spinbox.setValue(1)
            self.main_window.l_count_spinbox.setEnabled(False)
        else:
            self.main_window.l_count_spinbox.setEnabled(True)
        logging.debug(f"fix_l: {self.fix_l}")
        
    def set_a_min(self):
        self.a_min = self.main_window.a_min_spinbox.value()
        logging.debug(f"a_min: {self.a_min}")
        
    def set_a_max(self):
        self.a_max = self.main_window.a_max_spinbox.value()
        logging.debug(f"a_max: {self.a_max}")
        
    def set_a_count(self):
        self.a_count = self.main_window.a_count_spinbox.value()
        logging.debug(f"a_count: {self.a_count}")
        
    def set_fix_a(self):
        self.fix_a = self.main_window.fix_a_radio.isChecked()
        if self.fix_a:
            self.main_window.a_count_spinbox.setValue(1)
            self.main_window.a_count_spinbox.setEnabled(False)
        else:
            self.main_window.a_count_spinbox.setEnabled(True)
        logging.debug(f"fix_a: {self.fix_a}")
        
    