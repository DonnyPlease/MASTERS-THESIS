import sys, os

# Add the project path
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)


import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog, QTableWidgetItem
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from draw_dataset import AXIS_NAME, TICKS

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.colorbar = None
        super(GraphCanvas, self).__init__(self.fig)
        
        # Connects
        pass

    def _draw(self, X,Y,Z, axes=["l","a"]):
        if self.colorbar is not None:
            self.colorbar.remove()
            
        self.axes.cla()
        c = self.axes.pcolormesh(X, Y, Z, cmap='inferno', shading='auto')
        self.colorbar = self.fig.colorbar(c, label=r'$T_{hot}$ [keV]', ax=self.axes)
        
        self._set_x_axis(axes[0])
        self._set_y_axis(axes[1])
        self.draw()
        
    def draw_slice(self, i_values, l_values, a_values, values_to_plot, slice_at=0, axes=["l","a"]):
        final_values = None
        x_grid = None
        y_grid = None
        
        if axes == ["l","a"]:
            final_values = values_to_plot[:,slice_at,:]
            final_values = final_values.T
            x_grid = l_values
            y_grid = a_values
        elif axes == ["i","l"]:
            final_values = values_to_plot[:,:,slice_at]
            x_grid = i_values
            y_grid = l_values
        elif axes == ["i","a"]:
            final_values = values_to_plot[slice_at,:,:]
            final_values = final_values.T
            x_grid = i_values
            y_grid = a_values
        
        self._draw(x_grid, y_grid, final_values, axes=axes)
        
    def _set_x_axis(self, axis):
        if axis == "l":
            self.axes.set_xlabel(AXIS_NAME["l"])
            self.axes.set_xscale("log")
        elif axis == "i":
            self.axes.set_xlabel(AXIS_NAME["i"])
            self.axes.set_xscale("log")
        elif axis == "a":
            self.axes.set_xlabel(AXIS_NAME["a"])
            
            
    def _set_y_axis(self, axis):
        if axis == "l":
            self.axes.set_ylabel(AXIS_NAME["l"])
            self.axes.set_yscale("log")
        elif axis == "i":
            self.axes.set_ylabel(AXIS_NAME["i"])
            self.axes.set_yscale("log")
        elif axis == "a":
            self.axes.set_ylabel(AXIS_NAME["a"])
        