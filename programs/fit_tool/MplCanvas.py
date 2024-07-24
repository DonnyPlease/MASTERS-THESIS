import numpy as np
import statsmodels.api as sm

from Histogram import Histogram
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from Dataset import DatasetRecord


from scipy.optimize import curve_fit


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        
        # Connects
        self.mpl_connect('button_press_event', self.on_click)
        
        self.custom_fit_result = None
        self.autofit_result = None
        
        self.histogram = Histogram()
        self.setting_range = 0
        self.show_original_fit = True
        self.show_custom_fit = True
        
    def set_histogram(self, histogram):
        self.histogram = histogram
        
    def draw_histogram(self):
        self.axes.cla()
        self.axes.scatter(self.histogram.bins, self.histogram.counts,c="black",zorder=5,label="Simulation data")
        self.axes.set_yscale('log')
        
        self.axes.set_ylabel(r"$N$")
        self.axes.set_xlabel("E [keV]")
        self.axes.grid(True, zorder=0)
        
        
         
        if self.histogram.left_cut is not None:
            self.axes.axvline(x=self.histogram.left_cut, color='black', linestyle='--',zorder=6)
        if self.histogram.right_cut is not None:
            self.axes.axvline(x=self.histogram.right_cut, color='black', linestyle='--',zorder=6)
        if self.show_custom_fit and self.custom_fit_result is not None:
            self.plot_custom_fit()
        if self.show_original_fit and self.autofit_result is not None:
            self.plot_original_fit()
        
        self.axes.legend()
        try:
            self.draw_custom_legend()
        except:
            pass
            
        self.draw()
    
    def draw_custom_legend(self):
        handles, labels = self.axes.get_legend_handles_labels()
        custom_line = Line2D([0], [0], color='white', lw=2, linestyle='--')
        handles.append(custom_line)
        label = r"$T_\mathrm{hot}$" +" = {} ± {} keV".format(self.custom_fit_result.t_hot, self.custom_fit_result.t_hot_stdev)
        labels.append(label)
        self.axes.legend(handles=handles, labels=labels)
        
    def on_click(self, event):
        if event.inaxes != self.axes: return
        x = event.xdata
        if x is None: return
        
        print(f"Clicked at x = {x:.2f}")
        
        match self.setting_range:
            case 1:
                self.histogram.left_cut = x
            case 2:
                self.histogram.right_cut = x
        self.setting_range = 2 if self.setting_range == 1 else 0    
        self.draw_histogram()
        
    def fit_selected_range(self):
        x, y = self.histogram.get_selected_range()
        y /= 1e10
        # popt, pcov = curve_fit(self.expoonential_function, x ,y)
        # a, b = popt
        # a, b, a_stdev, b_stdev = self.fit_linearly_using_log(x, y)
        a, b, a_stdev, b_stdev = self.fit_one_exponential(x, y)
        
        a *= 1e10
        custom_fit_result = DatasetRecord()
        custom_fit_result.I = "1e"+str(self.histogram.I)
        custom_fit_result.L = str(self.histogram.L)
        custom_fit_result.alpha = str(self.histogram.alpha)
        custom_fit_result.t_hot = f"{-1/b:.2f}"
        custom_fit_result.type = 'e1'
        custom_fit_result.min_energy = str(self.histogram.left_cut) if self.histogram.left_cut is not None else int(np.min(x))
        custom_fit_result.max_energy = str(self.histogram.right_cut) if self.histogram.right_cut is not None else int(np.max(x))
        custom_fit_result.a = str(a)
        custom_fit_result.b = str(b)
        custom_fit_result.a_stdev = str(a_stdev)
        custom_fit_result.b_stdev = str(b_stdev)
        custom_fit_result.t_hot_stdev = f"{b_stdev/b**2:.2f}"
        self.custom_fit_result = custom_fit_result
        self.draw_histogram()
        return custom_fit_result

    def clear_range(self):
        self.histogram.left_cut = None
        self.histogram.right_cut = None
        self.draw_histogram()
    
    def fit_linearly_using_log(self, x, y):
        y_log = np.log(y)
        x_for_fit = x.reshape(-1, 1)
        x_for_fit = sm.add_constant(x_for_fit)
        model = sm.OLS(y_log, x_for_fit)
        results = model.fit()
        
        a = results.params[0]
        b = results.params[1]
        a_stdev = results.bse[0]
        b_stdev = results.bse[1]
        A = np.exp(a)  # Transform intercept to exponential form
        b = b  # Slope is the same
        A_stdev = a_stdev*A  # Transform intercept to exponential form
        return A,b, A_stdev, b_stdev
    
    def fit_one_exponential(self, x, y):
        a,b, *_ = self.fit_linearly_using_log(x,y)
        popt, pcov = curve_fit(self.expoonential_function, x ,y, p0=[a,b], sigma=np.sqrt(y))
        a, b = popt
        a_stdev, b_stdev = np.sqrt(np.diag(pcov))
        return a, b, a_stdev, b_stdev
     
    def expoonential_function(self, x, a, b):
        return a*np.exp(b*x)
    
    def double_exponential_function(self, x, a, b, c, d):
        return a*np.exp(b*x)+c*np.exp(d*x)
    
    def plot_custom_fit(self):
        x = np.array(self.histogram.bins)
        if self.custom_fit_result.type == 'e1':
            y = self.expoonential_function(x, float(self.custom_fit_result.a), float(self.custom_fit_result.b))
        elif self.custom_fit_result.type == 'j2_wo':
            y = self.double_exponential_function(x, 
                                                 float(self.autofit_result.b),
                                                 float(self.autofit_result.c),
                                                 float(self.autofit_result.d),
                                                 float(self.autofit_result.e))

        self.axes.plot(x, y, color='red',zorder=7, label=r"$N = a_1 \mathrm{e}^{b_1 E}$")
        
    def plot_original_fit(self):
        x = np.array(self.histogram.bins)
        if self.autofit_result.type == 'j2_wo':
            y = self.double_exponential_function(x, 
                                                 float(self.autofit_result.b),
                                                 float(self.autofit_result.c),
                                                 float(self.autofit_result.d),
                                                 float(self.autofit_result.e))

        self.axes.plot(x, y, color='orange')