from scipy.interpolate import griddata
from fit_tool.Dataset import DatasetUtils
import numpy as np
import matplotlib.pyplot as plt

T_HOT, T_HOT_STDEV = 0, 1

I_AXIS_NAME = r'$I$ [W/cm$^2$]'
L_AXIS_NAME = r'$L$ [μm]'
A_AXIS_NAME =  r'$\alpha$ [°]'

AXIS_NAME = {
    "i": I_AXIS_NAME,
    "l": L_AXIS_NAME,
    "a": A_AXIS_NAME
}

TICKS = {
    "i": [1e17, 1e18, 1e19],
    "l": [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5],
    "a": [0,10,20,30,40,50,60]
}


def plot_maximum_absorption_curve():
    curve_x = np.logspace(-1.13,0.75,30)
    curve_y = np.arcsin(0.68*np.power(curve_x*2*np.pi,np.ones_like(curve_x)*-1/3))*180/np.pi
    plt.plot(curve_x,curve_y, color='#7FFF00',linewidth=2)

def draw_slice(i_values, l_values, a_values, values_to_plot, slice_at=0, axes=["l","a"]):
    final_values = None
    x_grid = None
    y_grid = None
    
    if axes == ["l","a"]:
        final_values = values_to_plot[:,slice_at,:]
        final_values = final_values.T
        x_grid = l_values
        y_grid = a_values
        print("\t at I = ", i_values[slice_at])
    elif axes == ["i","l"]:
        final_values = values_to_plot[:,:,slice_at]
        x_grid = i_values
        y_grid = l_values
        print("\t at alpha = ", a_values[slice_at])
    elif axes == ["i","a"]:
        final_values = values_to_plot[slice_at,:,:]
        x_grid = i_values.T
        y_grid = a_values
        print("\tat l = ", l_values[slice_at])
    
    draw(x_grid, y_grid, final_values, axes=axes)
    

def draw(X,Y,Z, axes=["l","a"], scatterPoints=None):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, cmap='inferno', shading='auto')
    plt.colorbar(label=r'$T_{hot}$ [keV]',format="%.3f")
    
    if scatterPoints is not None:
        plt.scatter(scatterPoints[0],scatterPoints[1],c='white',s=3,marker='o')
            
    # Set x-axis 
    if axes[0] in "il":
        plt.xscale('log')
    plt.xlabel(AXIS_NAME[axes[0]])
    x_ticks = TICKS[axes[0]]
    
    # Set y-axis
    if axes[1] in "il":
        plt.yscale('log')
    plt.ylabel(AXIS_NAME[axes[1]])
    y_ticks = TICKS[axes[1]]    
    
    plt.xticks(x_ticks,x_ticks)
    plt.yticks(y_ticks,y_ticks)
    plt.show()

def plot_as_lines(data, show=False, save=False):
    slice_at = data["slice_at"]
    x_ticks = data["x_ticks"]["data"]
    x_ticks_scale = data["x_ticks"]["scale"]
    data_to_plot = data["data"]
    
    plt.figure()
    colors = ['b','g','r','c','m','y','k']
    for i,line in enumerate(data_to_plot):
        x = line["x"]
        y = line["y"]
        print(x)
        print(y)
        print("\n")
        # dotted line
        plt.plot(x,y,label=line["label"],linestyle='dashed',c=colors[i],linewidth=1)
        # then scatter points
        plt.scatter(x,y,s=10,marker='o',c=colors[i],zorder=10)
    
    plt.xlabel(data["x_label"])
    plt.ylabel(data["y_label"])
    plt.xscale(x_ticks_scale)
    plt.xticks(x_ticks,x_ticks)
    plt.grid(True, zorder=0)
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig(data["save_name"])


def draw_dataset(data, what=T_HOT, show=False, save=False, add_data_points=False):
    for intensity in ["1e17", "1e18", "1e19"]:
        # Extract the data points
        x = [item[0] for item in data[intensity]]
        y = [item[1] for item in data[intensity]]
        z = []
        if what == T_HOT:
            z = [item[2] for item in data[intensity]]
        elif what == T_HOT_STDEV:
            z = [item[3] for item in data[intensity]]
        

        # Define the grid for interpolation
        # Define the range of x and y
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)

        num_points = 10
        # Define the grid
        x_grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), num_points)
        y_grid = np.linspace(np.min(y), np.max(y), num_points)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Interpolate z values onto the grid
        Z = griddata((x, y), z, (X, Y), method='linear')

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(X, Y, Z, cmap='inferno', shading='auto')
        plt.colorbar(label=r'$T_{hot}$ [keV]')
        plt.xscale('log')
        plt.xlabel(r'$L$ [μm]')
        plt.ylabel(r'$\alpha$ [°]')
        ticks=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5]
        plt.xticks(ticks,ticks)
        if show: 
            plt.show()
        
        # if intensity == "1e17":
        #     plot_maximum_absorption_curve()
        
        if save: 
            name = "dataset/I_" + intensity + "t_hot.pdf" if what == T_HOT else "dataset/I_" + intensity + "t_hot_stdev.png"
            plt.savefig(name)
            
        if add_data_points:
            plt.scatter(x,y,c='white',s=10,marker='o')
            if save: plt.savefig(name[:-4]+"_wp.png") 

if __name__ == "__main__": 
    dataset, _ = DatasetUtils.load_datasets_to_dicts('dataset')
    data = DatasetUtils.dataset_to_dict(dataset)
    draw_dataset(data, T_HOT, show=False, save=True, add_data_points=False)
    
    data17 = data["1e17"]
    data18 = data["1e18"]
    data19 = data["1e19"]
    
    # Divide data17 into lists by angle
    angles = set([item[1] for item in data17])
    angles = sorted(list(angles))
    data17_by_angle = []
    for angle in angles:
        items = [item for item in data17 if item[1] == angle]
        items = sorted(items, key=lambda x: x[0])
        y = [float(item[2]) for item in items]
        x = [float(item[0]) for item in items]
        data_line = {"x": x, "y": y, "label": r'$\alpha = $' + str(angle) + '°'}
        data17_by_angle.append(data_line)
     
    data_to_plot = {
        "slice_at": 0,
        "x_ticks": {
            "data": [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5],
            "scale": "log",
        },
        "data": data17_by_angle[6:],
        "x_label": r"$L$ [μm]",
        "y_label": r"$T_{\mathrm{hot}}$ [keV]",
        "save_name": "dataset/t_hot_l_17.png"
    }
        
    plot_as_lines(data_to_plot, show=False, save=True)