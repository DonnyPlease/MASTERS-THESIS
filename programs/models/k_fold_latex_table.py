import os
import sys
PATH_TO_PROJECT = 'C:/Users/samue/OneDrive/Dokumenty/FJFI/MASTERS-THESIS/programs/'
sys.path.append(PATH_TO_PROJECT)
PATH_TO_MODELS = PATH_TO_PROJECT + 'models/models/'

import numpy as np
import json

if __name__ == "__main__":
    stats = json.load(open('models/k_fold_stats.json', 'r'))
    
    latex_table = ""
    # Latex table from the k-fold stats
    latex_table += "\\begin{table}[h]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{ c c c c c c c }\n"
    latex_table += "\\toprule\n"
    latex_table += "\\  & $SVR_\\mathrm{rmse}$& $SVR_\\mathrm{r2}$ & $NN_\\mathrm{rmse}$   & $NN_\\mathrm{r2}$ & $GP_\\mathrm{rmse}$ & $GP_\\mathrm{r2}$ \\\\ \n"
    
    latex_table += "\\midrule\n"
    for i in range(8):
        latex_table += f"Fold {i+1} & {stats['svr']['rmse'][i]:.2f} & {stats['svr']['r2'][i]:.2f} & {stats['nn']['rmse'][i]:.2f} & {stats['nn']['r2'][i]:.2f} & {stats['gp']['rmse'][i]:.2f} & {stats['gp']['r2'][i]:.2f} \\\\ \n"
    
    svr_rmse_mean = np.mean(stats['svr']['rmse'])
    nn_rmse_mean = np.mean(stats['nn']['rmse'])
    gp_rmse_mean = np.mean(stats['gp']['rmse'])
    
    latex_table += "\\midrule\n"
    latex_table += f"Mean & {svr_rmse_mean:.2f} & & {nn_rmse_mean:.2f} & & {gp_rmse_mean:.2f} & \\\\ \n"
    latex_table += "\\botomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Cross-validation of SVR, NN and GP models.} \n \\label{tab:cross-val} \n"
    latex_table += "\\end{table}\n"

    with open('models/k_fold_latex_table.txt', 'w') as f:
        f.write(latex_table)
