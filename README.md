# Docummentation
This is a repository for my master thesis. This document provides description of the basic structure of this project and a little bit deeper explanation of what is going on in particular parts of the source code. 

The structure of the repo is described below.

## Folders:
1. _programs_ contains most of the source code
2. _literature_ contains some of the referenced articles and books
3. _tex_ contains the source code for the text part of the thesis

The folders _tex_ and _literature_ are unimportant and looking back maybe I did not even have to put them here.

# PROGRAMS

This folder contains a good portion of all scripts used for the thesis. The most of the programs are not yet prepared for forking the whole repo, because of hardcoded path dependancies with data and other helper modules. It was not specified as a goal of the thesis that all codes should be automatically reusable on other machines, but we acknowledge that it is a defect of this repository which should be fixed in the future. Many scripts were used for generating graphs, tables or some other one-time activity. Also, the vast majority of the programs would benefit from refactoring.

I believe I learned a lot during the development of this big project because I already see so many things that I should have been doing differently from the beginning. Feel free to use this repository as an example how things should not be done, because I made many mistakes that dragged with me untill the end. A complete refactoring is a question of future motivation, but it has to be said that it served its purpose.

For every file one is going to use, it is better if one checks for hardcoded paths (mostly path to the inside of the _programs_ folder) and modifies them accordingly.

## Requirements:

- we were using python virtual enfironment 3.12.3
- all dependencies are found in programs/requirements.txt For the reader of the thesis, probably the most imporarant subfolders are:

## Explanation of folder contents

### programs/dataset

Important contents:

- final_dataset.txt - dataset from fitting the hot electron temperatures manually used for ML models
- Dataset.py - classes and functions for easier dataset handling

### programs/fit_tool

This folder contains the source code of the UI application making manual fitting easier. 

- It is started by running main.py. 

- The user selects a folder wiht histograms - it needs to follow a certain strucure as can be found in _histograms/data/spectra/_ - bins and counts are of the same size and the names of the folders describe the simulation paramters. Looking back this is not very good for making the dataset more dense because it is restricted to certain precision. In the future, this problem should be resolved in a more elegant way.

- When the folder with histpgrams is selected, one histogram can be selected by double click. Currently the plotting is a little slow, because we started using latex rendering and it seem so slow things down substantially.

- Also, the user should select folder with the datasets (auto_fit.txt and the final_dataset.txt). This was not further worked on but also is not optimal. Again, for the purpose of the thesis, it was enough.

### programs/fitting

- _classical_fitting_ - an attempt to fit the histograms in classical way, was not promising so we stopped.

- _fit_exp_jacquelin.py_ - implementation of Jacquelin fitting method for sum of exponentials - it was refactored several times, but it still has issues. Feel free to be inspired by it, but for personal use it is best to write it from scratch probably. It has an option to estimate errors of the estimates as presented in the text of the thesis - end of Chapter 2.

- _main.py_ starts the Jacquelin fit
- _scanning_method.py_ implementation of the scanning method

- other files often contain helpful functions or are not used anymore
   
- there is generally in many files a lot of currently unused code from studying the fitting possibilites

### programs/graphs_for_thesis
Contains some scripts for generating graphs for the text of the thesis

### programs/integration
Contains scripts for the integrated histogram graphs.

### programs/models
Contains a core of the models of hot electron temperature as well as some other attampts to model it - krigging or kernel_ridge.

The "winning" model - Gaussian processes - was train using the _gp3d_gpy.py_ script.

folder programs/models/models/ contains pickled final models that were used in the thesis and are used by the models_tool UI program.

### programs/models_tool

Contains the implementaion of the UI tool shown in the text of the thesis.

1. The program is started by running main.py.
2. The models are expeted to be in the folder ../models/models/, but it can be chosen in File -> choose folder with models.
3. Selection of the model is a dropdown menu on the right side.
4. Selection of the fixed axis is done just below.
5. Draw predictions button draws the selected region from the model.
6. (If supported) draw uncertainty is triggered by the last button

A support for new model requires the model class to follow an abstrastion:

- static method _load(path)_ for example:

  ```
  @staticmethod
  def load(path):
    return joblib.load(path)
  ```
- member variable _transformer_ of type Transformer defined in the Transformer.py which is transforming the inputs accordingly before the prediction
- method _predict(x)_ which returns the predictions of the model

If the model is satisfying these three conditions, then it can be added in the method _load_selected_model(self)_ in _main.py_. Follow the same naming conventions to avoid further trouble. If you do not want to use _pickle_ to save the model, further changes have to be made, because the current implemntation counts on it.




