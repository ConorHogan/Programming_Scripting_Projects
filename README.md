# 1.0 OVERVIEW




# 2.0 INTRODUCTION TO FISHER'S IRIS DATASET
Fisher's Iris dataset describes the biological characterisitics of three species of iris. The dataset includes 50 examples of each species, with each example having four measured characteristics:
(1) sepal length in cm
(2) sepal width in cm
(3) petal lenght in cm
(4) petal width in cm

The dataset has become a standard used to test new methods of sorting data, for example in machine learning. 
The dataset represents a difficult case for unsupervised analysis i.e. if the species for each example is not know. As I will show in my own analysis, Iris Setosa is a clearly distinct species, whereas Iris Versicolor and Iris Virginica are more difficult to distinguish from eachother for certain without knowing the species name in advance. 

### Sources for this section: 
https://www.techopedia.com/definition/32880/iris-flower-data-set
http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/doc/seminar8.pdf
https://en.wikipedia.org/wiki/Iris_flower_data_set

# 3.0 ANALYSIS OF THE DATASET

## 3.1 INTRODUCTION TO THIS SECTION
In this section I will show the types of  analysis I performed on the dataset and the Python code I used in for each type. Please note the code I have written is all contained within the one "Iris_Analysis.py" file available in the respository. The code is designed to be run in one go and it will output all the analysis outlined below to either the users terminal or as Matplotlib graphs.

## 3.2 PACKAGES USED IN ANALYSIS 
I imported the following packages for in my analysis: 

````python 
#######################################################################
#      IMPORTING & STRUCTURING SECTION     #
#######################################################################
import pandas as pd
import numpy as np
import math as math
import matplotlib.pyplot as plot
import seaborn as graph
from scipy.cluster import hierarchy # for dendrogram
import statistics # for mode
````
### 3.2.1 PANDAS
Panadas allows you to create spreadsheet format dataframes which can then be used to perform analysis on rows and columns in that dataframe, assign headers to columns, and also easily create subsets or pivots of that data. I previously came across Pandas when investigating how to tackle a project in work that I did at the start of this course and looked at it more closely when beginning this project. It quickly become obvious that it would be very useful for the types of analysis I would be trying to perform, especially creating sliced or summarised table to graph.
To get up to speed on how to use Pandas I watched the [Data Analysis with Python and Pandas](https://www.youtube.com/watch?v=Iqjy9UqKKuo&list=PLQVvvaa0QuDc-3szzjeP6N6b0aDrrKyL-) introductory tutorial on YouTube which was very useful. I also used the official [Pandas Documentation](http://pandas.pydata.org/pandas-docs/stable/index.html), but this was a bit more difficult to digest.

### 3.2.2 NUMPY, MATH, & STATISTICS
NumPy and Math were imported for calculating averages, standard deviation, variance etc. I later also added the Statistics package as there is no Mode function in either NumPy or Math.

### 3.2.3 MATPLOTLIB, SEABORN, SCIPY
These three packages were used for graphing results or output of analysis. All the graphs in this project use Seaborn which builds on top of Matplotlib. The SciPy cluster package was imported for use in building the dengrograms in the cluster analysis section of this project. 
Two resources I found extremely useful when building the graphs in this project, were the [The Python Graph Gallery](http://python-graph-gallery.com) and to a lesser extent the [Seaborn official documentation](https://seaborn.pydata.org/).

## 3.3 IMPORTING THE DATASET
