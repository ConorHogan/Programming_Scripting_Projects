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
I used the following packages in my analysis. 

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
