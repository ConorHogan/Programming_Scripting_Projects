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
These three packages were used for graphing results or output of analysis. All the graphs in this project use Seaborn which builds on top of Matplotlib. The SciPy cluster package was imported for use in building the dendrograms in the cluster analysis section of this project. 
Two resources I found extremely useful when building the graphs in this project, were the [The Python Graph Gallery](http://python-graph-gallery.com) and to a lesser extent the [Seaborn official documentation](https://seaborn.pydata.org/).

## 3.3 IMPORTING THE DATASET
The source I used for the Iris dataset was the .CSV file available on the UC Irvine Machine Learning Reporistory Site (http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).
The initial plan was to import my data directly from here, but the website notes that there are two mistakes in the dataset at row 35 and 38. I therefore decided to instead copy the data into a .CSV file, make the two correction outlined on the site and save the .CSV file in this repository [here](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Iris_Data.csv). 
I then used the Pandas package to read the ammended dataset directly from the repositroy into a Pandas dataframe. See code below:

````python
url = 'https://raw.githubusercontent.com/ConorHogan/Programming_Scripting_Projects/master/Iris_Data.csv'
irisdf = pd.read_csv(url, header=None) 
irisdf.columns = ["S_Length","S_Width","P_Length","P_Width","Species"]
irisdf.columns.name = "Attributes" # added this to help with stacking https://www.youtube.com/watch?v=reTeOfEebeA
irisdf.set_index("Species", inplace=True)

##CREATE 3 DATA SUBSETS FOR EACH IRIS SPECIES
setosa_df = irisdf[irisdf.index == "Iris-setosa"]
versicolor_df = irisdf[irisdf.index == "Iris-versicolor"]
virginica_df = irisdf[irisdf.index == "Iris-virginica"]
````

The above code created the "irisdf" dataframe that I would be using as master dataframe for the rest of the analysis. 
-The code first creates a variable storing the link to the dataset file as a URL. 
-It then creates a dataframe by reading the data in the URL. 
-"Header=None" ensures that the top row of the dataset is not counted as header row. If I had not of done this, the top row of the dataset would have been skipped in all my future calculations. 
-I then assigned names to each of the columns in the dataset as these were missing from the original file. I also gave the column header rows the name "Attributes" to help with pivots the data using "stack" later in my analysis. 
-Finally, I set the "Species" column as the dataframes Index as this would be column I would be using to filter, or slice the dataframe in most of my analysis. Setting an index also removes the count column when printing the dataframe. 

I also created three seperate dataframes for each species, but in the end the were not required. 

## 3.4 BASIC ANALYSIS

### 3.4.1  PRINTING THE HEAD 
The first thing I did when examing the new dataframe was to us the "dataframe.head()" function to print the first 5 rows of the dataset and check my index and column headers had been saved correctly. 

````python
print("##########################")
print("       SAMPLE DATA        ")
print("##########################")
print ("")
print (irisdf.head())
print ("")
````
**Output:**
![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Dataframe_head.png)

Using the index function had moved the "Species" column for the rightmost column to the leftmost column and the column headers has been assigned correctly.

### CHECKING COUNTS
I then wanted to 






