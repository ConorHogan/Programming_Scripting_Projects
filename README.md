# 1.0 OVERVIEW




# 2.0 INTRODUCTION TO FISHER'S IRIS DATASET
Fisher's Iris dataset describes the biological characterisitics of three species of iris. The dataset includes 50 examples of each species, with each example having four measured characteristics:

1. sepal length in cm
2. sepal width in cm
3. petal lenght in cm
4. petal width in cm

The dataset has become a standard used to test new methods of sorting data, for example in machine learning. 
The dataset represents a difficult case for unsupervised analysis i.e. if the species for each example is not know. As I will show in my own analysis, Iris Setosa is a clearly distinct species, whereas Iris Versicolor and Iris Virginica are more difficult to distinguish from eachother for certain without knowing the species name in advance. 

### Sources for this section: 
https://www.techopedia.com/definition/32880/iris-flower-data-set
http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/doc/seminar8.pdf
https://en.wikipedia.org/wiki/Iris_flower_data_set

# 3.0 CODE USED FOR ANALYSIS

## 3.1 INTRODUCTION TO THIS SECTION
In this section I will show the types of analysis I performed on the dataset and the Python code I used in for each type. Please note the code I have written is all contained within the one "Iris_Analysis.py" file available in the respository. The code is designed to be run in one go and it will output all the analysis outlined below to either the users terminal or as Matplotlib graphs.

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

The above code created the "irisdf" dataframe that I would be using as master dataframe for the rest of the analysis. The code first creates a variable storing the link to the dataset file as a URL. It then creates a dataframe by reading the data in the URL. "Header=None" ensures that the top row of the dataset is not counted as header row. If I had not of done this, the top row of the dataset would have been skipped in all my future calculations. I then assigned names to each of the columns in the dataset as these were missing from the original file. I also gave the column header rows the name "Attributes" to help with pivots the data using "stack" later in my analysis. Finally, I set the "Species" column as the dataframes Index as this would be column I would be using to filter, or slice the dataframe in most of my analysis. Setting an index also removes the count column when printing the dataframe. 

I also created three seperate dataframes for each species, which are used later in the Corrolation Analysis. 

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

### 3.4.2 CHECKING COUNTS
To check the total count of rows in the dataset, the number of unique Species, and the count of data samples per species were correct I created the following code. 

````python
species_list = irisdf.index.tolist() # converts index into an array
species_count = len(set(species_list)) # removes duplicates and counts 
count_per_speciesdf = irisdf[["S_Length"]].groupby(irisdf.index).size().reset_index(name="Count") # create new df summarising counts per species
count_per_speciesdf.set_index("Species", inplace=True) # change index column back to species
count_per_species = count_per_speciesdf.iloc[0]["Count"] # get value if first cell in the Count column
total_count = count_per_speciesdf["Count"].sum()
print ("")
print ("")
print (f"There are {species_count} unique species.") #f allows for a variable to be insterted into string
print ("")
print ("")
print (f"As shown below, there are {count_per_species} rows of data samples for each species amounting to {total_count} rows of data in total.")
print ("")
print("##########################")
print("     ROWS PER SPECIES     ")
print("##########################")
print ("")
print (count_per_speciesdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Rowsper_species.png)

The above code first gets a count of the unique species by converting the Index into a list, then getting the list of unique values in that list using the "set()" function, and finally counting the number of these values using "len()". This is used to created the *"species_count"* variable that is inserted into the output text.

The code also creates a new dataframe ("count_per_speciesdf") summarising the count of rows for each species. The "groupby" function allows you to choose a column to summarise the data by, (in this case the Index: "Species") and perform a calculation. Here I used the "size()" function to get the count of the data in the column and then add this data to a column in the new dataframe called "Count". 

I also created two variables; the first to get first value in the "Count" column using "iloc" row plus column name which equates to the count of rows per Species, and the second to get the sum of the "Count" column which equates to the count of rows for all species. 

Finally, I insterted the variables in to the a string to print using the "f-strings" feature and also printed the "count_per_speciesdf" dataframe. 

### 3.4.3 STATISTICS FUNCTIONS
I next moved on to doing some statistical calculation on the dataset. While I count of used the groupby and calculate method I used above to create dataframes summarising the data in different way, I instead opted to create function. I mainly took this approach to use some of what we he learned on the course. The functions I create were used to calculate the range(max/min), median, average, standard deviation, mode, and variance for each of the four "Attribute" columns in the dataset. 

Each of the functions work in roughly the same way:

1. The user enters a dataframe to pass through the function

2. The column names are converted into a list to use a reference

3. To function iterates through each column and performs the calculation converting the result(s) and column name into a tuple(a fixed list).

4. The tuple is coverted to a list. I could not figure out how to print elements from a tuple in a string, so I converted the tuple to a list.

5. Items are taken from the list and insterted into a string which is printed. I used this tutorial (https://www.digitalocean.com/community/tutorials/how-to-use-string-formatters-in-python-3) to learn how to get the text alignment(padding and justification) and formating of floating point numbers to include decimal places. The [PyFormat](https://pyformat.info/) website was also very useful as reference for setting the format of the string output. 

#### 3.4.3.1 MIN AND MAX 
The first function finds the min and max value in each column (ignoring the index) and coverts these values into strings in tuple that are insterted into the output text. There was a bit of trial and error in getting the text padding right.
````python
print("################################")
print("     MIN AND MAX PER COLUMN     ")
print("################################")
print ("")
#CALCULATE MIN MAX OF EACH COLUMN IN DATATFRAME AND PRINT
def maxandmin(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columntuple = str(header), str(min(dataframe[header])), str(max(dataframe[header]))
    columnlist = list(columntuple)
    print("Min & Max for {0:>8}: {1:>6} {2:>6}".format(*columnlist)) # reference https://www.digitalocean.com/community/tutorials/how-to-use-string-formatters-in-python-3
maxandmin(irisdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/MinMax.png)

#### 3.4.3.2 MEDIAN
This function is slightly different in that the Median value is outputed as a floating point number instead of a string. Setting as a string caused the output to go out of alignment for "P_Length" as there was more than one decimal place. Setting as a floating point number allowed me to set the number of decimal places for all values to 2 and keep the correct alignment.
````python
print("###############################")
print("       MEDIAN PER COLUMN       ")
print("###############################")
print ("")
def columnsmedian(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnsmedtuple = str(header), float(dataframe[header].median()) 
    columnsmedlist = list(columnsmedtuple)
    print("Median of {0:>8}: {1:>6.2f}".format(*columnsmedlist))
columnsmedian(irisdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Median.png)

#### 3.4.3.3 MODE
I initially skipped this function as I wasn't sure what benefit knowing the mode for each Attribute would give. When I did come back to it, the formatting proved difficult as there are two mode values for "P_length" (1.4 and 1.5). Setting this as a float resulted in an error because there are two numbers. Setting as a string value worked, but the output formatting is not in line with the other functions. I counldn't find a way to tidy the formatting.  

Mode is not included in the NumPy package so I had to import the Statistics package to perform this calculation.

````python
print("###########################################")
print("             MODE PER COLUMN               ")
print("###########################################")
print ("")
def columnsmode(dataframe):
  columnslist = dataframe.columns.values.tolist()
  for header in columnslist or []:
    columnmodetuple = str(header), str(dataframe[header].mode()) #had to make this a string to work
    columnsmodelist = list(columnmodetuple)
    print("Mode of {0:>8}: {1:>6}".format(*columnsmodelist))
columnsmode(irisdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Mode.png)

#### 3.4.3.4 AVERAGE 
This function calculates the average(mean) for each column/Attribute. As I show later, there is an easier way of getting this data using using Pandas "groupby" function, which used for my barchart showing averages for each Attribute by Species.

````python
print("################################")
print("       AVERAGE PER COLUMN       ")
print("################################")
print ("")
def columnaverage(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnavgtuple = str(header), float(dataframe[header].mean()) # mean includes decimal so must be set as float. Importing the decimal function was causing trouble
    columnavglist = list(columnavgtuple)
    print("Average(mean) of {0:>8}: {1:>6.2f}".format(*columnavglist))
columnaverage(irisdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Average.png)

#### 3.4.3.5 STANDARD DEVIATION
This function calculates the standard deviation for each column. This reuses the same code as the Averages function, but subs in the NumPy ".std()" function.

````python
print("###########################################")
print("       STANDARD DEVIATION PER COLUMN       ")
print("###########################################")
print ("")
def columnstddev(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnstddevtuple = str(header), float(dataframe[header].std()) 
    columnstddevlist = list(columnstddevtuple)
    print("Standard dev of {0:>8}: {1:>6.2f}".format(*columnstddevlist))
columnstddev(irisdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Deviation.png)

#### 3.4.3.6 VARIANCE
This function calculates the variance for each column. Again, this reuses same code as the Averages function, but subs in the NumPy ".var()" function.

````python
print("#################################")
print("       VARIANCE PER COLUMN       ")
print("#################################")
print ("")
def columnsvar(dataframe):
  columnslist = dataframe.columns.values.tolist() 
  for header in columnslist or []:
    columnvartuple = str(header), float(dataframe[header].var()) 
    columnsvarlist = list(columnvartuple)
    print("Variance of {0:>8}: {1:>6.2f}".format(*columnsvarlist))
columnsvar(irisdf)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Variance.png)

## 3.5 CORROLATION ANALYSIS
After completing the above basic analysis, I then moved on to looking at if any of the attributes influenced other attributes. I did this using corrolation analysis.

### 3.5.1 CORROLATION TABLE
The method I used to identify any potential corrolation was creating a corrolation dataframe using the Pandas "dataframe.corr()" function and then printing the dataframe. I also used Pandas ".round()" function to limit the output to two decimal places.

````python
#####################
#CORROLATION TABLE
######################
corrdf = irisdf.corr() # create dataframe for corrolation
print(corrdf.round(2)) #print rounded to two decimal places to match heatmap
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/corrtable.png)

### 3.5.2 CORROLATION HEATMAP
I also referenced [The Python Graph Gallery](https://python-graph-gallery.com/90-heatmaps-with-various-input-format/) to learn how to generate a Seaborn heatmap as an alternative way of illustrating this dataframe. I added annotation to display the values in heatmap after watching the [video tutorial](https://www.youtube.com/watch?v=bA7ZcNmhnTs) referenced in the code below.

In the code below, the "corrdf" dataframe that was generated previously is used as the source dataframe. "Ticklabels" is used to assign the values for each axis. "Annot=True" adds the annotation for each square in the heatmap. cmap= "bwr" sets the colour pallet for the chart which is blue for negative values and red for positive. The colour map was sourced from the Matplotlib colourmap reference page available [here](https://matplotlib.org/examples/color/colormaps_reference.html).

````python
#####################
#CORROLATION HEATMAP
######################
heat = graph.heatmap(corrdf, xticklabels=corrdf.columns,yticklabels=corrdf.columns, annot=True, cmap= "bwr") #https://www.youtube.com/watch?v=bA7ZcNmhnTs showed the annotate trick
plot.show(heat)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/corrheatmap.png)

### 3.5.2 CORROLATION BY SPECIES
I also used the species dataframes I created earlier to create corrolation table for each Species to compare against the data as a whole.

````python
print("####################")
print("#CORROLATION SETOSA#")
print("####################")
corr_setosadf = setosa_df.corr()
print(corr_setosadf.round(2))
print ("")
print ("")
print("########################")
print("#CORROLATION VERSICOLOR#")
print("########################")
corr_verdf = versicolor_df.corr()
print(corr_verdf.round(2))
print ("")
print ("")
print("#######################")
print("#CORROLATION VIRGINICA#")
print("#######################")
corr_virgdf = virginica_df.corr()
print(corr_virgdf.round(2))
````

**Output:**

SETOSA                     |  VERSICOLOR               |  VIRGINICA
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Setosacorr.png)  |  ![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Versicorr.png)   |  ![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Virgincorr.png)


### 3.5.4 CORROLATION PAIRPLOT
While looking at The Python Graph Gallery I also noticed the [Corrolation Matrix](https://python-graph-gallery.com/110-basic-correlation-matrix-with-seaborn/) which is a Seaborn Pairplot graph. To add a "hue" (reference colour scheme) that shows different species, I removed the index from the dataframe to allow me to set the "Species" column as a reference. Try to set the "df.index" as a reference for the "hue" returned an error.

````python
#https://python-graph-gallery.com/110-basic-correlation-matrix-with-seaborn/ & https://python-graph-gallery.com/111-custom-correlogram/
pairplotdf = irisdf.reset_index() # had to remove index to get the "hue" to pick up the seperate species. Didn't work with df.index
corr_pairplot = graph.pairplot(pairplotdf, kind="scatter", hue="Species")
plot.show(corr_pairplot)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/corrplot.png)

## 3.6 COMPARING / CONTRASTING SPECIES
From my initial research into the dataset, I learned that Iris-setosa is easy to distinguish from the the other two species, while the Versicolor and Virginica are very similar. To illustrate this, I generated a bar chart showing average values for each Attibute grouped by Species. I also generate a Seaborn [Swarmplot](https://seaborn.pydata.org/generated/seaborn.swarmplot.html) to show difference between each Species for each Attribute.

### 3.6.1 AVERAGES COMPARISON
To create the Average barchart, I first restructured the master "irisdf" dataframe into a format that suited the graph I wanted to generate (Column 1 = Species, Column 2 = Attributes, Column 3 = Average per Attribute per Species). I first played around with data in Excel and generated the graph there to help me visualise what I needed to do. When I began researching how to pivot and perform a calculation on the data I saw references to the Pandas "stack()" function. While I didn't generally find official Pandas documentation very helpful in this project, it's page on [Reshaping and Pivot Tables](https://pandas.pydata.org/pandas-docs/stable/reshaping.html) was pretty clear and understable. To allow for using the "stack" function I had to go back to the "Importing & Strucuring" section of my code to add name to the "Attribute" row of the dataframe so I could be converted into a column in the new "iris_stackdf" dataframe. I also named the new column created from all the Attribute values "Measures". Finally, I converted this dataframe into another new dataframe tha summarised the "Measures" values into averages using the "grouby" function I had used to summarise by the count of rows earlier in my analysis.

To show the chart the data seperated out by "Species" I used the Seaborn [Factorplot](https://seaborn.pydata.org/generated/seaborn.factorplot.html) with the graph "kind" set to "bar". I then set the "hue" colour scheme clearly show the difference between petal width, petal length, sepal width, and sepal length.

````python
# CREATING AVERAGES STACKED DATAFRAME
iris_stackv = irisdf.stack() # restructure dataframe using stack
iris_stackdf = pd.DataFrame(iris_stackv, columns= ["Measures"])
iris_stackavgdf = iris_stackdf.groupby(['Species', 'Attributes']).mean().reset_index() # remove index to make axis easier to assign

#AVERAGES COMPARISON
grouped_avgs_graph = graph.factorplot(x='Species', y='Measures', hue='Attributes', data=iris_stackavgdf, kind='bar')
plot.show(grouped_avgs_graph)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/averagesbarchart.png)

### 3.6.2 ATTRIBUTES SWARMPLOT
While browsing the Seaborn examples gallery I saw an example of a [Swarmplot](https://seaborn.pydata.org/examples/scatterplot_categorical.html) using Fisher's Iris Dataset. I was very clear illustration of how the Setosa species differed from Versicolor and Viriginica so I included my own version in my analysis. 

To chart the data I reused the stacked dataframe I had created in the first step for the Averages barchart above, but removed the indexes to make it easier to set the graphs parameters as using df.index as parameter always returned an error when plotting graphs.

````python
#SWARM DISTRIBUTION
iris_stackswarm = iris_stackdf.reset_index()
swarm_by_attr_graph = graph.swarmplot(x="Attributes", y="Measures", hue="Species", data=iris_stackswarm)
plot.show(swarm_by_attr_graph)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/scatterplot.png)

## 3.7 CLUSTER ANALYSIS
While researching the Iris dataset, I came found a [project analysing the data using the R programming language](https://cran.r-project.org/web/packages/dendextend/vignettes/Cluster_Analysis.html). This encluded some examples of dendrograms, which felt like a very appropriate way of representing the data since it concerns species and biological classification. This also peaked my interest because in the [wikipedia article](https://en.wikipedia.org/wiki/Iris_flower_data_set) on the Iris data set, it had also mention the this dataset was a difficult case for cluster analysis when the species are not know in advance. It was also another useful way of illustrating that the Setosa species has very distinct charecteristics / attributes while Viriginica and Versicolor are similar to a degree that they are sorted into the same clusters. 

### 3.7.1 DENDROGRAM
I once again used [The Python Graph Gallery](https://python-graph-gallery.com/400-basic-dendrogram/) as a reference to learn how to generate this chart. I first edited the dataframe by removing the index header and then the column headers using [this method](https://stackoverflow.com/questions/44917675/pandas-delete-column-name) described on Stack Overflow. In order to generate the dendrogram, I added SciPy's clustering package to my list of packages to import in the "Importing & Structuring" section at the start of my code. It is also necessary to set a linkage type in order to specify the distance between clusters (see this [blog](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/) I consulted). I also generated the graph using "single", "complete", and "average" linkage, but the default "ward" linkage seemed to be getting the best results. I did briefly investigate what the difference between the linkage types were, but I don't come from a statistics background and found it very difficult to follow. 

When generating the dendrogram I switched the orientation so the species name for each datapoint label would be easier to read and changes the labels from a the datapoint number to the species name. The method for doing both these steps was taken from the [Customised Dendrogram](https://python-graph-gallery.com/401-customised-dendrogram/#tab-3) tutorial on The Python Graph Gallery site. 

````python
#####################
#DENDROGRAM
###################### 
# see https://python-graph-gallery.com/400-basic-dendrogram/  
del irisdf.index.name
irisdf.columns = [''] * len(irisdf.columns) #remove column names
Z = hierarchy.linkage(irisdf, 'ward') # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ explain linkage
dendrogram_chart = hierarchy.dendrogram(Z, orientation="left", labels=irisdf.index, color_threshold=10, above_threshold_color="grey")
plot.show(dendrogram_chart)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/irisdendrogram.png)

### 3.7.2 CLUSTERMAP
After finishing my dendrogram, I continued looking through the Python Graph Gallery. It stated the when using a dendrogram to display the results of a cluster analyis it is best practice to also add a heatmap of the information (see the [Dendrogram with heat map](https://python-graph-gallery.com/404-dendrogram-with-heat-map/) page of the Python Graph Gallery.

I particularly liked the [Dendrogram with Coloured Leaves](https://python-graph-gallery.com/405-dendrogram-with-heatmap-and-coloured-leaves/) example as it would clearly show the overlap between the Versicolor and Virginica species. 

To generate the "colour leaves" section of the graph I did the following steps using a [post on Stack Overflow as a guide](https://stackoverflow.com/questions/47292737/change-the-color-for-ytick-labels-in-seaborn-clustermap):

1. Restored the index name "Species" using "df.index.name" and restored column names

2. Turned of the index setting in order to allow me to "pop" the column in the next step

3. Used "df.pop" to cut out the "Species" column.

4. Created a dictionary of the 3 unique values in the "Species" column and and assigned the "rbg" colour scheme to this list.

For the clustermap itself, I again set "ward" as the linkage method, set the colour scheme to "mako". Finally, I again borrowed code from the Stack Overflow example referenced above to make the datalables on the right hand side match the colour of the "Species Leaves" section. 

````python
#####################
#CLUSTERMAP
###################### 
#http://seaborn.pydata.org/generated/seaborn.clustermap.html - documentation for this one is really good, lots of examples
irisdf.index.name = "Species" # restore index names
irisdf.columns = ["S_Length","S_Width","P_Length","P_Width"] # restore column names
clusterdf = irisdf.reset_index() # strip index again
species = clusterdf.pop("Species") # cuts Species column
lut = dict(zip(species.unique(), "rbg")) # assigns values to unique species
row_colors = species.map(lut) # finds species names in graph and assigns colours
clustermap = graph.clustermap(clusterdf, method="ward", cmap="mako", figsize=(10, 15), row_colors=row_colors)
for tick_label in clustermap.ax_heatmap.axes.get_yticklabels(): # make colour of labels match Species row colour https://stackoverflow.com/questions/47292737/change-the-color-for-ytick-labels-in-seaborn-clustermap
    tick_text = tick_label.get_text()
    species_name = species.loc[int(tick_text)]
    tick_label.set_color(lut[species_name])
plot.show(clustermap)
````

**Output:**

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/clustermap.png)

# 4.O SUMMARY OF INVESTIGATIONS
In this section I will discuss what I observed about the Iris dataset, when reviewing the output of the code I have discussed above.

## 4.1 STATISICAL FUNCTIONS

### 4.1.1 MIN AND MAX (RANGE)

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/MinMax.png)

As you can see from the above figures, the range for Petal Length and Petal Width is relatively large compared to the two Sepeal characteristics. As we will see later, this is due to the Setosa species having a significantly smaller Petal size in comparison to the other two Species.

### 4.1.2 MEDIAN & MODE 

MEDIAN                     |  MODE
:-------------------------:|:-------------------------:
![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Median.png)  |  ![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Mode.png)

Looking at the relatively high Median values for Petal Length and Petal Width in comparison to the minimum values observed above is further evidence that there is a minority of that with small petals. Also, the low Mode for the petal characteristics in comparison to Median and Average is an indicator that there may be a minority of the dataset that is disproportionately responsible keeping these figures low.

### 4.1.3 AVERAGE, STANDARD DEVIATION & VARIANCE

AVERAGE                    |  STANDARD DEVIATION       |  VARIANCE
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Average.png)  |  ![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Deviation.png)   |  ![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Variance.png)

The Averages don't tell us much on their own, but acts a baseline for comparison with the other outputs. For example, looking at the Standard Deviation and Variance for the two Petal characterisitcs we can see that values are very high in comparison to the Average values. This again suggests a large distribution of values in the samples for these two characterisitics and suggest that we are dealing with species that look very different from each other. 


## 4.2 CORROLATION ANALYSIS

![alt text](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/corrheatmap.png)

The Corrolation heatmap gives us some interesting results. Based on the data as a whole we can observe very strong positive corrolation between:

1. Sepal Length & Petal Width (0.82)
2. Sepal Lenght & Petal Length (0.87)
3. Petal Length & Petal Width (0.92) 

This above tells us that as one of these attributes gets higher so does the other. However, Sepal Width acts in opposite manner. The bigger the Iris gets, the smaller the Sepal Width is in comparison. 

However, this is actually misleading. If you look at all three species seperately below, the only strong corrolation (0.8+) in any of the species is Sepal Length & Petal Length in Iris Virginica and there is no negative corrolation between characterisitics within species. As would be expected, corrolation in the Virgincia and Versicolor species are more similar, but Setosa and Virginica have an exactly matching corrolation figures (0.28) for Sepal Length and Petal Width. 

SETOSA                     |  VERSICOLOR               |  VIRGINICA
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Setosacorr.png)  |  ![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Versicorr.png)   |  ![](https://github.com/ConorHogan/Programming_Scripting_Projects/blob/master/Images/Virgincorr.png)

In summary, there is a no significant corrolation between characteristics when viewing the species on an individual basis, but there may be some corrolation for Irises in general.


## COMPARATIVE ANALYSIS



## CLUSTER ANALYSIS
