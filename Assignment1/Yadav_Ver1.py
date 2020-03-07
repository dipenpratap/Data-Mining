#!/usr/bin/env python
# coding: utf-8

# # Part 1


'''
1. Identify each column as nominal, ordinal, interval, or ratio in the Auto_mpg_raw.csv data set.
a.	Miles per gallon: ratio
b.	Cylinders: ratio
c.	Displacement: nominal
d.	Horsepower: ratio
e.	Weight: ratio
f.	Acceleration: nominal
g.	Model Year: interval
h.	Origin: nominal
i.	Car Name: nominal

'''

# # Part 2


# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)

# importing the Auto_mpg_raw.csv file using pandas
auto_Mpg = pd.read_csv("Auto_mpg_raw.csv")

# Plotting all the histograms
# histogram for miles per gallon
plt.figure()
plt.hist(auto_Mpg["Miles per gallon"])
plt.title("Miles per gallon")

# histogram for Cylinders
plt.figure()
plt.hist(auto_Mpg["Cylinders"])
plt.title("Cylinders")

# histogram for Displacement
plt.figure()
plt.hist(auto_Mpg["Displacement"])
plt.title("Displacement")

# histogram for Horsepower
plt.figure()
plt.hist(auto_Mpg["Horsepower"])
plt.title("Horsepower")

# histogram for Weight
plt.figure()
plt.hist(auto_Mpg["Weight"])
plt.title("Weight")

# histogram for Acceleration
plt.figure()
plt.hist(auto_Mpg["Acceleration"])
plt.title("Acceleration")

# histogram for Model Year
plt.figure()
plt.hist(auto_Mpg["Model year"])
plt.title("Model year")

# histogram for origin
plt.figure()
plt.hist(auto_Mpg["Origin"])
plt.title("Origin")

""" Answer to question number 2
 Here from seeing the distribution in the histograms, some misleading datapoints are observed.
    The misleading data points are discussed below:
    1. The first attribute Miles_per_gallon has an extreme value 1000 that is an outlier in the distribution
        and definately is a misleading data that needs some adjustment. Summary of Miles per gallon:
        
count     406.000000
mean       42.755665
std       136.102120
min         9.000000
25%        17.500000
50%        23.000000
75%        29.800000
max      1000.000000
Name: Miles per gallon, dtype: float64




    2. The second attribute that requires attention is the "Horsepower". Since horsepower is a ratio,
        0 horsepower means absence of horsepower that does not make sense in the real world. Therefore, 
        0's need some other relevant values.
        
count    405.000000
mean     103.785185
std       40.241668
min        0.000000
25%       75.000000
50%       94.000000
75%      129.000000
max      230.000000
Name: Horsepower, dtype: float64

    
    Other than those two attributes, the rest of the attributes does not have misleading datapoints. 
        """

# Lets change all the values equal to 1000 for miles_Per_gallon to NaN
auto_Mpg["Miles per gallon"] = auto_Mpg["Miles per gallon"].replace({1000: np.NaN})
print(f" The new distribution of Miles Per Gallon is: ")
auto_Mpg["Miles per gallon"].describe()

# Again, lets change all the values equal to 0 for horsepower to NaN
auto_Mpg["Horsepower"] = auto_Mpg["Horsepower"].replace({0: np.NaN})
print(f" The new distribution of Horsepower is: ")
auto_Mpg["Horsepower"].describe()

# Now lets create a table with only three attributes (miles_per_gallon, Cylinders, Horsepower)
new_Table = auto_Mpg[["Miles per gallon", "Cylinders", "Horsepower"]]
new_Table.describe()

# Now lets find an appropriate value to replace the NaN's in the miles_per_gallon column
""" One approach is to find cylinders for each cars is to find a mean to replace those NaNs.
    We are going to take this approach to fill in the blanks or NaNs."""

# All rows with NaN mileage

new_Table[new_Table['Miles per gallon'].isnull()]

# All rows with NaN horsepowers
new_Table[new_Table['Horsepower'].isnull()]


# In[8]:


# Calculating the mean mileage for different cars 
def meanOfMileage(numOfCylinders):
    num_Cylinder_Cars = new_Table[new_Table.Cylinders == numOfCylinders]
    meanOfMileage = (num_Cylinder_Cars['Miles per gallon']).mean()
    return meanOfMileage


# Function to calculate the mean of horsepowers for different cars
def meanOfHorsepower(numOfCylinders):
    num_Cylinder_Cars = new_Table[new_Table.Cylinders == numOfCylinders]
    meanOfHorsepower = (num_Cylinder_Cars['Horsepower']).mean()
    return meanOfHorsepower


# Function to update the horsepowers
def horsepowerUpdater(numOfCylinders):
    num_Cylinder_Cars = new_Table[new_Table.Cylinders == numOfCylinders]
    NaNHorsepower = num_Cylinder_Cars[num_Cylinder_Cars['Horsepower'].isnull()]
    NaNHorsepower['Horsepower'] = meanOfHorsepower(numOfCylinders)
    return NaNHorsepower


# function to update the mileage
def mileageUpdater(numOfCylinders):
    num_Cylinder_Cars = new_Table[new_Table.Cylinders == numOfCylinders]
    NaNMileage = num_Cylinder_Cars[num_Cylinder_Cars['Miles per gallon'].isnull()]
    NaNMileage['Miles per gallon'] = meanOfMileage(numOfCylinders)
    return NaNMileage


updatedHorsepower4 = horsepowerUpdater(4)
updatedHorsepower6 = horsepowerUpdater(6)
updatedHorsepower8 = horsepowerUpdater(8)
updatedHorsepower = pd.concat([updatedHorsepower4, updatedHorsepower6, updatedHorsepower8])
updatedHorsepower

# In[9]:


updatedMileage4 = mileageUpdater(4)
updatedMileage6 = mileageUpdater(6)
updatedMileage8 = mileageUpdater(8)
updatedMileage = pd.concat([updatedMileage4, updatedMileage6, updatedMileage8])
updatedMileage

# In[10]:


# Add two dataframes together
updatedVals = pd.concat([updatedMileage, updatedHorsepower])
print("The dataframe with updated values are: ")
updatedVals

# In[11]:


# Updating the original dataset with the interpolated values
auto_Mpg.update(updatedVals, join='left', overwrite=True, filter_func=None, errors='ignore')
print("The new dataset is: ")
auto_Mpg.head()

# In[12]:


# Plotting histograms with cleaned dataset
# Plotting all the histograms
# histogram for miles per gallon
plt.figure()
plt.hist(auto_Mpg["Miles per gallon"])
plt.title("Miles per gallon")

# histogram for Cylinders
plt.figure()
plt.hist(auto_Mpg["Cylinders"])
plt.title("Cylinders")

# histogram for Displacement
plt.figure()
plt.hist(auto_Mpg["Displacement"])
plt.title("Displacement")

# histogram for Horsepower
plt.figure()
plt.hist(auto_Mpg["Horsepower"])
plt.title("Horsepower")

# histogram for Weight
plt.figure()
plt.hist(auto_Mpg["Weight"])
plt.title("Weight")

# histogram for Acceleration
plt.figure()
plt.hist(auto_Mpg["Acceleration"])
plt.title("Acceleration")

# histogram for Model Year
plt.figure()
plt.hist(auto_Mpg["Model year"])
plt.title("Model year")

# histogram for origin
plt.figure()
plt.hist(auto_Mpg["Origin"])
plt.title("Origin")

# In[13]:


# Exporting the file as a .csv to local folder
auto_Mpg.to_csv("Auto_mpg_adjust.csv", index=False)

# ## Part 3

# In[14]:


# Problem 3
""" Using z values to check for outliers in the dataset """
from scipy import stats

auto_Mpg['milesPerGallon_Z'] = stats.zscore(auto_Mpg["Miles per gallon"])
# print(auto_Mpg['Miles per gallon_Z'])

# Looking for outliers in the dataset using z values
MPGOutliers = auto_Mpg.query('milesPerGallon_Z >3 | milesPerGallon_Z <-3')
numberOfMPGOutliers = len(MPGOutliers)
print(f'There are %d outliers present in MPG data.' % numberOfMPGOutliers)

# In[15]:


# Looking into Cylinder data
auto_Mpg['cylinders_Z'] = stats.zscore(auto_Mpg["Cylinders"])
# print(auto_Mpg['cylinders_Z'])
cylindersOutliers = auto_Mpg.query('cylinders_Z >3 | cylinders_Z <-3')
numberOfCylindersOutliers = len(cylindersOutliers)
print(f'There are %d outliers present in Cylinders data.' % numberOfCylindersOutliers)

# In[16]:


# Looking into Displacement data
auto_Mpg['displacement_Z'] = stats.zscore(auto_Mpg["Displacement"])
# print(auto_Mpg['displacement_Z'])
displacementOutliers = auto_Mpg.query('displacement_Z >3 | displacement_Z <-3')
numberOfDisplacementOutliers = len(displacementOutliers)
print(f'There are %d outliers present in Displacements data.' % numberOfDisplacementOutliers)

# In[17]:


# Looking into Horsepower data
auto_Mpg['horsepower_Z'] = stats.zscore(auto_Mpg["Horsepower"])
# print(auto_Mpg['horsepower_Z'])
horsepowerOutliers = auto_Mpg.query('horsepower_Z >3 | horsepower_Z <-3')
numberOfHorsepowerOutliers = len(horsepowerOutliers)
print(f'There are %d outliers present in Horsepower data.' % numberOfHorsepowerOutliers)
horsepowerSort = auto_Mpg.sort_values('horsepower_Z', ascending=False)
horsepowerSort.head(n=4)

# In[18]:


# Looking into Weight data
auto_Mpg['weight_Z'] = stats.zscore(auto_Mpg["Weight"])
# print(auto_Mpg['weight_Z'])
weightOutliers = auto_Mpg.query('weight_Z >3 | weight_Z <-3')
numberOfWeightOutliers = len(weightOutliers)
print(f'There are %d outliers present in Weight data.' % numberOfWeightOutliers)

# In[19]:


# Looking into Acceleration data
auto_Mpg['acceleration_Z'] = stats.zscore(auto_Mpg["Acceleration"])
# print(auto_Mpg['acceleration_Z'])
accelerationOutliers = auto_Mpg.query('acceleration_Z >3 | acceleration_Z <-3')
numberOfAccelerationOutliers = len(accelerationOutliers)
print(f'There are %d outliers present in Acceleration data.' % numberOfAccelerationOutliers)
accelerationSort = auto_Mpg.sort_values('acceleration_Z', ascending=False)
accelerationSort.head(n=2)

# In[20]:


# Looking into modelYear data
auto_Mpg['modelYear_Z'] = stats.zscore(auto_Mpg["Model year"])
# print(auto_Mpg['modelYear_Z'])
modelYearOutliers = auto_Mpg.query('modelYear_Z >3 | modelYear_Z <-3')
numberOfModelYearOutliers = len(modelYearOutliers)
print(f'There are %d outliers present in Model year data.' % numberOfModelYearOutliers)

# In[21]:


# Looking into Origin data
auto_Mpg['origin_Z'] = stats.zscore(auto_Mpg["Origin"])
# print(auto_Mpg['origin_Z'])
originOutliers = auto_Mpg.query('origin_Z >3 | origin_Z <-3')
numberOfOriginOutliers = len(originOutliers)
print(f'There are %d outliers present in Origin data.' % numberOfOriginOutliers)

# ## Part 4

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)

cereals = pd.read_csv("cereals.csv")
cereals.head()

# ### Part a

# In[23]:


# create a bar graph and normalized bar graph of the “Manuf” variable with “Type” overlay
crosstab_01 = pd.crosstab(cereals["Manuf"], cereals['Type'])
plt.figure()
crosstab_01.plot(kind='bar', stacked=True)
plt.title('Bar Graph of Manuf with Type Overlay')

# Normalized bar graph 
plt.figure()
crosstab_norm = crosstab_01.div(crosstab_01.sum(1), axis=0)
crosstab_norm.plot(kind='bar', stacked=True)
plt.title('Normalized Bar Graph of Manuf with Type Overlay')

# ### Part b

# In[24]:


# create a contingency table of “Manuf” and “Type”

crosstab_02 = pd.crosstab(cereals["Type"], cereals['Manuf'])
crosstab_norm_02 = round(crosstab_02.div(crosstab_02.sum(0), axis=1) * 100, 1)
crosstab_02

# In[25]:


crosstab_norm_02

# ### Part c

# In[26]:


# create a histogram and normalized histogram of “Calories” with “Manuf” overlay
crosstab_03 = pd.crosstab(cereals["Calories"], cereals['Manuf'])
plt.figure()
crosstab_03.plot(kind='bar', stacked=True)
plt.title('Bar Graph of Calories with Manuf Overlay')

# Normalized bar graph
plt.figure()
crosstab_03_norm = crosstab_03.div(crosstab_03.sum(1), axis=0)
crosstab_03_norm.plot(kind='bar', stacked=True)
plt.title('Normalized Bar Graph of Calories with Manuf Overlay')

# Stacked histogram 
plt.figure()
cerealCaloriesManuf_A = cereals[cereals.Manuf == 'A ']['Calories']
cerealCaloriesManuf_G = cereals[cereals.Manuf == "G "]['Calories']
cerealCaloriesManuf_K = cereals[cereals.Manuf == "K "]['Calories']
cerealCaloriesManuf_N = cereals[cereals.Manuf == "N "]['Calories']
cerealCaloriesManuf_P = cereals[cereals.Manuf == "P "]['Calories']
cerealCaloriesManuf_Q = cereals[cereals.Manuf == "Q "]['Calories']
cerealCaloriesManuf_R = cereals[cereals.Manuf == "R "]['Calories']
plt.hist([cerealCaloriesManuf_A, cerealCaloriesManuf_G, cerealCaloriesManuf_K,
          cerealCaloriesManuf_N, cerealCaloriesManuf_P, cerealCaloriesManuf_Q,
          cerealCaloriesManuf_R], bins=10, stacked=True)
plt.legend(['Manuf A', 'Manuf G', 'Manuf K', 'Manuf N', 'Manuf P', 'Manuf Q', 'Manuf R'])
plt.title('Histogram of Calories with Manuf Overlay')
plt.ylabel('Frequency')
plt.xlabel('Calories')

# Normalized histogram of Calories with Manuf Overlay
(n, bins, patches) = plt.hist([cerealCaloriesManuf_A, cerealCaloriesManuf_G, cerealCaloriesManuf_K,
                               cerealCaloriesManuf_N, cerealCaloriesManuf_P, cerealCaloriesManuf_Q,
                               cerealCaloriesManuf_R], bins=10, stacked=True)

# Creating a new plot
plt.figure()
n_table = np.column_stack((n[0], n[1]))
n_norm = n_table / n_table.sum(axis=1)[:, None]

# creating an array of bin cuts
our_bins = np.column_stack((bins[0:10], bins[1:11]))
p1 = plt.bar(x=our_bins[:, 0], height=n_norm[:, 0], width=our_bins[:, 1] - our_bins[:, 0])
p2 = plt.bar(x=our_bins[:, 0], height=n_norm[:, 1], width=our_bins[:, 1] - our_bins[:, 0], bottom=n_norm[:, 0])

''' I could not figure this out. '''

# ### Part d

# In[27]:


# Bin the “Calories” variable using bins for 0 to 90, 91-110, and over 110 calories and ...
# create a bar chart of the binned calories variable with “Manuf” overlay
# Binning the calories
cereals["caloriesBinned"] = pd.cut(x=cereals['Calories'], bins=[0, 91, 111, 165],
                                   labels=['0-90', '90-110', 'Over 110'], right=False)
crosstab_04 = pd.crosstab(cereals['caloriesBinned'], cereals['Manuf'])
crosstab_04.plot(kind='bar', stacked=True, title='Bar chart of Binned Caloris with Manuf Overlay')

# importing sequential model
from keras.models import Sequential

model = Sequential()

# stacking layers
from keras.layers import Dense

model.add(Dense(units=10, activation='relu', input_dim=5))
model.add(Dense(units=20, activation='sigmoid', input_dim=10))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# train the model (Fit the model with training data)
model.fit(x_train, y_train, epochs=5, batch_size=10)

# Evaluating the model
loss_and_metrics = model.evaluate(x_test, y_test, batch_sie=10)
# Predicting on new data
classes = model.predict(x_test, batch_size=10)
