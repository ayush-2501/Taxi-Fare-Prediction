# Taxi-Fare-Prediction
Predicting the fare amount for a taxi ride in New York City

Kaggle Test Dataset Score - 2.97

## Business Understanding

It is important to understand the idea of business behind the data set. The given data set is asking us to
predict fare amount.It really becomes important for us to predict the fare amount.Thus, we have to concentrate on making the model
most efficient.

## Data Understanding

To get the best results. Here, the given train data is a CSV file that consists 7 variables and 5000000 Observation. A snapshot of the data
provided.

![image](https://user-images.githubusercontent.com/67412893/114295901-e5d35800-9ac5-11eb-89fb-21c8ebc3a162.png)

The different variables of the data are:
fare_amount : fare of the given cab ride.
pickup_datetime : timestamp value explaining the time of ride start.
pickup_longitude : a float value explaining longitude location of the ride start.
pickup_latitude : a float value explaining latitude location of the ride start.
dropoff_longitude: a float value explaining longitude location of the ride end.
dropoff_latitude : a float value explaining latitude location of the ride end
passenger_count : an integer indicating the number of passengers

## Data Preparation

Missing Value Analysis
Missing value is availability of incomplete observations in the dataset. This is found because of reasons
like, incomplete submission, wrong input, manual error etc.These Missing values affect the accuracy of
model. So, it becomes important to check missing values in our given data.

### Missing Value Analysis in Given Data:

In the given dataset it is found that there are lot of values which are missing. It is found in the following
types:
1. Blank spaces : Which are converted to NA and NaN in R and Python respectively for
further operations
2. Zero Values : This is also converted to NA and Nan in R and python respectively prior
further operations
3. Repeating Values : there are lots of repeating values in pickup_longitude,
pickup_latitude, dropoff_longitude and dropoff_latitude. This will hamper our model, so such
data is also removed to improve the performance. 

![image](https://user-images.githubusercontent.com/67412893/114296152-2f707280-9ac7-11eb-94e9-14ca9b9a7bb3.png)

### Impute the missing value:

After the Identification of the missing values the next step is to impute the missing values. And this
imputation is normally done by following methods.

1. Central Tendencies: by the help of Mean, Median or Mode
2. Distance based or Data mining method like KNN imputation
3. Prediction Based: It is based on Predictive Machine Learning Algorithm

To use the best method it is necessary for us to check, which method predicts values close to the original
data. And this done by taking a subset of data, taking an example variable and noting down its original
value and the replacing that value with NA and then applying available methods. And noting down every
value from the above methods for the example variable we have taken, now we chose the method which
gives most close value.
In this project, I am droping all missing values missing Values. 

### Outlier Analysis:

Outlier is an abnormal observation that stands or deviates away from other observations. These happens
because of manual error, poor quality of data and it is correct but exceptional data. But, It can cause an
error in predicting the target variables. So we have to check for outliers in our data set and also remove
or replace the outliers wherever required
Outliers in this project.
In this dataset, I have found some irregular data, those are considered as outliers. These are explained
below.

1) Fare_Amount :
I have always seen fare of a cab ride as positive, I have never seen any cab driver, giving me money to take
a ride in his cab. But in this dataset, there are many instances where fare amount is negative. Keeping a Threshold I'll remove them:

Python:

df['fare_amount'] = df['fare_amount'][(df['fare_amount'] > 0) & (df['fare_amount'] <= 250)]

2) Passenger_count:
I have always found a cab with 4 seats to maximum of 6 seats. But in this dataset I have found passenger
count more than this, and in some cases a large number of values. This seems irregular data, or a manual
error. Thus, these are outliers and needs to be removed.

Python:

df['passenger_count'] = df['passenger_count'][(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

3) Location points:
When I checked the data it is found that most of the longitude points are within the 70 degree and most
of the latitude points are within the 40 degree. This symbolizes all the data belongs to a specific location
and a specific range. But I also found some data which consists location points too far from the average
location point’s range of 70 Degree Longitude and 40 Degree latitude. It seems these far point locations
are irregular data. And I consider this as outlier.

Python:

df['pickup_longitude']= df['pickup_longitude'].replace({0:np.nan})
df['pickup_latitude']= df['pickup_latitude'].replace({0:np.nan})
df['dropoff_longitude']= df['dropoff_longitude'].replace({0:np.nan})
df['dropoff_latitude']= df['dropoff_latitude'].replace({0:np.nan})

df['pickup_latitude'] = df['pickup_latitude'][(df['pickup_latitude'] >= 40) & (df['pickup_latitude'] <= 42)]
df['pickup_longitude'] = df['pickup_longitude'][(df['pickup_longitude'] <= -72) & (df['pickup_longitude'] >= -75)]
df['dropoff_latitude'] = df['dropoff_latitude'][(df['dropoff_latitude'] >= 40) & (df['dropoff_latitude'] <= 42)]
df['dropoff_longitude'] = df['dropoff_longitude'][(df['dropoff_longitude'] <= -72) & (df['dropoff_longitude'] >= -75)]

## Feature Extraction

## Feature Selection

Sometimes it happens that, all the variables in our data may not be accurate enough to predict the target
variable, in such cases we need to analyze our data, understand our data and select the dataset variables
that can be most useful for our model. In such cases we follow feature selection. Feature selection helps
by reducing time for computation of model and also reduces the complexity of the model.
After understanding the data, preprocessing and selecting specific features, there is a process to engineer
new variables if required to improve the accuracy of the model.
In this project the data contains only the pick up and drop points in longitude and latitude. The
fare_amount will mailnly depend on the distance covered between these two points. Thus, we have to
create a new variable prior further processing the data. And in this project the variable I have created is
Distance variable (dist), which is a numeric value and explains the distance covered between the pick up
and drop of points. After researching I found a formula called The haversine formula, that determines the
distance between two points on a sphere based on their given longitudes and latitudes. These formula
calculates the shortest distance between two points in a sphere.
The function of haversine function is described, which helped me to engineer our new variable,
Distance. 

Python :

def sphere_dist(pick_lat, pick_lon, drop_lat, drop_lon):
    R_earth = 6371 # Earth radius (in km)
    pick_lat, pick_lon, drop_lat, drop_lon = map(np.radians, [pick_lat, pick_lon,
                                                              drop_lat, drop_lon])
    dlat = drop_lat - pick_lat
    dlon = drop_lon - pick_lon
    a = np.sin(dlat/2.0)**2 + np.cos(pick_lat) * np.cos(drop_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))
   
![image](https://user-images.githubusercontent.com/67412893/114296229-c76e5c00-9ac7-11eb-9148-e0ca98ca3010.png)

## Model Development

After all the above processes the next step is developing the model based on our prepared data.
In this project we got our target variable as “fare_amount”. The model has to predict a numeric value.
Thus, it is identified that this is a Regression problem statement. And to develop a regression model, the
various models that can be used are Linear Regression, Decision trees, Random Forest, XG Boost. 

### Decision Tree:

Decision Tree is a supervised learning predictive model that uses a set of binary rules to calculate the
target value/dependent variable.

Decision trees are divided into three main parts this are :
Root Node : performs the first split
Terminal Nodes : that predict the outcome, these are also called leaf nodes
Branches : arrows connecting nodes, showing the flow from root to other leaves.

Python:
![image](https://user-images.githubusercontent.com/67412893/114296439-fe913d00-9ac8-11eb-8a3e-d38378dc00fc.png)

Regression Graph:
![image](https://user-images.githubusercontent.com/67412893/114566538-04854a80-9c90-11eb-8bd8-e18e0214f6dd.png)


### Random Forest:

The next model to be followed in this project is Random forest. It is a process where the machine follows
an ensemble learning method for classification and regression that operates by developing a number of
decision trees at training time and giving output as the class that is the mode of the classes of all the
individual decision trees. 

Python:
![image](https://user-images.githubusercontent.com/67412893/114296880-326d6200-9acb-11eb-8ef1-6a641f7602b2.png)

Regression Graph:
![image](https://user-images.githubusercontent.com/67412893/114566621-1666ed80-9c90-11eb-85f7-2840831bf097.png)


### Linear Regression:

The next method in the process is Linear regression. It is used to predict the value of variable Y based on
one or more input predictor variables X. The goal of this method is to establish a linear relationship
between the predictor variables and the response variable. Such that, we can use this formula to estimate
the value of the response Y, when only the predictors (X- Values) are known. 

Python:
![image](https://user-images.githubusercontent.com/67412893/114296529-79f2ee80-9ac9-11eb-8c48-4ceb4468b1ba.png)

Regression Graph:
![image](https://user-images.githubusercontent.com/67412893/114566439-e4ee2200-9c8f-11eb-8015-fadd350358d1.png)


### XG Boost:

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the
Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code
runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

