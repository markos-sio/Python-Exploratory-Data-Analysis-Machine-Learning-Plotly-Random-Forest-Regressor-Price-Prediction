# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:41:54 2024

@author: markos
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import math
import matplotlib.pyplot as plt


# Importing the csv file and creating a dataframe using pandas
data = pd.read_csv("Clean_Dataset.csv")

# Printing the head of the dataframe 
print(data.head())

# Printing the column names of the dataframe
print(data.columns)


# General statistics
print(data.dtypes)
print(data.describe())
print(data.info())


# inspecting the unique values of some columns

columns_of_interest = ["airline", "source_city", "destination_city", "departure_time", "arrival_time", "stops", "class"]

for column in columns_of_interest:
    print(f'The unique values of {column} are: {data[column].unique()}\n')
    
"""
Preprocessing

We are going to proceed one_hot encoding for the columns:
airline, source_city, destination_city and departure_time
as they have a small number of non-numerical unique values .

Then we are going to turn the class column to binary as it contains ony two non numerical values
and the the stops column values to numerical 0, 1 and 2.

Also we are going to drop the Unnamed: 0 and flight columns as the are not nessecary for our analysis
and finaly we are going to let columns duration, days_left and price as they are.

"""

data = data.drop(["Unnamed: 0", "flight"], axis=1)
data["class"] = data["class"].apply(lambda x: 0 if x == "Economy" else 1)
data["stops"] = pd.factorize(data["stops"])[0]

print(data.head())

# Get the dummies of source city column and join them  to the dataframe
source_city_dummies = pd.get_dummies(data["source_city"], prefix="source_city")
data = data.join(source_city_dummies)

# Get the dummies of destination_city column and join them  to the dataframe
destination_city_dummies = pd.get_dummies(data["destination_city"], prefix="destination_city")
data = data.join(destination_city_dummies)

# Get the dummies of airline column and join them  to the dataframe
airline_dummies = pd.get_dummies(data["airline"], prefix="airline")
data = data.join(airline_dummies)

# Get the dummies of arrival_time column and join them  to the dataframe
arrival_time_dummies = pd.get_dummies(data["arrival_time"], prefix="arrival_time")
data = data.join(arrival_time_dummies)

# Get the dummies of departure_time column and join them  to the dataframe
departure_time_dummies = pd.get_dummies(data["departure_time"], prefix="departure_time")
data = data.join(departure_time_dummies)

data = data.drop(["airline", "source_city", "destination_city", "departure_time", "arrival_time"], axis=1)

print(data.head())





# Training Regression Model


data_without_price = data.drop("price", axis=1) # Excluding the target variable 'price' column
X, y = data_without_price, data["price"] # Features are everything except 'price', target is 'price'

# Spliting the data into training and testing sets, with 20% of the data for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random_state for ensuring that the random process produce the same results in every run

# Initialize the RandomForestRegressor model
rf = RandomForestRegressor()

# Train the model on the training data
rf.fit(X_train, y_train)

# Calculate and print the R^2 score on the test data
score = rf.score(X_test, y_test)
print(score)

# Predict the prices on the test data
y_pred = rf.predict(X_test)

# Calculate and print the R^2 score
print('R Squared', r2_score(y_test, y_pred))

# Calculate and print the Mean Absolute Error
print('Mean Absolute Error', mean_absolute_error(y_test, y_pred))

# Calculate and print the Mean Squared Error
print('Mean Squared Error', mean_squared_error(y_test, y_pred))

# Calculate and print the Root Mean Squared Error
print('Root Mean Squared', math.sqrt(mean_squared_error(y_test, y_pred)))

# Plot the predicted prices vs the original prices
plt.scatter(y_test, y_pred)
plt.xlabel("Original Price")
plt.ylabel("Predicted Prices")
plt.title("Predicted Prices vs Original Prices")
plt.show()

# Calculate and print feature importances from the trained RandomForest model
importances = dict(zip(rf.feature_names_in_, rf.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print(sorted_importances)
