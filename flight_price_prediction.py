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


# Inspecting the unique values of some columns

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

# Getting the dummies of source city column and joining them  to the dataframe
source_city_dummies = pd.get_dummies(data["source_city"], prefix="source_city")
data = data.join(source_city_dummies)

# Getting the dummies of destination_city column and joining them  to the dataframe
destination_city_dummies = pd.get_dummies(data["destination_city"], prefix="destination_city")
data = data.join(destination_city_dummies)

# Getting the dummies of airline column and joining them  to the dataframe
airline_dummies = pd.get_dummies(data["airline"], prefix="airline")
data = data.join(airline_dummies)

# Getting the dummies of arrival_time column and joining them  to the dataframe
arrival_time_dummies = pd.get_dummies(data["arrival_time"], prefix="arrival_time")
data = data.join(arrival_time_dummies)

# Getting the dummies of departure_time column and joining them  to the dataframe
departure_time_dummies = pd.get_dummies(data["departure_time"], prefix="departure_time")
data = data.join(departure_time_dummies)

data = data.drop(["airline", "source_city", "destination_city", "departure_time", "arrival_time"], axis=1)

print(data.head())



"""Regression Model"""

from sklearn.model_selection import KFold, cross_val_predict
# Training Regression Model
data_without_price = data.drop("price", axis=1) # Excluding the target variable 'price' column
X, y = data_without_price, data["price"] # Features are everything except 'price', target is 'price'

# Initializing the RandomForestRegressor model
rf = RandomForestRegressor()

# Using K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation predictions
y_pred = cross_val_predict(rf, X, y, cv=kf)

# Calculating and printing the R^2 score
print('R Squared', r2_score(y, y_pred))

# Calculating and printing the Mean Absolute Error
print('Mean Absolute Error', mean_absolute_error(y, y_pred))

# Calculating and printing the Mean Squared Error
print('Mean Squared Error', mean_squared_error(y, y_pred))

# Calculating and printing the Root Mean Squared Error
print('Root Mean Squared', math.sqrt(mean_squared_error(y, y_pred)))

# Plotting the predicted prices vs the original prices
plt.scatter(y, y_pred)
plt.xlabel("Original Price")
plt.ylabel("Predicted Prices")
plt.title("Predicted Prices vs Original Prices")
plt.show()

""" A) Comparison between Mean Absolute Error (MAE) and  range Price (dependent Variable) """

price_min = data["price"].min()
price_max = data["price"].max()
price_range = price_max - price_min
print(f'Price Range: {price_range}')

MAE =  mean_absolute_error(y, y_pred)
MAE_percentage = (MAE / price_range) * 100
print(f'MAE as percentage of Range: {MAE_percentage:.2f}%')


""" B) Residual Analysis """

residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.show()


""" C) Cross-Validation Consistency Check """

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
print(f'Cross-validated MAE: {-cv_scores.mean()} (std: {cv_scores.std()})')


""" D) Acceptable range of MAE """

avg_price = data["price"].mean()
print(f'Average Price: {avg_price}')
if MAE < 0.1 * avg_price:  # Assuming 10% of average price as threshold
    print("MAE is within acceptable range.")
else:
    print("MAE is outside acceptable range.")

        
""" E) Feature Importance and Interpretation """

# Training the model on the entire dataset to get feature importances
rf.fit(X, y)

#Extracting feature names and their importances from the model
feature_names = rf.feature_names_in_ # Getting the names of the features used by the model
feature_importances = rf.feature_importances_ # Getting the importance of each feature

# Combining the feature names and their importances into a dictionary
importances =  dict(zip(feature_names, feature_importances))

#Sorting the features by their importances in descending order
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True) # importances.items() -list of tuples, x[1] for the second ellement, reverse=True for descending order

#Printing the top three sorted list of features and their importances
top_three_importances = sorted_importances[:3]
display(top_three_importances)

plt.figure(figsize=(7, 5))
plt.bar([x[0] for x in top_three_importances],
        [x[1] for x in top_three_importances]
)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 3 Feature Importances')
plt.show()


""" Predicting price value given unprocessed data """

# Function to preprocess new data
def preprocess_new_data(new_data):
    # Apply the same transformations as the training data
    new_data["class"] = new_data["class"].apply(lambda x: 0 if x == "Economy" else 1)
    new_data["stops"] = pd.factorize(new_data["stops"])[0]

    # One-hot encoding categorical variables
    new_data = pd.get_dummies(new_data, columns=["airline", "source_city", "destination_city", "departure_time", "arrival_time"])

    # Ensuring the new data has the same columns as the training data
    missing_cols = set(X.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[X.columns]

    return new_data

# Example new data for prediction
new_data = pd.DataFrame({
    "airline": ["IndiGo"],
    "source_city": ["Delhi"],
    "destination_city": ["Cochin"],
    "departure_time": ["Morning"],
    "arrival_time": ["Evening"],
    "stops": ["1 stop"],
    "class": ["Economy"],
    "duration": [180],
    "days_left": [30]
})

# Preprocessing the new data
new_data_preprocessed = preprocess_new_data(new_data)

# Predicting the price for the new data
predicted_price = rf.predict(new_data_preprocessed)

print("Predicted Price:", predicted_price)

