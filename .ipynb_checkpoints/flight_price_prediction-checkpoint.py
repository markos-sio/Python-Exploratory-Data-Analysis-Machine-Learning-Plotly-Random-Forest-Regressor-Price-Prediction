# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 2024

@author: markos-sio
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import KFold, cross_val_predict
import math
import missingno as msno

""" 1) Exploratory Data Analysis (EDA)"""
# Importing the csv file and creating a dataframe using pandas
data = pd.read_csv("Clean_Dataset.csv")

# Printing the head of the dataframe 
display(data.head())

# Printing the column names of the dataframe
print(data.columns)


# General statistics
print(f'Data shape :\n {data.shape}\n')
print(f'Data types :\n {data.dtypes}\n')
print(data.iloc[:, 1:].describe()) # Excluding the fisrt Unname: 0 column
print(data.info())

# Checking for missing values
display(data.isnull().sum())
msno.matrix(data)  

# Checking for duplicate values if any according to unique valued column Unnamed: 0  
display(data.duplicated("Unnamed: 0").sum())

# Inspecting the unique values of some columns

categorical_columns = ["airline", "source_city", "destination_city", "departure_time", "arrival_time", "stops", "class"]

for column in categorical_columns:
    print(f'{column} values:\n {data[column].unique()}\n')

# Counting values
for column in categorical_columns:
    # Getting the value counts for the current column
    column_data = data[column].value_counts().reset_index() #converting value_counts() into a dataframe
    column_data.columns = [column, 'count'] # Setting the name of the columns in the dataframe
    
    # Create a histogram with Plotly Express
    fig = px.bar(column_data, 
                 x=column, 
                 y='count', 
                 color=column, 
                 color_continuous_scale='Viridis',
                 labels={'count': 'Count', column: column}, 
                 title=f"Frequency of {column}")
    
    # Show the plot
    fig.show()

# Donut charts revealing the percentages
# Defining the number of rows and columns for subplots
rows, cols = 2, 4

# Creating a subplot for each categorical column with domain type
fig = make_subplots(rows=rows, cols=cols, subplot_titles=categorical_columns[:rows*cols],
                    specs=[[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)])

# Iterating over each categorical column
for i, column in enumerate(categorical_columns):
    if i >= rows * cols:  # Ensure not to exceed the subplot grid
        break
    
    # Calculating the counts and percentages
    counts = data[column].value_counts()
    percentages = counts / counts.sum() * 100

    # Creating a donut chart
    donut_chart = go.Pie(labels=counts.index, values=percentages, hole=0.4, 
                         textinfo='label+percent', insidetextorientation='radial',
                         marker=dict(colors=px.colors.qualitative.Pastel))
    
    # Determining the row and column for the current plot
    row = (i // cols) + 1
    col = (i % cols) + 1
    
    # Addind the donut chart to the subplot
    fig.add_trace(donut_chart, row=row, col=col)

# Updating layout to increase the gap between subplots and adjust figure size
fig.update_layout(height=600, width=1400, showlegend=False, 
                  title_text='Donut Charts with Percentages for Categorical Features', title_x=0.5)

# Displaying the figure
fig.show()
 
# Let΄s examine price distribution
display(data.iloc[:, -1:].describe())
price_range = data['price'].max() - data['price'].min()
print(f'The range of price is: {price_range}')


# Creating the histogram 
fig = px.histogram(
    data, 
    x='price', 
    nbins=20, 
    marginal='violin',
    title='Price Distribution', 
    color_discrete_sequence=['navy'],
    width=800,  
    height=750   
                   
)                 
# Show the plot
fig.show()



#Let΄s check the mean price by each categorical column

for column in categorical_columns:
    # Computing the mean price for each category
    mean_prices = data.groupby(column)['price'].mean().reset_index()
    
    # Creating a bar plot 
    fig = px.bar(
        mean_prices,
        x=column,
        y='price',
        labels={'price': 'Mean Price', column: column},
        title=f"Mean Price by {column}",
        color='price',
        color_continuous_scale='Viridis'
    )
    
    # Show the plot
    fig.show()

""" 2) Modeling """

"""
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

# Let΄s dive into the correlations between variables
corr_matrix = data.corr()

# Creating the heatmap using Plotly Express
fig = px.imshow(
    corr_matrix, 
    text_auto=True,  
    color_continuous_scale='Viridis', 
    title='Correlation Matrix Heatmap',
    width=2000,  # Adjust the width of the plot if needed
    height=2000
)

fig.show()

"""Regression Model"""

# Excluding the target variable 'price' column
data_without_price = data.drop("price", axis=1)
X, y = data_without_price, data["price"]  # Features are everything except 'price', target is 'price'

# Initializing the RandomForestRegressor model
rf = RandomForestRegressor()

# Using K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation predictions
y_pred = cross_val_predict(rf, X, y, cv=kf)

# Calculating metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mse)

# Printing metrics
print('R Squared:', r2)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)

# Creating a DataFrame for plotting
plot_data = pd.DataFrame({
    'Original Price': y,
    'Predicted Price': y_pred
})

# Creating scatter plot 
fig = px.scatter(
    plot_data, 
    x='Original Price', 
    y='Predicted Price', 
    title='Predicted Prices vs Original Prices',
    labels={'Original Price': 'Original Price', 'Predicted Price': 'Predicted Price'},
    width=800,
    height=600
)

fig.show()

""" A) Comparison between Mean Absolute Error (mae) and  range Price (dependent Variable) """

price_min = data["price"].min()
price_max = data["price"].max()
price_range = price_max - price_min
print(f'Price Range: {price_range}')

mae_percentage = (mae / price_range) * 100
print(f'mae as percentage of Range: {mae_percentage:.2f}%')

""" B) Residual Analysis """

residuals = y - y_pred

# Creating a DataFrame for plotting
plot_data = pd.DataFrame({
    'Predicted Prices': y_pred,
    'Residuals': residuals
})

# Create scatter plot 
fig = px.scatter(
    plot_data, 
    x='Predicted Prices', 
    y='Residuals', 
    title='Residuals vs Predicted Prices',
    labels={'Predicted Prices': 'Predicted Prices', 'Residuals': 'Residuals'},
    width=800,
    height=600
)

# Adding horizontal line at y=0 (zero line)
fig.add_hline(y=0, line_dash="dash", line_color="red")

fig.show()


""" C) Cross-Validation Consistency Check """

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
print(f'Cross-validated mae: {-cv_scores.mean()} (std: {cv_scores.std()})')


""" D) Acceptable range of mae """

avg_price = data["price"].mean()
print(f'Average Price: {avg_price}')
if mae < 0.1 * avg_price:  # Assuming 10% of average price as threshold
    print("mae is within acceptable range.")
else:
    print("mae is outside acceptable range.")

        
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
print(top_three_importances)

# Creating a DataFrame for plotting 
plot_data = pd.DataFrame({
    'Features': [x[0] for x in top_three_importances],
    'Importance': [x[1] for x in top_three_importances]
})

# Creating a bar plot 
fig = px.bar(
    plot_data,
    x='Features',
    y='Importance',
    title='Top 3 Feature Importances',
    labels={'Features': 'Features', 'Importance': 'Importance'},
    width=700,
    height=500
)

fig.show()


""" Predicting price value given unprocessed data """

# Function to preprocess new data
def preprocess_new_data(new_data):
    # Apply the same transformations as the training data
    new_data["class"] = new_data["class"].apply(lambda x: 0 if x == "Economy" else 1)
    new_data["stops"] = pd.factorize(new_data["stops"])[0]

    # One-hot encode categorical variables
    new_data = pd.get_dummies(new_data, columns=["airline", "source_city", "destination_city", "departure_time", "arrival_time"])

    # Ensure the new data has the same columns as the training data
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

# Preprocess the new data
new_data_preprocessed = preprocess_new_data(new_data)

# Predict the price for the new data
predicted_price = rf.predict(new_data_preprocessed)

print("Predicted Price:", predicted_price)
