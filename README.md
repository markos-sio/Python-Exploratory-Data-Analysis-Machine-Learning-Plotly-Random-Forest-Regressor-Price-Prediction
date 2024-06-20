Flight Price Prediction and Analysis
This project focuses on predicting flight prices using machine learning techniques and performing exploratory data analysis (EDA) to understand the dataset. Here's a breakdown of what's included:

1) Exploratory Data Analysis (EDA)
Data Import and Inspection: Loaded dataset from Clean_Dataset.csv using pandas. Explored basic information such as shape, data types, and statistical summaries.
Handling Missing Values: Visualized missing data patterns using missingno.
Duplicate Check: Ensured data integrity by checking for duplicate entries based on a unique column.
Visualizing Categorical Data: Utilized Plotly Express to create bar charts and donut charts for categorical variables to understand their distributions.
Price Distribution and Analysis: Analyzed price distribution and its range.
Mean Price Analysis by Categories: Investigated mean prices across categorical columns.
2) Data Preprocessing
Encoding: Applied one-hot encoding to categorical columns (airline, source_city, destination_city, departure_time).
Column Transformation: Converted class column to binary and stops column to numerical format.
Feature Selection: Dropped unnecessary columns (Unnamed: 0, flight) and retained relevant features (duration, days_left, price).
3) Modeling
Regression Modeling: Built a Random Forest Regressor model to predict flight prices.
Cross-validation: Used K-Fold cross-validation for robust evaluation.
Evaluation Metrics: Calculated metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Visualization of Predictions: Visualized predicted prices against actual prices using scatter plots.
4) Analysis and Interpretation
MAE and Price Range Comparison: Analyzed MAE relative to the price range to assess model performance.
Residual Analysis: Examined residuals to check model assumptions.
Cross-validation Consistency: Ensured consistency through cross-validated MAE.
Feature Importance: Identified top features influencing price predictions using Random Forest feature importances.
Prediction on New Data: Created a function to preprocess new data and predict flight prices.
This repository provides a comprehensive analysis of flight prices, from initial data exploration to predictive modeling, aimed at understanding and predicting flight costs effectively.