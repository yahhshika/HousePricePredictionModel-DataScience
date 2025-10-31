# House Price Prediction Model – Data Science

## Overview  
This project focuses on building a predictive model to estimate house prices using machine learning techniques and a real-world housing dataset. The goal is to analyze the data, build and validate models, and then deploy or use the best model for predictions.

## Project Structure  
/ <br>
├─ 1-main.py # Main script: data preprocessing, model training <br>
├─ 2-CheckModelsAccuracy.py # Script to check and compare models’ accuracy <br>
├─ housing.csv # Dataset of houses and their attributes/prices <br>
└─ README.md # Project overview (this file) <br>

## Dataset  
The dataset (`housing.csv`) contains various features of houses (e.g., number of rooms, area, location proxies) and the target variable: house price.  
You can use this dataset to:  
- explore data distribution & relationships  
- clean / preprocess data  
- engineer features  
- train & validate machine learning models  
- select the best performer  

## Key Steps  
1. **Data Exploration** – Inspect data, get summary statistics, visualize key relationships.  
2. **Preprocessing & Feature Engineering** – Handle missing values, encode categorical variables, create new features as needed.  
3. **Model Training & Validation** – Try out different models (e.g., Linear Regression, Random Forest, Gradient Boosting), tune hyperparameters, validate performance using cross-validation.  
4. **Model Comparison & Accuracy Checking** – Use `2-CheckModelsAccuracy.py` to compare models, check metrics like RMSE, MAE, R².  
5. **Final Model Selection** – Choose the best model and prepare it for prediction use or deployment.
