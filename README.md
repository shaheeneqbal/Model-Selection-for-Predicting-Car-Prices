# Model-Selection-for-Predicting-Car-Prices
### About
In this project, I am using "car data" dataset for predicting Car prices using supervised machine learning techniques. Comparing and analyzing the results of 3 different algorithms: LinearRegression, RandomForestRegressor, and GradientBoostingRegressor to identify the best performing model.The goal of this project is to develop a model that can accurately estimate the price of a car based on various features.
### Introduction
Predicting the price of a car is a valuable task in the automotive industry. It provides insights to both buyers and sellers, which aids in their decision-making.  Car price prediction involves using machine learning algorithms to estimate the price of a car based on various factors. By analyzing historical data and identifying patterns, a predictive model can be trained to accurately estimate the price of a car.
### Prerequisites
Before begin, ensure that we have the following prerequisite:
* Jupyter Notebook
* Excel
### Data Source
"Car data" dataset from kaggle https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
### Dataset 
The dataset used for this project is a collection of car data, they are stored in a CSV (Comma Separated Values) file named "car data.csv" . It contains 301 rows and 9 columns. 

![image](https://github.com/shaheeneqbal/Model-Selection-for-Predicting-Car-Prices/assets/67499556/87518c46-1fa0-4099-a232-171f38cfd907)

### Model Selection

![image](https://github.com/shaheeneqbal/Model-Selection-for-Predicting-Car-Prices/assets/67499556/8f82794c-4452-4d4e-95ea-3327131999d3)

##### Step 1: Dependencies
To run the code in this repository, need to have the following dependencies installed:
* Python 3.6+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
##### Step 2: Explore the Dataset
Explore the dataset to understand its structure, identify any missing values, outliers, or inconsistencies.
##### Step 3: Data Cleaning and Preprocessing
Preprocessing the dataset involves handling missing values and outliers by either imputing them or removing the corresponding instances. Encode categorical variables using techniques such as one-hot encoding or label encoding. Scale numerical features if necessary to ensure all features have a similar range.
##### Step 4: Split the Dataset into Dependent and Independent Features
Spliting the preprocessed dataset into dependent and independent features. The dependent feature is the target variable, which in this case is the price and independent features are the remaining variables that will be used to predict the car price. 
##### Step 5: Feature Selection and Engineering
Feature selection and engineering are used to improve the model's accuracy and interpretability by including the most informative features and transforming them appropriately. It requires a combination of data analysis, statistical techniques, and domain knowledge to select and engineer relevant features effectively for prediction.
##### Step 6: Split the Dataset into Training and Testing
Spliting the preprocessed dataset into training and testing sets. The training set will be used to train the machine learning model, while the testing set will be used to evaluate its performance.
##### Step 7: Machine Learning Algorithms
The choice of algorithm depends on the characteristics of the dataset. For this dataset, using 3 different algorithms: LinearRegression, RandomForestRegressor, and GradientBoostingRegressor
##### Step 8: Train the Model
Train all the 3 machine learning models using the training data. The model will learn the patterns and relationships between the car features and their corresponding prices.
##### Step 9: Evaluate the Model
Evaluate the performance of the trained model using appropriate evaluation metrics such as mean absolute error (MAE), mean squared error (MSE), or root mean squared error (RMSE). These metrics provide insights into how well the model is predicting the car prices.
##### Step 10: Fine-tune the Model
If the model's performance is not satisfactory, consider fine-tuning the model by adjusting hyperparameters or trying different algorithms. This iterative process helps improve the model's accuracy.
##### Step 11: Make Predictions
Once the model is trained and evaluated, it is ready to make predictions on new, unseen car data. Provide the relevant features of a car as input to the model, and it will estimate the price based on the learned patterns.
##### Step 12: Compare Algorithm
Comparing and analyzing the results of 3 different algorithms to identify the best performing model.



### License
This project is licensed under the MIT License. Feel free to modify and use it as per your requirements.
### Conclusion
Car price prediction has gained significant attention due to its practical implications in the automotive industry. Accurate forecasts can be created by using machine learning algorithms and historical data. This has practical implications for both buyers and sellers in the car market, as well as for car dealerships, insurers, and financial institutions. It is an iterative process, and it may require experimentation and fine-tuning to achieve the best results. By following these steps, can build a reliable machine learning model for predicting car prices.
