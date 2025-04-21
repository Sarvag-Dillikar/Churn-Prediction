# Customer Churn Prediction

This project aims to predict customer churn (i.e., whether a customer will leave a service) using a machine learning model. The dataset used for this analysis contains information about customer attributes and their subscription status. Various machine learning algorithms are employed to predict the churn status.

## Dataset Overview

The dataset used in this project is stored in the file `churn_dataset.csv` and contains the following columns:

- **customerID**: Unique identifier for each customer.
- **gender**: Gender of the customer (Male/Female).
- **SeniorCitizen**: Whether the customer is a senior citizen (1 for Yes, 0 for No).
- **Partner**: Whether the customer has a partner (Yes/No).
- **Dependents**: Whether the customer has dependents (Yes/No).
- **tenure**: Number of months the customer has been with the company.
- **PhoneService**: Whether the customer has phone service (Yes/No).
- **MultipleLines**: Whether the customer has multiple lines (Yes/No/No phone service).
- **InternetService**: Type of internet service (DSL/Fiber optic/No internet service).
- **OnlineSecurity**: Whether the customer has online security (Yes/No/No internet service).
- **OnlineBackup**: Whether the customer has online backup (Yes/No/No internet service).
- **DeviceProtection**: Whether the customer has device protection (Yes/No/No internet service).
- **TechSupport**: Whether the customer has tech support (Yes/No/No internet service).
- **StreamingTV**: Whether the customer has streaming TV (Yes/No/No internet service).
- **StreamingMovies**: Whether the customer has streaming movies (Yes/No/No internet service).
- **Contract**: The type of contract the customer has (Month-to-month/One year/Two year).
- **PaperlessBilling**: Whether the customer has paperless billing (Yes/No).
- **PaymentMethod**: The payment method used by the customer (Electronic check/Mailed check/Bank transfer/Credit card).
- **MonthlyCharges**: The monthly charges for the customer.
- **TotalCharges**: The total charges incurred by the customer.
- **Churn**: Whether the customer has churned (Yes/No).

## Objective

The goal of this project is to predict whether a customer will churn based on the provided features using various machine learning algorithms. The model will be evaluated using accuracy scores and confusion matrices.

## Key Steps

1. **Data Loading**: The dataset is loaded using `pandas.read_csv("churn_dataset.csv")`.

2. **Data Preprocessing**:
   - Null values in the `TotalCharges` column are handled by coercing non-numeric values.
   - The dataset is checked for null values and missing data is removed using `dropna()`.

3. **Data Exploration**:
   - Boxplots are used to identify outliers in numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
   - Joint and scatter plots are used to explore the relationships between features and visualize trends in the data.
   - A pie chart is used to show the distribution of internet service types (DSL, Fiber optic, No internet service).

4. **Feature Engineering**:
   - Categorical variables are encoded using One-Hot Encoding.
   - Numerical features are standardized using `StandardScaler`.

5. **Modeling**:
   - The dataset is split into training and testing sets using `train_test_split`.
   - Several machine learning algorithms are applied to the dataset, including:
     - Logistic Regression
     - K-Nearest Neighbors
     - Support Vector Classifier (SVC)
     - Decision Tree Classifier
     - Gaussian Naive Bayes
     - AdaBoost Classifier
     - Random Forest Classifier

6. **Model Evaluation**:
   - The models are evaluated using accuracy scores and confusion matrices.
   - The performance of each model is visualized using histograms.

7. **Conclusion**:
   - Logistic Regression provides the highest accuracy score for churn prediction, making it the best model for this task.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical plotting.
- **Scikit-learn**: For machine learning algorithms, preprocessing, and evaluation.

## Setup

To run this project locally, you need to install the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
