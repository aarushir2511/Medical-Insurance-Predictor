# **Medical Insurance Predictor**

## **Project Overview**
The Medical Insurance Predictor is a machine learning-based application that predicts medical insurance costs for individuals based on various factors such as age, BMI, smoking status, and more. This tool can help insurance companies estimate premiums and assist users in understanding the factors influencing their insurance costs.

## **Features**
- Predicts medical insurance costs based on user input.
- Utilizes various machine learning algorithms to provide accurate predictions.
- User-friendly web interface to interact with the model.
- Data visualization to help users understand the relationship between features and insurance costs.

## **Dataset**
The dataset used for training the model includes the following features:
- `age`: Age of the individual.
- `sex`: Gender of the individual (`male` or `female`).
- `bmi`: Body Mass Index, a measure of body fat based on height and weight.
- `children`: Number of children/dependents covered by the insurance.
- `smoker`: Smoking status (`yes` or `no`).
- `region`: The individual's residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`).
- `charges`: The medical insurance costs (target variable).

## **Model Training**
The model training process involves the following steps:

1. **Data Preprocessing**:
   - **Handling Missing Values**: The dataset is checked for missing values, and appropriate strategies such as imputation or removal are applied.
   - **Encoding Categorical Variables**: The categorical variables like `sex`, `smoker`, and `region` are encoded using techniques like one-hot encoding or label encoding.
   - **Feature Scaling**: Continuous features such as `age` and `bmi` are scaled using StandardScaler or MinMaxScaler to ensure uniformity across features.

2. **Model Selection**:
   - Various machine learning algorithms were explored, including Linear Regression, Decision Trees, Random Forests, KNN and SVM.
   - The best-performing model, in this case, [e.g., Random Forest Regressor], was chosen for deployment based on metrics like RMSE and R-squared.

3. **Training the Model**:
   - The model was trained on 60% of the data, with the remaining 40% was divided equally between test set and cv set.
   - The model was then tuned by making prediction on the cv set.
   - The trained model is saved as a serialized file (e.g., .pkl or .sav format).

4. **Model Evaluation**:
   - The performance of the model is evaluated using the test set, and metrics such as Root Mean Square Error (RMSE), and R-squared are calculated.
   - The results are compared against baseline models to ensure that the chosen model provides a significant improvement.

## **Web Application**
The web application is built using Streamlit and provides an intuitive interface for users to interact with the model:

1. **User Input**:
   - Users can input their data, such as age, BMI, number of children, smoking status, and region, through the web interface.
   - The application immediately processes these inputs and passes them to the trained model for prediction.

2. **Prediction**:
   - The predicted insurance cost is displayed on the screen, providing users with an estimate based on their inputs.
   - Additional visualizations, such as feature importance charts and distribution plots, are available to help users understand how each factor influences the prediction.

3. **Data Visualization**:
   - The app includes several charts and graphs to show relationships between features and insurance costs, such as how smoking status or BMI impacts the predicted charge.
   - Users can explore these visualizations to gain insights into the underlying data and model behavior.

## **Results**
The results of the model training and evaluation are as follows:
The model performed exceptionally well, with a Root Mean Square Error (RMSE) of 2449.58 and an R-squared value of 0.94,(These values are before training the dataset on the cv set) showing it fits the data nicely. It cleverly captured complex relationships between features like BMI and smoking status, surpassing simpler models. The key predictors were found to be smoking status, age, and BMI, which aligns with what we expect. When stacked up against basic models like linear regression, our Random Forest model outperformed them by significantly cutting down prediction errors, making it the star of the show!


## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
