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

You can find the dataset [here](link-to-dataset).

## **Installation**
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/medical-insurance-predictor.git
    cd medical-insurance-predictor
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## **Usage**
Once installed, you can run the application locally:

1. Start the web application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Enter the required details such as age, BMI, and smoking status to predict the insurance cost.

## **Model Training**
The model training process involves the following steps:

1. **Data Preprocessing**:
   - **Handling Missing Values**: The dataset is checked for missing values, and appropriate strategies such as imputation or removal are applied.
   - **Encoding Categorical Variables**: The categorical variables like `sex`, `smoker`, and `region` are encoded using techniques like one-hot encoding or label encoding.
   - **Feature Scaling**: Continuous features such as `age` and `bmi` are scaled using StandardScaler or MinMaxScaler to ensure uniformity across features.

2. **Model Selection**:
   - Various machine learning algorithms were explored, including Linear Regression, Decision Trees, Random Forests, and Gradient Boosting Machines (GBM).
   - A grid search with cross-validation was performed to find the best hyperparameters for the selected model.
   - The best-performing model, in this case, [e.g., Gradient Boosting Regressor], was chosen for deployment based on metrics like Mean Absolute Error (MAE) and R-squared.

3. **Training the Model**:
   - The model was trained on 80% of the data, with the remaining 20% used as a test set.
   - During training, techniques like early stopping and regularization were applied to prevent overfitting.
   - The trained model is saved in the `models` directory as a serialized file (`.pkl` or `.h5` format).

4. **Model Evaluation**:
   - The performance of the model is evaluated using the test set, and metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared are calculated.
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

1. **Model Performance**:
   - The final model achieved a Mean Absolute Error (MAE) of `X.XX` and an R-squared value of `Y.YY` on the test set, indicating a good fit for the data.
   - The model was able to capture the non-linear relationships between features like BMI, smoking status, and insurance costs, outperforming simpler linear models.

2. **Feature Importance**:
   - The most important features influencing the prediction were found to be `smoker`, `age`, and `bmi`. These features had the highest impact on the model’s predictions, which aligns with domain knowledge.

3. **Model Interpretability**:
   - SHAP (SHapley Additive exPlanations) values were used to interpret the model's predictions. This method provides insight into how each feature contributes to the final prediction for individual instances.

4. **Comparison with Baselines**:
   - The chosen model was compared with baseline models such as simple linear regression and decision trees. The Gradient Boosting model showed a significant reduction in prediction error, making it the best choice for deployment.

## **Contributing**
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Add your commit message"
    ```
4. Push to the branch:
    ```bash
    git push origin feature/your-feature-name
    ```
5. Create a pull request.

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Contact**
For any questions or issues, feel free to reach out:

- Your Name - [Your Email](mailto:your-email@example.com)
- LinkedIn - [Your LinkedIn](https://www.linkedin.com/in/yourprofile/)
