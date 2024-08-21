import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pickle 
import numpy as np

if 'page' not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1


if st.session_state.page == 1:
    st.title("Medical Insurance Predictor")
    
    st.header('ABOUT')
    st.write('The Medical Insurance Predictor is a machine learning-based application that predicts medical insurance costs for individuals based on various factors such as age, BMI, smoking status, and more. This tool can help insurance companies estimate premiums and assist users in understanding the factors influencing their insurance costs.')

    st.header('Dataset :-')
    df = pd.read_csv('medical_insurance.csv')
    st.dataframe(df, height=500)

    st.subheader("Dataset Features")
    st.markdown("""
    The dataset used for training the model includes the following features:
    - **age**: Age of the individual.
    - **sex**: Gender of the individual (`male` or `female`).
    - **bmi**: Body Mass Index, a measure of body fat based on height and weight.
    - **children**: Number of children/dependents covered by the insurance.
    - **smoker**: Smoking status (`yes` or `no`).
    - **region**: The individual's residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`).
    - **charges**: The medical insurance costs (target variable).
    """)

    st.header("Features")
    st.markdown("""
    - **Predicts medical insurance costs based on user input.**
    - **Utilizes various machine learning algorithms to provide accurate predictions.**
    - **User-friendly web interface to interact with the model.**
    - **Data visualization to help users understand the relationship between features and insurance costs.**
    - **The main feature of this model is that I have used multiple machine learning models and judged them based on their RMSE and R2 score and chose the best one out of them.**                
    """)

    st.header('How was the predictor built')
    st.markdown("""
    ### Data Preprocessing:
    - **Handling Missing Values**: The dataset is checked for missing values, and appropriate strategies such as imputation or removal are applied.
    - **Encoding Categorical Variables**: The categorical variables like `sex`, `smoker`, and `region` are encoded using the technique label encoding.
    - **Feature Scaling**: Continuous features such as `age` and `bmi` are scaled using StandardScaler to ensure uniformity across features.

    ### Model Selection:
    - Various machine learning algorithms were explored, including **Linear Regression**, **Decision Trees**, **Random Forests**, **KNN** and **SVR**.
    - The best-performing model, in this case, *Random Forest Regressor*, was chosen for deployment based on metrics like Root Mean Square Error (RMSE) and R-squared score.

    ### Training the Model:
    - The model was trained on **60%** of the data, with the remaining **40%** was divided equally between test set and cv set.
    - The model was then tuned by making prediction on the cv set.
    - The trained model is saved as a serialized file (e.g., `.pkl` or `.sav` format).

    ### Model Evaluation:
    - The performance of the model is evaluated using the test set, and metrics such as  **Root Mean Square Error (RMSE)**, and **R-squared** are calculated.
    - The results are compared against baseline models to ensure that the chosen model provides a significant improvement.
    """)

    st.header("Web Application")
    st.markdown("""
    The web application, built with Streamlit, offers an intuitive interface for users to interact with the model. Users can input their details, including age, sex, BMI, number of children, smoking status, and region, through the web interface. These inputs are processed in real-time and passed to the trained model for prediction. The predicted insurance cost is then displayed on the screen, providing users with an estimate based on their inputs. Additionally, the application features visualizations such as charts and plots to illustrate how different factors like smoking status or BMI influence the prediction, allowing users to gain insights into the data and model behavior.
    """)

    st.header("Results")
    st.markdown("""
    The model performed exceptionally well, with a Root Mean Square Error (RMSE) of 2449.58 and an R-squared value of 0.94,(These values are before training the dataset on the cv set) showing it fits the data nicely. It cleverly captured complex relationships between features like BMI and smoking status, surpassing simpler models. The key predictors were found to be smoking status, age, and BMI, which aligns with what we expect. When stacked up against basic models like linear regression, our Random Forest model outperformed them by significantly cutting down prediction errors, making it the star of the show!
    """)

    st.button("NEXT",on_click=next_page)

if st.session_state.page == 2:
    st.title('Data Visualization')
    st.write('Here we will only discuss how smoker, age and bmi affect predictor and the correlation between every every entity to each entity.')

    df = pd.read_csv('medical_insurance.csv')

    st.subheader("Age VS Charges")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    ax1.scatter(df['age'], df['charges'], color='b', marker='o')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Charges')
    ax1.set_title('Age vs Charges')
    st.pyplot(fig)

    st.write('This scatter plot illustrates the relationship between age (on the x-axis) and insurance charges (on the y-axis). Each blue dot represents an individuals data point, with their age and the corresponding amount of charges. The plot reveals a general trend that as age increases, insurance charges tend to increase as well. However, the data is widely spread, indicating variability in charges for individuals of the same age. The plot also shows distinct horizontal bands, which suggest the presence of groups within the data with similar charges despite differences in age.')

    st.subheader('bmi VS Charges')
    fig,ax2 = plt.subplots(figsize=(10,6))
    ax2.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    ax2.scatter(df['bmi'], df['charges'], color='b', marker='o')
    ax2.set_xlabel('bmi')
    ax2.set_ylabel('Charges')
    ax2.set_title('bmi vs charges')
    st.pyplot(fig)

    st.write('This scatter plot illustrates the relationship between Body Mass Index (BMI) and insurance charges. Each blue dot represents an individuals BMI and the corresponding insurance charges. The plot shows that while there is some variation in charges as BMI increases, there is no clear linear trend. Instead, the data is widely dispersed, indicating that BMI alone may not be a strong predictor of insurance charges. The dense cluster of points in the lower BMI range (20 to 30) with lower charges suggests that many individuals with average BMI tend to have lower insurance charges. However, the horizontal band of points at the upper end of charges (around 35,000) suggests that some individuals, regardless of their BMI, are incurring high insurance costs, potentially due to other factors not represented in this plot.')

    st.subheader('smoker VS charges')
    fig,ax3 = plt.subplots(figsize=(10,6))
    ax3.set_facecolor('lightcyan')
    fig.patch.set_facecolor('lightcyan')
    ax3.scatter(df['smoker'],df['charges'],color='b',marker='o')
    ax3.set_xlabel('smoker')
    ax3.set_ylabel('charges')
    ax3.set_title('smoker vs charges')
    st.pyplot(fig)

    st.write('The scatter plot reveals that smokers tend to have higher medical charges compared to non-smokers. On the x-axis, "yes" represents smokers and "no" represents non-smokers, while the y-axis shows the charges. The data points for smokers are more dispersed and concentrated at higher values, indicating that smoking is associated with significantly higher medical costs. Non-smokers have a more clustered distribution at lower charges, suggesting lower overall medical expenses. This pattern highlights a strong correlation between smoking and increased healthcare costs.')

    st.subheader('Correlation')
    df_copy = df.copy()
    lc = LabelEncoder()
    df_copy['sex']=lc.fit_transform(df['sex'])
    df_copy['region']=lc.fit_transform(df['region'])
    df_copy['smoker']=lc.fit_transform(df['smoker'])

    df_corr = df_copy.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='magma')
    st.pyplot(plt)

    st.write("""
    ### Analysis:
    - **Smoker** has the highest positive correlation with **charges** (0.79), indicating that being a smoker is strongly associated with higher medical charges.
    - **Age** also shows a moderate positive correlation with **charges** (0.3), suggesting that older individuals tend to incur higher medical costs.
    - **BMI** (Body Mass Index) has a smaller positive correlation with **charges** (0.17), indicating that higher BMI may slightly increase medical costs.
    - Other factors such as **sex**, **children**, and **region** show weak or negligible correlations with **charges**.

    Overall, the heatmap emphasizes the significant impact of smoking and age on medical expenses, with smoking being the most influential factor in driving up charges.
    """)

    left,right = st.columns([1,0.15])
    with right:
     st.button('NEXT',on_click=next_page)
    with left:
        st.button('BACK',on_click=prev_page)
if st.session_state.page == 3:
    st.header('Medical Insurance Predictor')

    with open('med_model_rfr.sav' , 'rb') as file:
        rfr2 = pickle.load(file)

    df = pd.read_csv('medical_insurance.csv')
    
    left,right = st.columns(2)
    
    sex = st.selectbox('Sex' , ['Male' , 'Female'])
    if sex == 'Male':
        sex_male = 1
    else:
        sex_male = 0
    

    age = st.slider('Age',1,100)


    with left:
        height = st.number_input("Enter your height (in m)", min_value=0.0, format="%.2f")
    with right:
        weight = st.number_input("Enter your weight (in kg)", min_value=0.0, format="%.2f")

    with left:
        if height>0 and weight>0:
          bmi = weight/(height**2)
          st.write(f"Your BMI is {bmi}")
        else:
            bmi = None

    children = st.slider('Number of Children' ,0,10)

    smoker = st.selectbox('Are you a Smoker',['Yes','No'])
    if smoker == 'Yes':
        is_smoker = 1
    else:
        is_smoker = 0
   
    region = st.selectbox('Enter your region' , ['southwest' , 'southeast' , 'northwest' , 'northeast' ])
    if region == 'northeast':
        reg = 0
    elif region == 'northwest':
        reg = 1
    elif region == 'southeast':
        reg = 2
    else:
        reg  = 3

    def prediction_rfr_pt2(new_input):
        prediction = rfr2.predict(new_input)
        prediction = prediction.round(2)
        return prediction

    if st.button("Predict"):
        if bmi==None:
            st.error('**please input height and weight**')
        else:
            inputs = np.array([[age, sex_male, bmi, children, is_smoker, reg]])
            result = prediction_rfr_pt2(inputs)
            st.success(f"The predicted insurance cost is {result}")

    st.button('BACK',on_click=prev_page)
