# --------------- INITIAL CONFIGURATION ---------------
# Library imports
import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Environment configuration
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Streamlit configurations
st.set_page_config(
    page_title="Telecom Churn",
    page_icon = "📶",
    layout = "wide"
)

st.title("📶 Telco Telecom Case")

# --------------- FUNCTIONS ---------------

@st.cache_data
def load_data():
    """
    Loads and processes data from the .csv file
    """
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    
    return df

def plot_bar(data, x, y, color, title, barmode, xlabel, ylabel):
    """
    Creates a bar chart using Plotly Express
    """
    fig = px.bar(
        data,
        x = x,
        y = y,
        color = color,
        title = title,
        barmode = barmode,
        labels = {x: xlabel, y: ylabel},
        color_discrete_sequence=['#0f4c5c', '#9a031e']
    )

    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = xlabel,
        yaxis_title = ylabel,
    )

    return fig

def plot_hist(data, x, color, title, xlabel, ylabel):
    """
    Creates a histogram using Plotly Express
    """
    fig = px.histogram(
        data,
        x = x,
        color = color,
        barmode = 'overlay',
        title = title,
        labels = {x: xlabel, 'count': ylabel},
        color_discrete_sequence=['#0f4c5c', '#9a031e']
    )

    fig.update_layout(
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis_title = xlabel,
        yaxis_title = ylabel,
        bargap = 0.05
    )

    return fig

# --------------- DATA ---------------

df = load_data()
model = joblib.load("models/classifier.pkl")

# --------------- TABS ---------------
tab_report, tab_pred, tab_analytics = st.tabs(["📝 Report","🤖 Predictor", "📊 Dashboard"])

# ------------- ANALYSIS REPORT -------------
with tab_report:
    st.title("Analysis Report")
    st.markdown(
        '''
        ## Executive Summary

        **Purpose**: Identification of patterns and insights about churn cases in the third 
        quarter of 2024.

        **Key Insights**:

        - Customers with short contract time have higher churn probability.
        - Monthly renewal contracts have higher churn rate.
        - Customers who make automatic payments have lower churn probability.

        ---

        ## 1. Introduction

        **Problem**: Telco Telecom needs an understanding of Churn cases to 
        interpret its current state in the company, and plan actions to reduce it.

        **Data Source**: 
        
        - Internal CRM (Quarter 3 - 2024)
        - 7043 consumers, 21 variables 

        **Scope**: Relationships between contract type and duration, service type, 
        lifetime value, among other variables with Churn cases were explored. 

        ---

        ## 2. Data 

        ### Dataset Structure
        |Feature|Description|
        |---|---|
        |`customerID`|Unique customer identifier|
        |`gender`|Gender|
        |`SeniorCitizen`|Is senior?|
        |`Partner`|Has partner?|
        |`Dependents`|Has dependents?|
        |`tenure`|Relationship time (in months)|
        |`PhoneService`|Has phone service?|
        |`MultipleLines`|Has multiple lines?(Yes, no, no phone service)|
        |`InternetService`|Internet service provider (DSL, Fiber or none)|
        |`OnlineSecurity`|Has online security?|
        |`OnlineBackup`|Has online backup?|
        |`DeviceProtection`|Has device protection?|
        |`TechSupport`|Has tech support?|
        |`StreamingTV`|Has TV streaming?|
        |`StreamingMovies`|Has movie streaming?|
        |`Contract`|Contract type(month-to-month, annual or bi-annual)|
        |`PaperlessBilling`|Receives bills?|
        |`PaymentMethod`|Payment method|
        |`MonthlyCharges`|Service fee|
        |`TotalCharges`|Total paid by customer|
        |`Churn`|Target|

        ### Data Quality
        - Null values: `TotalCharges` has a total of 11 null values.
        - Outliers: No extreme values were found.

        ### Cleaning and manipulation
        - For the analysis, it was necessary to adjust the `SeniorCitizen` column from numeric binary (0, 1) 
        to text form ('Yes' and 'No').
        - The `TotalCharges` column had filling problems that were corrected.

        ---
        '''
    )
    st.markdown(
        '''
        ## 3. Analysis and insights
        ### 3.1. Customer retention
        '''
    )

    churn = df['Churn'].value_counts(normalize = True).reset_index()
    churn['proportion'] = (churn['proportion'] * 100).round(2)

    st.plotly_chart(
        plot_bar(
            churn, 
            x = 'Churn', 
            y = 'proportion', 
            color = 'Churn', 
            title = 'Customer Retention Distribution', 
            barmode = 'relative',
            xlabel = 'Churn', 
            ylabel = 'Proporção'
        ), 
        use_container_width=True
    )
    st.write(
        '''
        Customer retention is one of the great challenges in the telephony sector, Telco Telecom 
        maintains a retention rate of 73.46% in its contracts, with an average of 32 months, that is, 
        a little more than two years in the duration of the customer/company relationship.

        The most frequent contract type is monthly renewal, the most common payment method 
        is eCheck (electronic check). 

        ### 3.2. Internet services
        '''
    )

    internet = df.groupby(['Churn', 'InternetService']).agg(Count = ('InternetService', 'count')).reset_index()

    st.plotly_chart(
        plot_bar(
            internet, 
            x = 'InternetService', 
            y = 'Count', 
            color = 'Churn', 
            title = 'Internet Service x Churn', 
            barmode = 'group',
            xlabel = 'Service', 
            ylabel = 'Quantity'
        ), 
        use_container_width=True
    )
    st.markdown(
        '''
        Fiber optic service is the second most used among internet services but has a high 
        proportion of churn compared to DSL and customers who do not have internet contracted.

        ### 3.3. Contract Type
        '''
    )

    contract = df.groupby(['Churn', 'Contract']).agg(Count = ('Contract', 'count')).reset_index()
    contract_plot = plot_bar(
            contract,
            x = 'Contract',
            y = 'Count',
            color = 'Churn',
            title = 'Contract Type x Churn',
            barmode = 'group',
            xlabel = 'Type',
            ylabel = 'Quantity'
            )

    st.plotly_chart(contract_plot, use_container_width=True)
    st.markdown(
        '''
        The monthly renewal contract is the most frequent and the one with the highest proportion of Churn, the other 
        types (annual, and bi-annual) have a proportionally very low rate and can be key to increasing 
        retention.
        
        ### 3.4. Payment Method
        '''
    )

    pay = df.groupby(['Churn', 'PaymentMethod']).agg(Count = ('PaymentMethod', 'count')).reset_index()
    pay_plot = plot_bar(
            pay,
            x = 'PaymentMethod',
            y = 'Count',
            color = 'Churn',
            title = 'Payment Method x Churn',
            barmode = 'group',
            xlabel = 'Method',
            ylabel = 'Quantity'
        )
    st.plotly_chart(pay_plot, use_container_width=True)

    st.markdown(
        '''
        eCheck is the most used method and the one with the highest proportion of Churn, a factor that draws 
        attention is the very low number of cases in automatic payment methods, which is another 
        key point to plan actions to increase customer retention.

        ### 3.4. Relationship Time
        '''
    )
    tenure_plot = plot_hist(
            df,
            x = 'tenure',
            color = 'Churn',
            title = 'Churn Distribution by Relationship Time',
            xlabel = 'Months',
            ylabel = 'Quantity'
        )
    st.plotly_chart(tenure_plot, use_container_width=True)
    
    st.markdown(
        '''
        The longer we consume a service, whether out of convenience or attachment, the harder it is
        to leave it. But at the beginning of the contract, attention to details is greater, so after testing the hypothesis 
        of — more recent customers have a higher probability of becoming Churners, the behavior was confirmed.

        ### 3.5. Monthly Bill
        '''
    )
    charges_plot = plot_hist(
            df,
            x = 'MonthlyCharges',
            color = 'Churn',
            title = 'Churn Distribution by Monthly Fee Value',
            xlabel = 'USD',
            ylabel = 'Quantity'
        )
    st.plotly_chart(charges_plot, use_container_width=True)
    st.markdown(
        '''
        Just like the behavior regarding recent contracts, I also decided to test the hypothesis 
        of contracts with higher monthly bills being under higher probability of being a Churn case, 
        customers willing to pay for more expensive services will also demand better quality in their provision 
        — which became another confirmed hypothesis.

        ---
        
        ## 4. Recommendations
        With the analysis concluded, the recommendations for increasing customer retention were as follows:

        - Create actions for customer loyalty with special offers and discounts.
        - Seek an improvement in service and offer advantages in contracts with higher monthly bills.
        - Encourage the implementation of annual contract plans, and automatic payment methods.

        With this in mind, I also suggest creating marketing campaigns and new service plans at Telco, 
        some options would be (1) offer an annual plan with discount if the chosen payment method is 
        one of the automatic ones, and (2) review prices of contracts for consumers with longer relationship time and 
        offer advantages in renewal for longer duration plans. 

        ---

        ## 5. Conclusions
        The average retention in the sector is 69%[*](https://customergauge.com/blog/average-churn-rate-by-industry), a mark 
        surpassed by Telco Telecom, which shows good performance in the third quarter but despite this, several 
        improvement points were detected that can increase customer retention such as attention to longer duration plans, and 
        automatic payment methods. The good performance can be improved through retention actions for new customers 
        (contracts with less than 6 months), and transition of customers who currently have monthly renewal to longer plans.

        '''
    )

# -------- CHURN PREDICTOR ---------
with tab_pred:
    st.header("🤖 Contract Cancellation Predictor")
    st.subheader("Enter the Data and Calculate the Probability")

    # User inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior", ["Yes", "No"])
    partner = st.selectbox("Has partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Contract Time in Months", min_value = 1, max_value = df['tenure'].max(), value = 1)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streamingtv = st.selectbox("TV Streaming", ["Yes", "No", "No internet service"])
    streamingmovies = st.selectbox("Movie Streaming", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", list(df["Contract"].unique()))
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    paymethod = st.selectbox("Payment Method", list(df["PaymentMethod"].unique()))
    monthlycharge = st.slider("Monthly Charge", 20, 120)

    # Input dictionary
    input_features = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phoneservice,
        'MultipleLines': lines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesec,
        'OnlineBackup': onlinebackup,
        'DeviceProtection': deviceprotection,
        'TechSupport': techsupport,
        'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': paymethod,
        'MonthlyCharges': monthlycharge,
        'TotalCharges': tenure * monthlycharge
    }

    # Creating DataFrame with entered data
    input_df = pd.DataFrame(input_features, index = [0])

    # Mapping responses
    mapping = {
        'Masculino': 'Male',
        'Feminino': 'Female',
        'Sim': 'Yes',
        'Não': 'No',
        'Fibra ótica': 'Fiber optic',
        'Não possui linha': 'No phone service',
        'Não possui internet': 'No internet service'
    }

    input_df = input_df.applymap(lambda x: mapping.get(x, x))

    with st.container():
        if st.button("Result"):
            prob = model.predict_proba(input_df)[:,1][0]
            if prob > 0.40:
                st.markdown("## High Cancellation Potential")
                st.error(f"Probability of {prob:.2%}")
            else:
                st.markdown("## Low Cancellation Potential")
                st.success(f"Probability of {prob:.2%} ")

# ------------- ANALYTICAL DASHBOARD -------------
with tab_analytics:
    st.subheader("📊 Analytical Dashboard")
    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label = "Total Customers", value = df.shape[0])
    with col2:
        st.metric(label = "Churn Rate", value = f"{(df['Churn'].map({'No': 0, 'Yes': 1}).mean() * 100):.2f} %")
    with col3:
        st.metric(label = "Average Lifetime Value", value = f"$ {df['TotalCharges'].mean():.2f}")
    with col4:
        st.metric(label = "Average Contract Time", value = f"{df['tenure'].mean():.0f} months")

    # Charts
    col1, col2 = st.columns(2)

    pm = df[df['Churn'] == 'Yes'].groupby(['Churn', 'PaymentMethod']).agg(Quantity = ('PaymentMethod', 'count')).reset_index()
    ct = df[df['Churn'] == 'Yes'].groupby(['Churn', 'Contract']).agg(Quantity = ('Contract', 'count')).reset_index()

    with col1:
        st.plotly_chart(
            plot_bar(
                pm,
                x = 'PaymentMethod',
                y = 'Quantity',
                color = None,
                title = 'Churn by Payment Method',
                barmode = 'relative',
                xlabel='Method',
                ylabel='Quantity'
        ),
        use_container_width=True
        )
        st.plotly_chart(
            plot_hist(
            df,
            x = 'tenure',
            color = 'Churn',
            title = 'Churn by Relationship Time',
            xlabel = 'Months',
            ylabel = 'Quantity'
        ),
        use_container_width=True
        )
    with col2:
        st.plotly_chart(
            plot_bar(
                ct,
                x = 'Contract',
                y = 'Quantity',
                color = None,
                title = 'Churn by Contract Type',
                barmode = 'relative',
                xlabel='Type',
                ylabel='Quantity'
        ),
        use_container_width=True
        )
        st.plotly_chart(
            plot_hist(
            df,
            x = 'MonthlyCharges',
            color = 'Churn',
            title = 'Churn by Monthly Fee Value',
            xlabel = 'USD',
            ylabel = 'Quantity'
        ),
        use_container_width=True
        )