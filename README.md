# Telecom Churn — Customer Churn Prediction and Analysis
A data analysis and customer churn prediction project for the telecommunications industry using Streamlit, Machine Learning, and Data Visualization.

## 📜 Table of Contents
1. [About the Project](#-about-the-project)
2. [Technologies Used](#️-technologies-used)
3. [How to Run](#-how-to-run)
4. [Project Structure](#-project-structure)
5. [License](#️-license)
6. [Contact](#-contact)


## 📌 About the Project
This project aims to identify patterns and predict the probability of customer churn in a telecommunications company.

Through an interactive report and dashboard, users can explore the key factors influencing churn and generate predictions using a trained Machine Learning model.

## ⚙️ Technologies Used

This project was developed using: 
- **Python 3.12+** 
- **Streamlit(Interface)** 
- **Pandas & NumPy (Data Manipulation)** 
- **Scikit-learn, CatBoost (Machine Learning)**
- **Optuna, Feature Engine, Category Encoders (Optimization and Feature Engineering)** 
- **Plotly (Data Visualization)** 
- **Joblib (Model Handling)**


## 🚀 How to Run

Access the web application on Streamlit Cloud:
https://telcotelecom-churn.streamlit.app/

#### Prerequisites
-   Python 3.12+
-   Git

#### Execution
1️⃣ **Clone the repository**
``` bash
git clone https://github.com/datalopes1/telecom_churn.git
cd telecom_churn
```

2️⃣ **Create and activate a virtual environment (recommended)**
``` bash
python -m venv .venv
source .venv/bin/activate  # Mac and Linux
.venv\Scripts\activate  # Windows
```

3️⃣ **Install the dependencies**

``` bash
pip install -r requirements.txt
```

4️⃣ **Run the project**

``` bash
streamlit run app.py
```

## 📊 Project Structure
``` plaintext
telecom-churn/
│-- data/                       # Project data
|   ├── raw/                    # Raw data
|   ├── processed/              # Treated data
|-- models/                     # Trained models
|-- notebooks
|   ├── plots/                  # .png files generated in the EDA
|   ├── eda.ipynb               # Exploratory Data Analysis Notebook
|   ├── modeling.ipynb          # Machine Learning model construction Notebook
|-- scr/                        # Scripts
|   ├── __init__.py
|   ├── data_preprocessing.py   # Pre-processing functions script
|   ├── evaluate_model.py       # Model evaluation script
|   ├── predict.py              # Script to generate predictions
|   ├── train_model.py          # Model training script
|   ├── utils.py                # Script with auxiliary functions
|-- .gitignore                  # Files ignored by Git
|-- app.py                      # Streamlit application
|-- LICENSE.md                  # License
|-- poetry.lock                 # Poetry configuration and project dependencies
|-- pyproject.toml              # Exact versions of installed dependencies
|-- README.md                   # Project documentation
|-- requirements.txt            # Dependency list
```

## 🗒️ License
This project is licensed under the MIT License - see the file LICENSE.md
for more details.

## 📞 Contact

-   LinkedIn: https://www.linkedin.com/in/ngoc-bao-chan-le-7757651b5/
-   Portfolio: 
-   E-mail: lengocbaochan@gmail.com
