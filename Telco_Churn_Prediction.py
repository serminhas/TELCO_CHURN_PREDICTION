#A machine learning model that can predict customers leaving the company is expected.
#Telco churn data contains information about a fictitious telecom company providing
#home phone and Internet services to 7,043 California customers in the third quarter.
#Which customers have left, stayed or signed up for their services are shown.

# Customers who left within the last month – the column is called Churn
# Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# Demographic info about customers – gender, age range, and if they have partners and dependents

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import missingno as msno
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

# Exploratory Data Analysis

df_ = pd.read_excel("TELCO/Telco_customer_churn.xlsx")
df=df_.copy()
df.shape
df.isnull().sum()

# Data Preprocessing

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car=grab_col_names(df)

# df["Total Charges"].astype(int) => Error occured.
# df["Total Charges"]=pd.to_numeric(df["Total Charges"]) => Error occured. Because df["Total Charges"][2234] is " ".
# errors='coerce' => df["Total Charges"][2234] returns nan

df["Total Charges"]=pd.to_numeric(df["Total Charges"], errors='coerce')
df["Total Charges"].dtypes
df.describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.99]).T

# Outliers

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

outlier_thresholds(df, "Total Charges")
check_outlier(df, "Total Charges")
grab_outliers(df, "Total Charges", index=True)
replace_with_thresholds(df, "Total Charges")

df.describe([0.01, 0.05, 0.25, 0.75, 0.90, 0.99]).T

#Missing Values

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
df["Total Charges"].fillna(df["Total Charges"].mean(), inplace=True)
df.isnull().sum()

df_churn=df[df["Churn"]=="Yes"]

#Kontrol amaçlı

#df_nochurn=df[df["Churn"]=="No"]
#df_no_churn=df[~(df["Churn"]=="Yes")]
#df_no_churn[(df_no_churn["Internet Service"]=="No") & (df_no_churn["Phone Service"]=="No")].sum()
#(df_no_churn["Internet Service"]=="No").sum()
#(df_no_churn["Phone Service"]=="No").sum()
#df_no_churn[(df_no_churn["Multiple Lines"]=="No phone service") & (df_no_churn["Phone Service"]=="Yes")] => Empty DataFrame

#df_nochurn_noservice = df_no_churn[((df_no_churn["Phone Service"]!="Yes") | (df_no_churn["Multiple Lines"]== "No phone service")) & ((df_no_churn["Internet Service"]=="False") | (df_no_churn["Online Security"]=="No internet service") | (df_no_churn["Online Backup"]=="No internet service") | (df_no_churn["Device Protection"]=="No internet service") | (df_no_churn["Tech Support"]=="No internet service") | (df_no_churn["Streaming TV"]=="No internet service") | (df_no_churn["Streaming Movies"]=="No internet service"))]
#df_nochurn_noservice.shape

#############################################
# Feature Engineering
#############################################

#Categorical Columns
def target_vs_othercols(dataframe, target, cat_cols):
    temp_df = dataframe.copy()
    for col in cat_cols:
        print(pd.DataFrame({"Count": temp_df.groupby(col)[target].count(),
                            "Ratio": (temp_df.groupby(col)[target].count()/ temp_df[col].shape[0])*100}), end="\n\n\n")

target_vs_othercols(df_churn, "Churn", cat_cols)

#Numerical Columns

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df_churn, "Churn", col)

#Feature Extraction

# Senior_Partner_Dependents
df.loc[((df['Senior Citizen'] == "No") & (df['Partner'] == "No") & (df['Dependents'] == "No")), "NEW_NO_SENIOR_NO_PD"] = "Yes"
df.loc[(~((df['Senior Citizen'] == "No") & (df['Partner'] == "No") & (df['Dependents'] == "No"))), "NEW_NO_SENIOR_NO_PD"] = "No"

# Services
df.loc[((df['Phone Service'] == "Yes") | (df['Internet Service'] == "Fiber optic")), "NEW_SERVICES_PHONE_FIBER"] = "Yes"
df.loc[(~((df['Phone Service'] == "Yes") | (df['Internet Service'] == "Fiber optic"))), "NEW_SERVICES_PHONE_FIBER"] = "No"

# Tech Services
#df.loc[((df['Internet Service'] == "Yes") & ((df['Online Security'] == "No") | (df['Online Backup'] == "No") | (df['Device Protection'] == "No") | (df['Tech Support'] == "No"))), "NEW_TECH_SERVICES"] = "Yes"
#df.loc[(~((df['Internet Service'] == "Yes") & ((df['Online Security'] == "No") | (df['Online Backup'] == "No") | (df['Device Protection'] == "No") | (df['Tech Support'] == "No")))), "NEW_TECH_SERVICES"] = "No"

# Contract Terms
df.loc[((df['Contract'] == "Month-to-month") & ((df['Paperless Billing'] == "Yes") | (df['Payment Method'] == "Electronic check"))), "NEW_CONTRACT_TERMS"] = "Yes"
df.loc[(~((df['Contract'] == "Month-to-month") & ((df['Paperless Billing'] == "Yes") | (df['Payment Method'] == "Electronic check")))), "NEW_CONTRACT_TERMS"] = "No"

#Label Encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#One_Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

categorical_cols = [col for col in df.columns if 5 >= df[col].nunique() > 2]

df=one_hot_encoder(df, categorical_cols)

#Rare Encoding

"""df_churn["Tenure Months"].value_counts(ascending=False)[50:60] #Till 12 months
df_churn["Monthly Charges"].value_counts(ascending=False).head(5) #Till 100
df_churn["Total Charges"].value_counts(ascending=False).head(10) # ==3706.12

df_churn["Tenure Months Qcut"] = pd.qcut(df["Tenure Months"], 10)
df_churn["Tenure Months Qcut"].value_counts(ascending=False)

df_churn["Monthly Charges Qcut"] = pd.qcut(df["Monthly Charges"], 10)
df_churn["Monthly Charges Qcut"].value_counts(ascending=False)

df_no_churn["Total Charges"]=df_no_churn["Total Charges"].astype(int)
df_no_churn["Tenure Months"]=df_no_churn["Tenure Months"].astype(int)
num_cols=num_cols+df_no_churn["Total Charges"]

def rare_encoder(dataframe, num_cols):
    temp_df = dataframe.copy()

    rare_columns = [col for col in num_cols if (((col == "Tenure Months") & (col>69.0)) |
                                                ((col == "Monthly Charges") & (18.249 < col < 20.05)))]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp.index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])"""

"""# StandardScaler

ss = StandardScaler()

df["Tenure_Months_standard_scaler"] = ss.fit_transform(df[["Tenure Months"]])
df["Monthly_Charges_standard_scaler"] = ss.fit_transform(df[["Monthly Charges"]])
df["Total_Charges_standard_scaler"] = ss.fit_transform(df[["Total Charges"]])

df["Tenure_Months_standard_scaler"].value_counts(ascending=False).head(5)
df["Monthly_Charges_standard_scaler"].value_counts(ascending=False).head(5)
df["Total_Charges_standard_scaler"].value_counts(ascending=False).head(5)"""

#MinMaxScaler

mms = MinMaxScaler()

df["Tenure_Months_min_max_scaler"] = mms.fit_transform(df[["Tenure Months"]])
df["Monthly_Charges_min_max_scaler"] = mms.fit_transform(df[["Monthly Charges"]])
df["Total_Charges_min_max_scaler"] = mms.fit_transform(df[["Total Charges"]])

df["Tenure_Months_min_max_scaler"].value_counts(ascending=False).head(5)
df["Monthly_Charges_min_max_scaler"].value_counts(ascending=False).head(5)
df["Total_Charges_min_max_scaler"].value_counts(ascending=False).head(5)

#Model

y = df["Churn"]
X = df.drop(["CustomerID", "Gender", "Churn"], axis=1)
y.info()
X.info()
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7373289204142204
cv_results['test_f1'].mean()
# 0.5067056780852977
cv_results['test_roc_auc'].mean()
# 0.6653107864141297

# Hyperparameter Optimization with GridSearchCV

cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_best_grid.best_params_

cart_best_grid.best_score_

random = X.sample(1, random_state=45)

cart_best_grid.predict(random)

# Final Model

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7946906856893993
cv_results['test_f1'].mean()
# 0.5815809319843425
cv_results['test_roc_auc'].mean()
# 0.8234468076875624

# Feature Importance

cart_final.feature_importances_

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X, num=5)


