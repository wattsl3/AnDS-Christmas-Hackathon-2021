# pylint: disable=no-member
""" Master Script"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

URL = (
    "https://raw.githubusercontent.com/wattsl3/AnDS-Christmas-Hackathon"
    "/main/Car_Insurance_Claims.csv"
)
data = pd.read_csv(URL)
data = pd.DataFrame(data)

# show first rows of data
data.head()

# show summary of data
data.describe()
# credit score and mileage will have nulls

# show columns with null value counts
data.isnull().sum()
# as expected, mileage and credit score have almost 10% null values

# replace null values with averages from columns
data["CREDIT_SCORE"] = data["CREDIT_SCORE"].fillna(data["CREDIT_SCORE"].mean())
data["ANNUAL_MILEAGE"] = data["ANNUAL_MILEAGE"].fillna(data["ANNUAL_MILEAGE"].mean())
data.describe()

# show credit score distribution by whether there has been an insurance claim
sns.displot(data=data, x="CREDIT_SCORE", col="OUTCOME", kde=True)

# show speeding violations by income level
sns.displot(data=data, x="SPEEDING_VIOLATIONS", col="INCOME", kde=True)
# upper class category has more people overall but have received more speeding violations

# credit score distribution by income levels
sns.catplot(data=data, kind="box", x="INCOME", y="CREDIT_SCORE")

# strip plot showing credit score distributions by income levels broken down further by outcome
sns.stripplot(data=data, x="INCOME", y="CREDIT_SCORE", hue="OUTCOME", linewidth=1)

# strip plot showing credit score distributions by education levels broken down further by outcome
sns.stripplot(data=data, x="EDUCATION", y="CREDIT_SCORE", hue="OUTCOME", linewidth=1)

# distribution of car mileage broken down by whether a claim was filed
sns.catplot(data=data, kind="box", x="OUTCOME", y="ANNUAL_MILEAGE")

# convert categorical variables to numeric
le = preprocessing.LabelEncoder()
data["AGE"] = le.fit_transform(data["AGE"])
data["GENDER"] = le.fit_transform(data["GENDER"])
data["RACE"] = le.fit_transform(data["RACE"])
data["DRIVING_EXPERIENCE"] = le.fit_transform(data["DRIVING_EXPERIENCE"])
data["EDUCATION"] = le.fit_transform(data["EDUCATION"])
data["INCOME"] = le.fit_transform(data["INCOME"])
data["VEHICLE_YEAR"] = le.fit_transform(data["VEHICLE_YEAR"])
data["VEHICLE_TYPE"] = le.fit_transform(data["VEHICLE_TYPE"])


def clean_dataset(dataframe):
    """
    Clean dataset of NaN or Inf to avoid ValueError when training model later
    """
    assert isinstance(dataframe, pd.DataFrame)
    dataframe.dropna(inplace=True)
    indices_to_keep = ~dataframe.isin([np.nan, np.inf, -np.inf]).any(1)
    return dataframe[indices_to_keep].astype(np.float64)


data = clean_dataset(data)

# drop ID and postal_code variables for now
# separate OUTCOME into target class variable
target = data["OUTCOME"]
data = data.drop("ID", axis=1)
data = data.drop("POSTAL_CODE", axis=1)
data = data.drop("OUTCOME", axis=1)

# scale remaining numeric features
scaler = StandardScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), columns=data.columns)

data.head()

# split data for classification
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=333
)

# use Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# check model accuracy
print("Overall Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("ROC AUC:", metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print(metrics.classification_report(y_test, y_pred))

# plot ROC
metrics.plot_roc_curve(model, X_test, y_test)
plt.show()
