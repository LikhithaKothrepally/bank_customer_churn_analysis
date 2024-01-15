from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from xgboost import XGBClassifier
import pickle

# Read and preprocess the data
df = pd.read_csv("web_app/bank_customer_churn.csv")

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.drop('RowNumber', axis=1, inplace=True)
df.drop('CustomerId', axis=1, inplace=True)
df.drop('Surname', axis=1, inplace=True)

df['Gender'] = df['Gender'].astype('category')
df['Gender'] = df['Gender'].cat.codes

df['Active Member'] = df['Active Member'].astype('category')
df['Active Member'] = df['Active Member'].cat.codes

df['Credit Card'] = df['Credit Card'].astype('category')
df['Credit Card'] = df['Credit Card'].cat.codes

df['Balance'] = df['Balance'].astype('int64')
df['EstimatedSalary'] = df['EstimatedSalary'].astype('int64')

df = df[(df['Age'] >= 18) & (df['Age'] <= 70)]
df = df[df['EstimatedSalary'] > 5000]
df = df[df['CreditScore'] > 500]
df = df[df['Age'] < 50]
df = df[df['Age'] > 25]
df = pd.get_dummies(df, columns=['Geography'])

num_features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                'Credit Card', 'Active Member', 'Geography_France', 'Geography_Germany', 'Geography_Spain']

# Scale the numerical features
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Save the scaler object to a pickle file
pickle.dump(scaler, open('scaler.pkl', 'wb'))

smote = SMOTE()

# Split the data into training and test sets
X = df.drop('Exited', axis=1)
y = df['Exited']

# balance the dataset
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the XGBoost classifier
xgb = XGBClassifier()
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
scores = cross_val_score(xgb, X_train, y_train, cv=kfold)
print(scores)

xgb.fit(X_train, y_train)

# Save the XGBoost model to a pickle file
pickle.dump(xgb, open('xgb_model.pkl', 'wb'))
