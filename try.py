import pandas as  pd


data = pd.read_csv('data/loan_data.csv')

features = data.copy()
labels = features.pop("not.fully.paid")

discrete_features = features.dtypes == int
print(discrete_features)