import pandas as pd

df = pd.read_csv('credit-card-fraud/creditcard.csv')
print(df.shape)
print(df['Class'].value_counts())