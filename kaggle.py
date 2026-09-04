import pandas as pd

train_transaction = pd.read_csv('ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('ieee-fraud-detection/train_identity.csv')

print(train_transaction.shape)
print(train_transaction.head())