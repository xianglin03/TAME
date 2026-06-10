import pandas as pd

path = 'data/Churn_Modelling/Churn_Modelling.csv'

df = pd.read_csv(path)
df = df.drop(columns=['Surname'])
df.to_csv(path, index=False)