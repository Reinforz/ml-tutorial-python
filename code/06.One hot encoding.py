import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#creating csv->dataframe
df = pd.read_csv("code/06.homeprices.csv")
# print(df)
#creating dummy variable for one hot encoding
dummies = pd.get_dummies(df.town)
# one hot encoded
merged = pd.concat([df,dummies],axis='columns')
# print(merged)

final = merged.drop(['town'], axis='columns')
#avoiding the dummy variable trap
final = final.drop(['west windsor'], axis='columns')

#fitting data to train the model
X = final.drop('price', axis='columns')
Y = final.price
model = LinearRegression()
model.fit(X,Y)
print(model.predict([[2800,0,1]]))

#Visit the below link to see hot to use one hot coding directly using sklearn library. But it seemed cumbersome to me.
#https://github.com/codebasics/py/blob/master/ML/5_one_hot_encoding/one_hot_encoding.ipynb