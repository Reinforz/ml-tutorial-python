import pandas as pd
import numpy as np
from sklearn import linear_model
import math
df=pd.read_csv("code/houseprices.csv")
print(df)

medianBedrooms=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(medianBedrooms)
print(df)

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.predict([[3000,3,40]]))