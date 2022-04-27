from pyexpat import model
import pandas as pd
import pickle
from sklearn import linear_model
import joblib


df=pd.read_csv("code/homeprices.csv")
print(df)

model=linear_model.LinearRegression()
model.fit(df[['area']],df.price)

print(model.predict([[5000]]))
#Saving model
with open('code/model_pickle','wb') as file:
    pickle.dump(model,file)

#Loading model
with open('code/model_pickle','rb') as file:
    mp = pickle.load(file)
print(mp.predict([[5000]]))

#Using Joblib
#Saving
joblib.dump(model, 'code/model_joblib')

# Loading Saved Model
mj = joblib.load('code/model_joblib')

print(mj.predict([[5000]]))