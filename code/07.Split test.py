from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("code/07.Carprices.csv")
df.head()


# The approach we are going to use here is to split available data in two sets
# Training: We will train our model on this dataset
# Testing: We will use this subset to make actual predictions using trained model
# The reason we don't use same training set for testing is because our model has seen those samples before, using same samples for making predictions might give us wrong impression about accuracy of our model. It is like you ask same questions in exam paper as you tought the students in the class.



X = df[['Mileage','Age(yrs)']]
Y = df['Sell Price($)']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2) 

clf = LinearRegression()
clf.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
X_test

clf.predict(X_test)

print(clf.score(X_test, y_test))

