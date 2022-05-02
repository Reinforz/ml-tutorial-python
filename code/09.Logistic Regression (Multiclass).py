from operator import mod
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

digits=load_digits()
model=LogisticRegression()

X_train, X_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.2)

model.fit(X_train,y_train)

print(model.score(X_test,y_test))

#Now finding the incorrect predictions using confusion matrix

y_test_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_test_pred)
print(cm)
