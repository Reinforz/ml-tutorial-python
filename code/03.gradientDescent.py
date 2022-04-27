import numpy as np

def gradientDescent(x,y):
  mCurr=bCurr=0
  iterations=1000
  n=len(x )
  learningRate=0.08

  for i in range(iterations):
    yPredicted=mCurr*x+bCurr
    cost=(1/n)*sum([val**2 for val in (y-yPredicted)])
    md = -(2/n)*sum(x*(y-yPredicted))
    bd = -(2/n)*sum(y-yPredicted)
    mCurr =mCurr - learningRate * md
    bCurr = bCurr - learningRate * bd
    print ("m {}, b {}, cost {} iteration {}".format(mCurr,bCurr,cost, i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradientDescent(x,y)