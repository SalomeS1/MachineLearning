from numpy import *

#-----Sigmoid------
def sigmoid(z):
    return 1/(1+e**(-z))

#---hypothesys-----
def h(x,theta):
    return dot(theta.transpose(),x)

#----CostFunction with regulization---
def costFunctionReg(theta, X, y, lambdaPar):
    # igulisxmeba rom X matricis  striqonebi warmoadgenen training Data-s
    m=len(X.transpose())
    return -(dot(y.transpose(),log(h(X,theta)))+dot((1-y).transpose(),log(1-h(X,theta))))/m+lambdaPar*sum(theta[1:]**2)/(2*m)


#---Prediction---
def predict(theta, X):
    if(sigmoid(dot(theta.transpose(),X))>=0.5):
        return 1
    return 0


