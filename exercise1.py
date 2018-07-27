from numpy import *
from matplotlib.pyplot import plot

#---------warmUp----------
def Warmup():
    A=eye(5)
    print(A)

#--------plotData----------
def PlotData():
    text=loadtxt('ex1data1.txt',delimiter=',')
    x=text[:,0]
    y=text[:,1]
    plot(x,y,'r+')

#----hypothesys-----

def h(x,theta):
    return dot(theta.transpose(),x)


#----GradientDescent-------

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    iter=0
    print(X)
    delta=dot((h(X,theta)-y),X)/m
    while(iter<num_iters):
        theta-=alpha*delta
        iter+=1


#------CostFunction--------

def computeCostMulti(X, y, theta):
    X = stack(ones(shape=X.shape), X, axis=1)
    m=len(X)
    return (1/(2*m))*dot((dot(X.transpose(),theta)-y).transpose(),(dot(X.transpose(),theta)-y))



#-----FeatureNormalization--
def featureNormalize(X):
    return (X-average(X))/std(X)


#-----Normal Equation------

def normalEqn(X, y):
    theta=dot(dot(linalg.inv((dot(X.transpose,X))),X.transpose()),y)
    return theta


