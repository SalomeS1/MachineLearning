from numpy import*


#-----Sigmoid------
def sigmoid(z):
    return 1/(1+e**(-z))

#---hypothesys-----
def h(x,theta):
    return dot(theta.transpose(),x)

#---Logistic Regression Cost Function-----
def lrCostFunction(theta, X, y, lambdaPar):
    #--igulisxmeba rom X matricis svetebi warmoadgenen training Data-s
    m=len(X.transpose)
    cost=-(dot(y.transpose(),log(h(X,theta)))+dot((1-y).transpose(),log(1-h(X,theta))))/m+lambdaPar*sum(theta[1:]**2)/(2*m)
    gradient=dot(X.transpose(),(sigmoid(h(X,theta))-y))/m+lambdaPar*sum(theta[1:])/m
    return [cost,gradient]



#----One-VS-All-----
def oneVsAll(X, y, num_labels, lambdaPar,max_iter,alpha):
    #---igulisxmeba rom X matricis striqonebi warmoadgenen training Data-s,(tito striqonze tito nomushia)
    theta=random.random(num_labels,len(X))
    for i in range (num_labels):
        for k in range(max_iter):
            theta-=alpha*lrCostFunction(theta,X,y,lambdaPar)[1]
    return theta



#---prediction1---
def predictOneVsAll(all_theta, X):
    predictions=sigmoid(h(X,all_theta))
    K=max(predictions)
    return where(predictions==K)[0][0]

#---prediction2----
def predict(Theta1, Theta2, X):
    HiddenLayer=sigmoid(h(X,Theta1))
    insert(HiddenLayer,0,random.rand())
    OutputLayer=sigmoid(h(HiddenLayer,Theta2))
    return where(OutputLayer==max(OutputLayer))[0][0]








