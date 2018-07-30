from numpy import *

#---Sigmoid----
def g(z):
    return 1/(1+e**(-z))

#---gradSIgmoid--
def sigmoidGradient(z):
    return g(z)*(1-g(z))

#---randomly initialize weights--
def randominItialize(L_in,L_out):
    return random.random((L_out,L_in+1))

#---CostFunction and Backprop for ANN---
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambdaPar):
    theta1=randominItialize(input_layer_size,hidden_layer_size)
    theta2=randominItialize(hidden_layer_size,num_labels)

    X=concatenate((ones(shape=len(X)),X),axis=1)
    z1=X
    a2=dot(theta1,z1)
    z2=sigmoid(a2)
    a3=dot(theta2,z2)
    output=sigmoid(a3)

    cost=sum(sum(-(dot(y,log(output))+dot((1-y),log(1-output)))))+lambdaPar*(sum(sum(theta1**2))+sum(sum(theta2**2)))/len(X)

    delta3=output-y
    delta2=dot(theta2.transpose(),delta3)*sigmoidGradient(a3)

    theta2_gradient=(dot(delta3,output)+lambdaPar*theta2)/len(X)
    theta1_gradient=(dot(delta2,z2)+lambdaPar*theta1)/len(X)

    theta2_gradient[:,1]=dot(delta3,output)/len(X)
    theta1_gradient[:,1]=dot(delta2,output)/len(X)


    return [cost,[theta1_gradient,theta2_gradient]]
    
    
    
    
