################## Imports section ###################
# -*- coding: cp1252 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt
from sklearn.preprocessing import normalize 

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

from sklearn.model_selection import train_test_split

################ Load Data section ##################
def load_MNIST_Data():
    train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
    return train_img, train_lbl, test_img, test_lbl

################# Neural section ####################


class Neural_Network(object):
    def __init__(self):
        self.inputSize = 784                #Imágenes de 28x28 
        self.outputSize = 10                #números del 0 al 9
        self.hiddenSize1 = 512              #Primera capa de 512
        self.hiddenSize2 = 128              #Segunda capa de 128
        self.learningRate = 0.0085

        #inicializo el W con pesos aleatorios con Xavier 
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize1) / sqrt(self.inputSize)# (784, 512) entrada
        self.W2 = np.random.randn(self.hiddenSize1, self.hiddenSize2) / sqrt(self.hiddenSize1)   # (512, 128) primera capa
        self.W3 = np.random.randn(self.hiddenSize2, self.outputSize) / sqrt(self.hiddenSize2)   # (128, 10)  segunda capa
        self.y = None
        
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.relu(self.z)                     # activation function capa 1
        self.z3 = np.dot(self.z2,self.W2) 
        self.z4 = self.relu(self.z3)                    # activation function capa 2
        self.z5 = np.dot(self.z4,self.W3)               # final activation function
        output = self.softmax(self.z5)
        return output

    def backward(self, X, loss, output):
        self.output_delta = loss * self.grad_CrossEntropy_Softmax(output)
        
        self.z4_error = self.output_delta.dot(self.W3.T)
        self.z4_delta = self.z4_error*self.derivate_relu(self.z4)

        self.z2_error = self.z4_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.derivate_relu(self.z2)
        
        self.W1 += self.learningRate*(X.T.dot(self.z2_delta))                                    #ajusta pesos (input->hidden1)
        self.W2 += self.learningRate*(self.z2.T.dot(self.z4_delta))                              #ajusta pesos (hidden1->hidden2)
        self.W3 += self.learningRate*(self.z4.T.dot(self.output_delta))                          #ajusta pesos (hidden2->output)
        
    def relu(self,x):
        return np.maximum(x, 0, x) #x * (x > 0) #np.maximum(x, 0, x)

    def derivate_relu(self,x):
        return 1 * (x > 0) #np.heaviside(x, 0)
        
    #https://deepnotes.io/softmax-crossentropy
    def softmax(self, X):
        exp_scores = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
##        X -= np.max(X)
##        exps = np.exp(X) + np.finfo(float).eps                     #calcula cada e**Xi (de cada elemento de la matriz)
##        suma = np.sum(exps,1) 
##        return exps / suma[:,None]
        
    
    def cross_entropy(self,p,y):
        return np.mean(np.sum(np.nan_to_num(-y * np.log(p) - (1 - y) * np.log(1 - p)), axis = 1))
##        m = y.shape[0]
##        log_likelihood = -np.log(p)
##        loss = np.sum(log_likelihood) / m
##        return loss

    def grad_CrossEntropy_Softmax(self,X):
##        num_examples = X.shape[0]
##        probs = X
##        probs[range(num_examples), np.argmax(self.y)] -= 1
##        return probs
        return self.y - X

def getRandomTesting(train_X,train_Y, porcentage):
    test_data = []
    test_label = []
    validation_data = []
    validation_label = []
    testDataSelected = random.sample(range(len(train_X)),int(len(train_X)*porcentage))     #Selecciona el 80 por ciento para entrenar
    test_data = np.take(train_X, testDataSelected, 0)
    test_label = np.take(train_Y, testDataSelected, 0)
    validation_data = np.delete(train_X, testDataSelected, 0)
    validation_label = np.delete(train_Y, testDataSelected, 0)
    return test_data, test_label, validation_data, validation_label

def getAccuracy(X, Y):
    aciertos = 0
    for i in range(len(X)):
        if (np.argmax(X[i]) == np.argmax(Y[i])):
            aciertos += 1
    return float((aciertos*100)/len(X))

def OneHotEncode(Y):
    #One Hot Encoding
    Y_vectorizado = np.zeros((len(Y), 10))              #Creacion de labels vectorizados para mandarlos a cross-entropy (10 columnas->10 clases)
    for i in range(len(Y)):                     
        Y_vectorizado[i][int(Y[i])] = 1                 #Se pone 1.0 en la posicion del vector
    return Y_vectorizado
    
def Train():
    cantTrain = 32        #Número de imágenes del train que se usarán como test
    data = load_MNIST_Data()
    train_X = data[0]       #imagenes de entrenamiento (60000)
    train_Y = data[1]       #Labeld de entrenamiento (60000)
    test_X = data[2]        #Imagenes de prueba (10000)
    test_Y = data[3]        #Labels de prueba (10000)
    
    #se calcula el 80 del total de los datos,
    #retorna una lista con imagenes de train(80%) y sus labels y imagenes de validacion(20%) y sus labels
    dataPorcentage = getRandomTesting(train_X,train_Y,0.8)
    
    train_X = dataPorcentage[0]
    train_X_aux = dataPorcentage[0]
    train_Y = dataPorcentage[1]
    train_Y_aux = dataPorcentage[1]
    
    validation_X = dataPorcentage[2]                    
    validation_Y = dataPorcentage[3] 
    epocs = 6
    NN = Neural_Network()
    for i in range(epocs):
        print("EPOC #"+str(i))
        train_X = train_X_aux
        train_Y = train_Y_aux
        
        for i in range(1500):
            #print("Iteracion #"+str(i))
            trainRandom = random.sample(range(len(train_X)),cantTrain)                       #toma los índices aleatoriamente para las imágenes de training
            X = np.array([train_X[i] for i in trainRandom]) / 255                                 #Datos de testing con los índices anteriores
            Y = [train_Y[i] for i in trainRandom]                                            #labels de los datos anteriores

            #Elimina las posiciones que ya fueron utilizadas
            train_X = np.delete(train_X, trainRandom, 0)
            train_Y = np.delete(train_Y, trainRandom, 0)

            #OneHotEncode
            NN.y = OneHotEncode(Y)
        
            output = NN.forward(X)
            
            #print("Exactitud "+str(getAccuracy(output,NN.y)))
            #print("Forward")
            #print(output)

            ce = NN.cross_entropy(output, NN.y)
            #print ce
            
            NN.backward(X, ce, output)


    test_X  = test_X / 255
    NN.y = OneHotEncode(test_Y)
    print("Analisis final")
    output = NN.forward(test_X)
    ce = NN.cross_entropy(output, NN.y)
    print("Loss "+str(ce))
    print("Exactitud "+str(getAccuracy(output,NN.y)))
    



    
Train()
        
