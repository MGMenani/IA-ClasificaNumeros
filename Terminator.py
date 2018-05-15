################## Imports section ###################
# -*- coding: cp1252 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
import os.path as path
import pickle
from PIL import Image   
from Tkinter import Tk
from tkFileDialog import askopenfilename
from math import sqrt
from sklearn.preprocessing import normalize 

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

from sklearn.model_selection import train_test_split

np.set_printoptions(threshold='nan')               #Para poder imprimir los arrays completos sin puntos suspensivos 

################ Load Data section ##################
def load_MNIST_Data():
    train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
    return train_img, train_lbl, test_img, test_lbl

################# Neural section ####################


class Neural_Network(object):
    def __init__(self):
        self.inputSize = 784                #Imágenes de 28x28 
        self.outputSize = 10                #números del 0 al 9
        self.hiddenSize1 = 1024              #Primera capa de 512
        self.hiddenSize2 = 512             #Segunda capa de 128
        self.learningRate = 0.0085
        self.probDrop = 0.5                 #Probabilidad de mantener unidad activa en dropout. Más alto = menos drop
        self.maskHidden1 = []               #Máscara con las posiciones para dropout de la capa oculta 1
        self.maskHidden2 = []               #Máscara con las posiciones para dropout de la capa oculta 2 

        #inicializo el W con pesos aleatorios con Xavier
        if path.exists("Pesos.pkl"):
            self.cargarPesos("Pesos.pkl")
        else:
            self.W1 = np.random.randn(self.inputSize, self.hiddenSize1) / sqrt(self.inputSize)# (784, 512) entrada
            self.W2 = np.random.randn(self.hiddenSize1, self.hiddenSize2) / sqrt(self.hiddenSize1)   # (512, 128) primera capa
            self.W3 = np.random.randn(self.hiddenSize2, self.outputSize) / sqrt(self.hiddenSize2)   # (128, 10)  segunda capa
        self.y = None
        
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.relu(self.z)                                          # activation function capa 1
        
        self.z3 = np.dot(self.z2,self.W2)
        self.z4 = self.relu(self.z3)                                         # activation function capa 2
        
        self.z5 = np.dot(self.z4,self.W3)                                    # final activation function
        output = self.softmax(self.z5)
        return output

    def backward(self, X, loss, output):
        self.output_delta = loss * self.grad_CrossEntropy_Softmax(output)
        
        self.z4_error = self.output_delta.dot(self.W3.T)
        self.z4_delta = self.z4_error*self.derivate_relu(self.z4)

        self.z2_error = self.z4_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.derivate_relu(self.z2)
        
        self.W1 += self.learningRate*(X.T.dot(self.z2_delta))                                    #set weights (input->hidden1)
        self.W2 += self.learningRate*(self.z2.T.dot(self.z4_delta))                              #set weights  (hidden1->hidden2)
        self.W3 += self.learningRate*(self.z4.T.dot(self.output_delta))                          #set weights  (hidden2->output

    def L2_regularization_cost(self, lambd):
        m = self.y.shape[1] # number of example
        return (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))*(lambd/(2*m))

    #Activation Function ReLU   
    def relu(self,x):
        return np.maximum(x, 0, x) #x * (x > 0) #np.maximum(x, 0, x)

    #ReLU Gradient
    def derivate_relu(self,x):
        return np.heaviside(x, 0)#1 * (x > 0) 
        
    #https://deepnotes.io/softmax-crossentropy
    def softmax(self, X):
        exp_scores = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    #Make log safe, avoid 0
    def safe_ln(x, minval=0.00001, maxval = 0.1):
        return np.log(np.clip(x,minval,maxval))

    #Cross entropy function, get loss
    def cross_entropy(self,p,y):
        return np.mean(np.sum(np.nan_to_num(-y * self.safe_ln(p) - (1 - y) * self.safe_ln(1 - p)), axis = 1)) #+ self.L2_regularization_cost(10)

    #Cross entropy gradient function
    def grad_CrossEntropy_Softmax(self,X):
        return self.y - X

    #Save the weights in the file 
    def guardarPesos(self):
        dic = {"W1":self.W1,"W2":self.W2,"W3":self.W3}
        with open("Pesos.pkl", "wb") as f:
            pickle.dump(dic,f,protocol=pickle.HIGHEST_PROTOCOL)

    #Load the weights from the file
    def cargarPesos(self, archivo):
        with open(archivo, "rb") as f:
            pesos = pickle.load(f)
            self.W1 = pesos["W1"]
            self.W2 = pesos["W2"]
            self.W3 = pesos["W3"]

    def Generate_Image_W(self):
        img = []
        vector_img = []
        imagen = self.W1.T
        for i in range (imagen.shape[0]):
            img = np.reshape(imagen[i], (28, 28))
            plt.imshow(img, cmap= plt.cm.binary)
            plt.savefig("Imagenes_W\\Image"+str(i)+".jpg")  

    #Clasify image loaded
    def clasificar(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.relu(self.z)                                          # activation function layer 1
        self.z3 = np.dot(self.z2,self.W2) 
        self.z4 = self.relu(self.z3)                                         # activation function layer 2
        self.z5 = np.dot(self.z4,self.W3)                                    # final activation function
        output = self.softmax(self.z5)
        return np.argmax(output)
        

def getRandomTesting(train_X,train_Y, porcentage):
    test_data = []
    test_label = []
    validation_data = []
    validation_label = []
    testDataSelected = random.sample(range(len(train_X)),int(len(train_X)*porcentage))     #Get the 80 percent of to train the model
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
    Y_vectorizado = np.zeros((len(Y), 10))              #Create the vectorized labels to send to cross entropy (10 columns->10 clases)
    for i in range(len(Y)):                     
        Y_vectorizado[i][int(Y[i])] = 1                 #Put 1.0 in the correct vector position
    return Y_vectorizado
    
def Train():
    loss = []
    accuracy = []
    
    cantTrain = 32          #Number of train images batch
    data = load_MNIST_Data()
    train_X = data[0]       #Trainning images (60000)
    train_Y = data[1]       #Trainning labels (60000)

    test_X = data[2]        #Test Images (10000)
    test_Y = data[3]        #Test Labels (10000)
    
    #se calcula el 80 del total de los datos,
    #retorna una lista con imagenes de train(80%) y sus labels y imagenes de validacion(20%) y sus labels
    epocs = 1
    NN = Neural_Network()
    while (True):
        opcion = menu()
        if opcion == 1:                                                                 #Entrenar el modelo
            for j in range(epocs):                                                      #Entrenar varias veces con las mismas imágenes (epochs)
                dataPorcentage = getRandomTesting(train_X,train_Y,0.8)

                train_X_aux = dataPorcentage[0]
                train_Y_aux = dataPorcentage[1]
    
                validation_X = dataPorcentage[2]                    
                validation_Y = dataPorcentage[3]
        
                print("EPOC #"+str(j))
                cont = 0
                for k in range(1500):                                                   #Iteraciones para entrenar con todas las imágenes
                    cont+=1
                    trainRandom = random.sample(range(len(train_X_aux)),cantTrain)      #toma los índices aleatoriamente para las imágenes de training
                    X = np.array([train_X_aux[i] for i in trainRandom]) / 255           #Tranning index of the before data
                    Y = [train_Y_aux[i] for i in trainRandom]                           #labels of the before values

                    #X = batchNormalization(X)

                    #Delete the positions that are used
                    train_X_aux = np.delete(train_X_aux, trainRandom, 0)
                    train_Y_aux = np.delete(train_Y_aux, trainRandom, 0)

                    #OneHotEncode
                    NN.y = OneHotEncode(Y)
                    output = NN.forward(X)
                    ce = NN.cross_entropy(output, NN.y)
                    NN.backward(X, ce, output)
                    if (k%100 == 0):
                        NN.y = OneHotEncode(validation_Y)
                        output = NN.forward(validation_X)
                        ce = NN.cross_entropy(output, NN.y)
                        efectividad = getAccuracy(output,NN.y)
                        loss += [ce]
                        accuracy += [efectividad]
                        print("Loss "+str(ce))
                        print("Exactitud "+str(getAccuracy(output,NN.y)))
                NN.guardarPesos()

            test_X  = test_X 
            NN.y = OneHotEncode(test_Y)
            print("Analisis")
            output = NN.forward(test_X)
            ce = NN.cross_entropy(output, NN.y)
            efectividad = getAccuracy(output,NN.y)
            loss += [ce]
            accuracy += [efectividad]
            print("Loss "+str(ce))
            print("Exactitud "+str(getAccuracy(output,NN.y)))

            plt.figure(1)
            plt.subplot(211)
            plt.plot(loss)
    
            plt.subplot(212)
            plt.plot(accuracy)
            plt.show()

            #Si se quiere que se comiencen a descarga las imagenes quitar el comentario
            #NN.Generate_Image_W()
            
        
        elif opcion == 2:                                           #Clasify the images make manually
            buscarPesos = Tk()
            buscarPesos.withdraw()
            pesos = askopenfilename(filetypes=[('.pkl files', '.pkl')], title = "Abra el archivo con los pesos")
            NN.cargarPesos(pesos)

            pathString = " "
            while pathString != "":
                Tk().withdraw()
                pathString = askopenfilename(initialdir = "ImagenesClasificar",filetypes=[('jpg files', '.jpg'),('png files','.png')], title = "Elija la imagen que desea clasificar")
                if pathString != "":
                    f = Image.open(pathString)
                    image = f.convert('L')                           #convert image to monochrome
                    image = np.array(image).ravel() / 255
                    NN.Generate_Image_W()
                    #Cuando las imágenes son negras con fondo blanco
                    print "Se clasificó un ",NN.clasificar(image)

        else:
            print "Opción incorrecta\n"


def menu():
    print "\n------- MENÚ -------"
    print "1) Entrenar modelo"
    print "2) Clasificar"
    print "Elija una opción:"
    return input()
    
Train()
        
