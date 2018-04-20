import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(
 mnist.data, mnist.target, test_size=1/7.0, random_state=0)


x = np.random.random((5,5)) - 0.5

def relu(x):
    return np.maximum(x, 0, x)
    
print(relu(x))


'''
def fun():
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        mnist.data, mnist.target, test_size=1/7.0, random_state=0)

fun()'''
