import sys,os
sys.path.append(os.pardir)
from mnist_dataset.mnist import load_mnist
# It takes a long time when you run the mnist.py file for the first time, 
# but it doesn't take long because you import 
# the local pickle from the second time
import numpy as np
from PIL import Image
import pickle
import optimizer 
import loss_function

class LeNet():
    