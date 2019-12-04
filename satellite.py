import pandas as pd 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle

test_mat  = loadmat("../fpp/test_x_only.mat")

p = open("mod.dat", "rb")
model = pickle.load(p)

