from scipy.io import loadmat
import numpy as np
import pandas as pd
import colorsys as cs
import statistics as st
from sklearn.svm import LinearSVC
import pickle

mat  = loadmat("../fpp/sat-4-full.mat")

train_x = np.array(mat['train_x'])
train_y = mat['train_y']

test_x = np.array(mat['test_x'])
test_y = mat['test_y']

bl = train_y[0]
tr = train_y[1]
gr = train_y[2]
no = train_y[3]

"""[find_class i lst] returns the class of an image. [i] is the image index."""
def find_class(i, bl, tr, gr, no):
  if bl[i] == 1:
    # return 'barren land'
    return 0
  elif tr[i] == 1:
    # return 'trees'
    return 1
  elif gr[i] == 1:
    # return 'grassland'
    return 2
  else:
    # return 'none'
    return 3

def get_img_data(img_n,f):
  hsv = []
  clss = []
  for k in range(img_n):
    cl = find_class(k,train_y[0],train_y[1],train_y[2],train_y[3])
    h,s,v = [],[],[]
    for i in range(28):
      for j in range(28):
        x = cs.rgb_to_hsv(train_x[i,j,0,k],train_x[i,j,1,k],train_x[i,j,2,k])
        h.append(x[0])
        s.append(x[1])
        v.append(x[2])
    hsv.append([f(h),f(s),f(v)])
    clss.append(cl)
  return hsv,clss

def get_svm(Xs, Ys):
  clf = LinearSVC(random_state=0)
  clf.fit(Xs,Ys)
  # print(clf.coef_)
  return clf 

x = get_img_data(0,20)
f = open("mod.dat", "wb")
model = get_svm(x[0],x[1])
pickle.dump(model, f)
f.close()

def tryd(f, x):
  return f(x)

def get_test_data(img_n,f):
  hsv = []
  for k in range(img_n):
    h,s,v = [],[],[]
    for i in range(28):
      for j in range(28):
        x = cs.rgb_to_hsv(test_x[i,j,0,k],test_x[i,j,1,k],test_x[i,j,2,k])
        h.append(x[0])
        s.append(x[1])
        v.append(x[2])
    hsv.append([f(h),f(s),f(v)])
  return hsv

def tester(svm, data):
  predictions = svm.predict(data)
  observations = []
  correct = 0
  for i in range(len(predictions)):
    observations.append(find_class(i,test_y[0],test_y[1],test_y[2],test_y[3]))
  observations = np.array(observations)
  for j in range(len(observations)):
    if observations[j] == predictions[j]:
      correct+=1
  return float(correct/len(observations))






# 1000 -> barren land -> 0
# 0100 -> trees -> 1
# 0010 -> grassland -> 2
# 0001 -> none -> 3



