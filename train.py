from scipy.io import loadmat
import numpy as np
import pandas as pd
import colorsys as cs
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pickle
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

mat  = loadmat("../fpp/sat-4-full.mat")

train_x = np.array(mat['train_x'])
train_y = mat['train_y']

test_x = np.array(mat['test_x'])
test_y = mat['test_y']

bl = train_y[0]
tr = train_y[1]
gr = train_y[2]
no = train_y[3]

f = np.mean
f1 = np.var

"""[find_class i lst] returns the class of an image. [i] is the image index."""
def find_class(i, bl, tr, gr, no):
  if bl[i] == 1:
    return 0
  elif tr[i] == 1:
    return 1
  elif gr[i] == 1:
    return 2
  else:
    return 3

def evi(nir,r,g,b):
  return float(2.5 * (nir - r) / (nir + (6*r) - (7.5*b) + 1))

def get_img_data(img_n,f):
  hsv = []
  clss = []
  for k in range(img_n):
    cl = find_class(k,train_y[0],train_y[1],train_y[2],train_y[3])
    h,s,v,nir,ev = [],[],[],[],[]
    for i in range(28):
      for j in range(28):
        r,g,b,n = train_x[i,j,0,k],train_x[i,j,1,k],train_x[i,j,2,k],train_x[i,j,3,k]
        x = cs.rgb_to_hsv(r,g,b) 
        h.append(x[0])
        s.append(x[1])
        v.append(x[2])
        nir.append(n)
        ev.append(evi(n,r,g,b))
    # hsv.append([f(np.array(h)),f(np.array(s)),f(np.array(v))])
    # hsv.append([f(np.array(h)),f(np.array(s)),f(np.array(v)), 
    # f1(np.array(h)),f1(np.array(s)),f1(np.array(v))])
    hsv.append([f(np.array(s)), f(np.array(v)), 
    f1(np.array(h))+f1(np.array(s))+f1(np.array(v)), f(np.array(ev))])
    clss.append(cl)
  return hsv,clss

def get_test_data(img_n,f):
  hsv = []
  for k in range(img_n):
    h,s,v,nir,ev = [],[],[],[],[]
    for i in range(28):
      for j in range(28):
        r,g,b,n = test_x[i,j,0,k],test_x[i,j,1,k],test_x[i,j,2,k],test_x[i,j,3,k]
        x = cs.rgb_to_hsv(r,g,b) 
        h.append(x[0])
        s.append(x[1])
        v.append(x[2])
        nir.append(n)
        ev.append(evi(n,r,g,b))
    # hsv.append([f(h),f(s),f(v),f1(h),f1(s),f1(v)])
    hsv.append([f(np.array(s)), f(np.array(v)), 
    f1(np.array(h))+f1(np.array(s))+f1(np.array(v)), f(np.array(ev))])
  return hsv

def get_svm(Xs, Ys):
  # clf = LinearSVC(random_state=0)
  clf = SVC(C=1.0, kernel='rbf')
  clf.fit(Xs,Ys)
  # print(clf.coef_)
  return clf 

correct_class = {0: 0, 1:0, 2:0, 3:0}
wrong_class = {0: 0, 1:0, 2:0, 3:0}

def tester(svm, data):
  predictions = svm.predict(data)
  observations = []
  correct = 0
  preds = {0: 0, 1:0, 2:0, 3:0}
  for i in range(len(predictions)):
    observations.append(find_class(i,test_y[0],test_y[1],test_y[2],test_y[3]))
    preds[predictions[i]] += 1
  observations = np.array(observations)
  for j in range(len(observations)):
    x = observations[j]
    if x == predictions[j]:
      correct+=1
      correct_class[x] += 1
    else:
      wrong_class[x] += 1
  print("Correct classification:")
  print(correct_class)
  print("Wrong classification:")
  print(wrong_class)
  return float(correct/len(observations)), preds

def truncate(n):
  return float(int(n * 1000) / 1000)

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 0.05*height, 
                str(truncate(values[i])),
                ha='center', va='bottom')


def plot_it(values, file, title):
  b1 = plt.bar(labels, values, label= "Correct Classification")
  plt.legend(loc='upper right')
  plt.title(title)
  plt.ylabel("Percentage of Images")
  autolabel(b1)
  plt.savefig(file)
  plt.clf()
  plt.cla()
  plt.close()

def get_init_scat_data(x, feat1,feat2):
  dat = []
  for i in range(len(x[0])):
    dat.append(([x[0][i][feat1],x[0][i][feat2]], x[1][i]))
  return dat

def get_var_sum(x, k):
  dat = []
  for i in range(len(x[0])):
    dat.append(([x[0][i][3]+x[0][i][4]+x[0][i][5], x[0][i][k] ], x[1][i]))
  return dat

def generate_scatter_data(data):
  bl,tr,gr,no = [],[],[],[]
  for tup in data:
    if tup[1] == 0:
      bl.append(tup[0])
    elif tup[1] == 1:
      tr.append(tup[0])
    elif tup[1] == 2:
      gr.append(tup[0])
    elif tup[1] == 3:
      no.append(tup[0])
  return bl,tr,gr,no

def make_scat(lst, lab):
  X, y = [],[]
  for coor in lst:
    X.append(coor[0])
    y.append(coor[1])
  plt.scatter(X,y, label= lab )
  plt.legend(loc= 'upper right')

def many_scat(tup,labs, x_lab, y_lab, title):
  make_scat(tup[0],labs[0])
  make_scat(tup[1], labs[1])
  make_scat(tup[2], labs[2])
  make_scat(tup[3], labs[3])
  plt.xlabel(x_lab)
  plt.ylabel(y_lab)
  plt.title(title)
  plt.savefig("figure6.pdf")
  plt.clf()
  plt.cla()
  plt.close()

# 1000 -> barren land -> 0
# 0100 -> trees -> 1
# 0010 -> grassland -> 2
# 0001 -> none -> 3

x = get_img_data(10000, f)
model = get_svm(x[0],x[1])

correct_class = {0: 0, 1:0, 2:0, 3:0}
wrong_class = {0: 0, 1:0, 2:0, 3:0}

# file = open("mod.dat", "wb")
# pickle.dump(model, file)
# file.close()

score = tester(model, get_test_data(10000, f))

labels = ["Barren Land", "Trees", "Grassland", "None"]

values = [float(correct_class[i]/(wrong_class[i]+correct_class[i])) for i in range(4)]

plot_it(values, "figure12.pdf", "RBF Kernel with Means of S V, Sum of Var HSV, Mean EVI")

# data = get_init_scat_data(x,1,2)
data = get_var_sum(x, 1)
tup = generate_scatter_data(data)
many_scat(tup,labels, "Sum of Var HSV", "Mean of S Value","Sum of Var HSV and Mean S Value for all classes" )