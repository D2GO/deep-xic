import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#import sys
#tt = sys.argv[1]
tt = 1

def fun(ys):
	yy = []
	for i in ys:
		yy = yy+i.tolist()
	return yy 

X_tr = np.load("train_datas/x_train_"+str((int(tt)-1))+".npy") #(n,向量长度，通道数) 
y_tr = fun(np.load("train_datas/y_train_"+str((int(tt)-1))+".npy")) #one_hot编码向量 (n,)

X_test = np.load("train_datas/X_test_"+str((int(tt)-1))+".npy")
y_test = fun(np.load("train_datas/y_test_"+str((int(tt)-1))+".npy"))

XX = X_tr

nn = XX.shape[0]
X_train = XX.reshape(nn,-1)
nn = X_test.shape[0]
X_test = X_test.reshape(nn,-1)
y_train = y_tr

#X (n,len(data)) Y(n)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

acc = accuracy_score(rfc_y_pred,y_test)
strs = "Test acc: {:.6f}".format(acc)
print('随机森林 ',strs)


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)

acc = accuracy_score(gbc_y_pred,y_test)
strs = "Test acc: {:.6f}".format(acc)
print('GradientBoosting ',strs)


from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train,y_train)
svm_y_pred = clf.predict(X_test)

acc = accuracy_score(svm_y_pred,y_test)
strs = "Test acc: {:.6f}".format(acc)
print('SVM ',strs)


from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=200, eta0=0.0001, random_state=0)
ppn.fit(X_train, y_train)
ppn_y_pred = ppn.predict(X_test)

acc = accuracy_score(ppn_y_pred,y_test)
strs = "Test acc: {:.6f}".format(acc)
print('LR ',strs)


