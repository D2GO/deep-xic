import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


n = len(np.load("x_train.npy"))
X_train = np.load("x_train.npy").reshape(n,-1)
y_train = np.load("y_train.npy").reshape(n)
n = len(np.load("d_x_test.npy"))
X_test = np.load("d_x_test.npy").reshape(n,-1)
y_test = np.load("d_y_test.npy").reshape(n)


def svm_classify(X_train,y_train,X_test,y_test):
	clf = SVC()
	clf.fit(X_train,y_train)
	svm_y_pred = clf.predict(X_test)
	acc = accuracy_score(svm_y_pred,y_test)
	f1 = f1_score(y_test, svm_y_pred)
	y_scores = clf.decision_function(X_test)
	auc = roc_auc_score(y_test,y_scores)
	strs = "SVM Test_acc: {:.6f}".format(acc)
	strs = strs + "F1-score: {:.6f}".format(f1)
	strs = strs + "AUC: {:.6f}".format(auc)
	return strs


def lr_classify(X_train,y_train,X_test,y_test):
	ppn = Perceptron(n_iter=1, eta0=0.0001, random_state=0)
	ppn.fit(X_train,y_train)
	ppn_y_pred = ppn.predict(X_test)
	acc = accuracy_score(ppn_y_pred,y_test)
	f1 = f1_score(y_test, ppn_y_pred)
	y_scores = ppn.decision_function(X_test)
	auc = roc_auc_score(y_test,y_scores)
	strs = "LR Test_acc: {:.6f}".format(acc)
	strs = strs + " F1-score: {:.6f}".format(f1)
	strs = strs + " AUC: {:.6f}".format(auc)
	return strs


def rf_classify(X_train,y_train,X_test,y_test):
	rfc = RandomForestClassifier(n_estimators = 100)
	rfc.fit(X_train,y_train)
	rf_y_pred = rfc.predict(X_test)
	acc = accuracy_score(rf_y_pred,y_test)
	f1 = f1_score(y_test, rf_y_pred)
	y_scores = rfc.predict_proba(X_test)[:,1]
	auc = roc_auc_score(y_test,y_scores)
	strs = "RF Test_acc: {:.6f}".format(acc)
	strs = strs + "F1-score: {:.6f}".format(f1)
	strs = strs + "AUC: {:.6f}".format(auc)
	return strs


def gdbt_classify(X_train,y_train,X_test,y_test):
	gbc = GradientBoostingClassifier(n_estimators = 100)
	gbc.fit(X_train,y_train)
	gb_y_pred = gbc.predict(X_test)
	acc = accuracy_score(gb_y_pred,y_test)
	f1 = f1_score(y_test, gb_y_pred)
	y_scores = gbc.predict_proba(X_test)[:,1]
	auc = roc_auc_score(y_test,y_scores)
	strs = "GDBT Test_acc: {:.6f}".format(acc)
	strs = strs + "F1-score: {:.6f}".format(f1)
	strs = strs + "AUC: {:.6f}".format(auc)
	return strs


res = svm_classify(X_train,y_train,X_test,y_test)
print(res)
res = lr_classify(X_train,y_train,X_test,y_test)
print(res)
res = rf_classify(X_train,y_train,X_test,y_test)
print(res)
res = gdbt_classify(X_train,y_train,X_test,y_test)
print(res)



