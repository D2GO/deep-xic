import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os

#labels -> one_hot
def one_hot(y):
	lb = LabelBinarizer()
	lb.fit(y)
	yy = lb.transform(y)
	return yy

#Z-score标准化
def z_score(x):
	#x = np.log(x)
	x = (x - np.average(x))/np.std(x)
	return x

def adds(path,num):
	files = os.listdir(path)
	nn = len(files)
	Xt = np.zeros((nn, 256, 32))
	for m,n in enumerate(files):
		ms = np.load(path+"/"+n)
		xs = []
		ss = []
		for j in range(32):
			ss = ms[j*256:(j+1)*256]
			xs.append(ss)
		xs = z_score(xs)
		m_s = np.array(xs).transpose(1,0)
		Xt[m,:,:] = m_s

	labels = [num]*nn
	X = Xt.tolist()
	return X,labels

#################################

# 训练集合并
def get_train_datas(ks):
	X2,labels_t2 = adds('datas_mz/AT',0)
	X3,labels_t3 = adds('datas_mz/PB',1)

	Xtt = [j for j in X2]+[k for k in X3]
	X_train = np.array(Xtt)

	labels_train = labels_t2 + labels_t3

	# 划分训练集和测试集，由于此处为样例程序，所以将验证集作为测试集
	X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, test_size=0.2,random_state = 123)

	y_tr = one_hot(lab_tr)
	y_vld = one_hot(lab_vld)

	np.save("train_datas/x_train_"+str(ks)+".npy",X_tr)
	np.save("train_datas/x_ver_"+str(ks)+".npy",X_vld)
	np.save("train_datas/X_test_"+str(ks)+".npy",X_vld)
	np.save("train_datas/y_train_"+str(ks)+".npy",y_tr)
	np.save("train_datas/y_ver_"+str(ks)+".npy",y_vld)
	np.save("train_datas/y_test_"+str(ks)+".npy",y_vld)
	print("训练数据处理完成")

get_train_datas(0)
