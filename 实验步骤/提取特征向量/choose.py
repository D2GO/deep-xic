import numpy as np
import os

paths = ['datas/AT','datas/PB']

def get_rand_vecs():
	lis = np.load("vecs.npy")
	n = 4096
	indexs = np.random.choice(lis.shape[0],n,replace=False)
	lists = lis[indexs]
	np.save("vecs_key.npy",lists)
	return lists

def get_vecs(path,lists,low,high,ks):
	files = os.listdir(path)
	for l in files:
		res = []
		arr = []
		f = np.load(path+'/'+l)
		x1 = f[:,0]
		x2 = f[:,1]
		me1,me2 = np.percentile(x1, [low, high])
		for i in range(len(x1)):
			if x1[i] > me1 and x1[i] < me2:
				arr.append(i)
		ff = f[arr]
		for j in lists:
			ss = ff[np.argwhere(ff[:,1]==j)][:,0]
			if ss.tolist() == []:
				ss = [0]
			res.append(np.mean(ss))
			res.append(np.std(ss))
		np.save("datas_mz/"+path[-2:]+"/"+str(ks)+l,res)
		print("datas_mz/"+path[-2:]+"/"+str(ks)+l)

def get_vec_mz(ks,key,l,h):
	for path in paths:
		get_vecs(path,key,l,h,ks)
	print("特征向量提取完成")

l = 5
h = 95

# 先随机选择4096个质合比作为key，然后根据key从上一步得到的数据中选择出这4096个质合比
key = get_rand_vecs()
get_vec_mz(0,key,l,h)
