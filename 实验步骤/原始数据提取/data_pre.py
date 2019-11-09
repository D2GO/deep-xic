import os
import re
import numpy as np

# 根据每批数据的特点修改对应的原始数据提取脚本
# 这里以该批处理后的癌症样本为例

all = os.listdir('IPX0000937001XIC1_3_10')

def split_(ii):
	for j in ii[::-1]:
		if j not in ['A','T','P','B']:
			continue
		else:
			if j in ['A','T']:
				return 1
			else:
				return 0

pb_all = []
at_all = []
for i in all:
	if split_(i) == 1:
		at_all.append(i)
	if split_(i) == 0:
		pb_all.append(i)	


def get_re(ff):
	for i,j in enumerate(ff[::-1]):
		if j not in ['A','T','P','B']:
			continue
		else:
			return ff[-i-8:-i]	

# 得到病例-数据文件的字典
	 
dic = {}
for i in pb_all:
	ind = get_re(i)
	if ind not in dic.keys():
		dic[ind] = [i]
	else:
		dic[ind].append(i)

redic = {}	
for i in at_all:
	ind = get_re(i)
	if ind not in redic.keys():
		redic[ind] = [i]
	else:
		redic[ind].append(i)

def load_data(file_path):
	ll = np.array([9,12])
	bars = []
	f = open(file_path,'r')
	data = f.readlines()
	for ii in data[1:-1]:
		cc = ii.split('\n')[0]
		cc = cc.split('\t')
		cc = (np.array(list(map(float,cc))))[ll]
		#print(cc)
		cc[1] = cc[1]+0.0001
		cc[1] = float(re.findall(r"\d{1,}?\.\d{2}", str(cc[1]))[0])
		bars.append(cc)
	bars = sorted(bars, key=lambda arr: arr[1])
	return bars

# 根据上一步得到的关系字典将每一个病例的数据文件提取后合并成一个文件保存至本地

for i in dic.items():
	n = 0
	for j in i[1]:
		print(j)
		gg = load_data('IPX0000937001XIC1_3_10/'+j)
		if n == 0:
			rr = gg
			n = 1
		else:			
			rr = np.concatenate((rr,gg),axis = 0)
	np.save('AT/'+str(i[0])+'.npy',rr)

for i in redic.items():
	n = 0
	for j in i[1]:
		gg = load_data('IPX0000937001XIC1_3_10/'+j)
		if n == 0:
			rr = gg
			n = 1
		else:
			rr = np.concatenate((rr,gg),axis = 0)
	np.save('PB/'+str(i[0])+'.npy',rr)







