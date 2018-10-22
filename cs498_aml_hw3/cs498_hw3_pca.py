import csv
import pandas as pd
import numpy as np
import tqdm as tqdm
from sklearn.decomposition import PCA

def read_data(data1, data2, data3, data4, data5, non_noise_data):
	data1 = pd.read_csv(data1)
	data1 = np.array(data1)
	data2 = pd.read_csv(data2)
	data2 = np.array(data2)
	data3 = pd.read_csv(data3)
	data3 = np.array(data3)
	data4 = pd.read_csv(data4)
	data4 = np.array(data4)
	data5 = pd.read_csv(data5)
	data5 = np.array(data5)
	non_noise_data = pd.read_csv(non_noise_data)
	non_noise_data = np.array(non_noise_data)
	print(non_noise_data)
	return [data1, data2, data3, data4, data5, non_noise_data]


def reconstruct_noise(data, num_component):
	pca = PCA(n_components=num_component, svd_solver='full')
	scores = pca.fit_transform(data)
	re_data = pca.inverse_transform(scores)
	return re_data

def reconstruct_noiseless(data, non_noise_data, num_component):
	pca = PCA(n_components=num_component, svd_solver='full')
	pca.fit(non_noise_data) #using non-noise to fit
	scores = pca.transform(data)
	re_data = pca.inverse_transform(scores)
	return re_data

def write_csv(dir, res, flag):
	if flag == 2:
		with open(dir + 'pcheng11-numbers.csv', 'w+') as f:
			writer = csv.writer(f)#, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow(["0N","1N","2N","3N","4N","0c","1c","2c","3c","4c"])
			for i in res:
				writer.writerow(i)
	else:
		with open(dir + 'pcheng11-recon.csv', 'w+') as f:
			writer = csv.writer(f)#, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow(["X1","X2","X3","X4"])
			for i in res:
				writer.writerow(i)

def main():
	data_dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw3/hw3-data/'
	write_dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw3/hw3_result/'
	datas = read_data(data_dir+'dataI.csv', data_dir+'dataII.csv', data_dir+'dataIII.csv',\
						data_dir+'dataIV.csv', data_dir+'dataV.csv', data_dir+'iris.csv' )
	non_noise_data = datas[-1]
	noiseless_mean_val = np.mean(non_noise_data, axis=0)
	res = []
	for data in datas[:-1]:
		sub_res = []
		#noiseless
		for num_component in range(5): #0~4 pca
			if num_component == 0:
				re_data = np.zeros((150, 4))
				for k in range(4):
					re_data[:,k] = noiseless_mean_val[k]
			else:
				re_data = reconstruct_noiseless(data, non_noise_data, num_component)
			MSE = 1/150 * np.sum((non_noise_data - re_data)**2)
			sub_res.append(MSE)
		#noise
		mean_val = np.mean(data, axis=0)
		for num_component in range(5): #0~4 pca

			if num_component == 0:
				re_data = np.zeros((150, 4))
				for k in range(4):
					re_data[:,k] = mean_val[k]
			else:
				re_data = reconstruct_noise(data, num_component)
			print(num_component)
			print(re_data[0,:])
			MSE = 1/150 * np.sum((non_noise_data - re_data)**2)
			sub_res.append(MSE)
		res.append(sub_res)
	print(res)
	#task 1
	data2 = reconstruct_noise(datas[1], 2)
	write_csv(write_dir, data2, 1)
	#task 2
	write_csv(write_dir, res, 2)

if __name__ == '__main__':
	main()
