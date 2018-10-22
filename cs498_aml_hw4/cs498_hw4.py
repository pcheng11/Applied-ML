"""
HW4 for UIUC CS 498 AML FA18
Author: Pengyu Cheng
"""


import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def read_data(directory):
	dicts = []
	for filename in os.listdir(directory):
		if filename[:10] == 'data_batch': 
			file = directory+filename
			dicts.append(unpickle(file))
		elif filename == 'test_batch':
			file = directory+filename
			test_data = unpickle(file)
		else:
			continue
	return dicts, test_data


def load_names(directory):
	names = unpickle(directory+"batches.meta")[b'label_names']
	label_names = [x.decode('utf-8') for x in names]	
	return label_names


def process_data(dicts):
	flag = 0
	for i in range(len(dicts)):
		batch_data = np.column_stack((dicts[i][b'data'], dicts[i][b'labels']))
		if flag == 1:
			data = np.row_stack((data, batch_data))
		else:
			data = batch_data
			flag = 1

	return data

def split_category(data):
	categories = []
	for i in range(10):
		subdata = np.squeeze(data[np.where(data[:,-1] == i), :-1])
		categories.append(subdata)
	return categories

def compute_mean(categories):
	mean_categories = []
	for category in categories:
		mean_categories.append(np.mean(category, axis=0))
	return mean_categories

def calculate_error(category, re_category):
	error = 1/5000 * np.sum((category - re_category)**2)
	return error

def compute_dis_mat(mean_categories):
	dis_mat = cdist(mean_categories, mean_categories, "euclidean")**2
	return dis_mat

def plot_error(errors, label_names):
	plt.bar(label_names, errors, align='center', alpha=1, color = 'red')
	plt.xlabel('categories')
	plt.title('plot of error vs category')
	plt.show()

def plot_2d(points, label_names):
	plt.scatter(points[:, 0], points[:, 1], marker='o', color ='red')
	for label, x, y in zip(label_names, points[:, 0], points[:, 1]):
		plt.annotate(label, xy=(x, y), xytext=(-10, 10), textcoords='offset points')
	plt.title("2D plot with the results of principal coordinate analysis")
	plt.show()

def perform_pca(category):
	pca = PCA(n_components=20)
	scores = pca.fit_transform(category)
	re_category = pca.inverse_transform(scores)
	return re_category

def perform_mds(D):
	A = np.identity(10) - 1/10 * np.ones((10, 10))
	W = -1/2 * np.dot(np.dot(A, D), A.T)
	eigen_val, eigen_vec = np.linalg.eig(W)
	idx = eigen_val.argsort()[::-1]
	eigen_val = eigen_val[idx]
	eigen_vec = eigen_vec[:, idx]
	eigen_val = np.diag(eigen_val)

	sigma = np.sqrt(eigen_val[0:2, 0:2]) 
	y = np.dot(eigen_vec[:, 0:2], sigma)
	print(y)
	return y


def main():
	directory = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw4/cifar-10-batches-py/'
	dicts, test_data = read_data(directory)
	label_names = load_names(directory)
	data = process_data(dicts)
	categories = split_category(data)
	mean_categories = compute_mean(categories)
	dis_mat = compute_dis_mat(mean_categories)
	errors = []
	for category in tqdm(categories):
		re_category = perform_pca(category)
		errors.append(calculate_error(category, re_category))
	plot_error(errors, label_names)
	y = perform_mds(dis_mat)
	plot_2d(y, label_names)


if __name__ == '__main__':
	main()
