def read_data(test_file, train_file):
	with open(test_file, 'r') as f:
		test_data = f.readlines()
		print(test_data)




def main():
	data_dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw2/cs498_aml_data/'
	read_data(data_dir+'test.data', data_dir+'train.data')

if __name__ == '__main__':
	main()