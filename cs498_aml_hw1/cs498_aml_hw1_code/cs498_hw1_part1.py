import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn import svm

def read_csv(file):
    print('Reading data...')
    f = pd.read_csv(file)
    f = np.array(f)
    print('Reading data done')
    return f

def random_split(df):
    train, test = train_test_split(df, test_size=0.2)
    return train, test

def prior(df):
    return [len(df[df[:, 8]==0])/df.shape[0], len(df[df[:, 8]==1])/df.shape[0]]

def mean_variance(train_set):
    class_dic = {}
    for i in range(2): #class num
        sub_data = train_set[train_set[:, 8] == i]
        means = np.mean(sub_data[:,:-1], axis = 0)
        stds = np.std(sub_data[:,:-1], axis = 0)
        class_dic[i] = [means, stds]#holding mean and std for each feature
    return class_dic

def mean_variance_missing(train_set):
    class_dic = {}
    for i in range(2): #class num
        sub_data = train_set[train_set[:, 8] == i]
        for j in [2,3,5,7]:
            sub_data[sub_data[:, j] == 0] = np.nan
        means = np.nanmean(sub_data[:,:-1], axis = 0)
        stds = np.nanstd(sub_data[:,:-1], axis = 0)
        class_dic[i] = [means, stds]#holding mean and std for each feature
    return class_dic

def compute_prob_accuracy(test_set, class_dic, prior):
    correct_num = 0
    for record in test_set:
        arg_list = []
        for i in range(2):
            max_lh = np.sum(np.log(scipy.stats.norm(class_dic[i][0], class_dic[i][1]).pdf(record[:-1]))) + np.log(prior[i])
            arg_list.append(max_lh)
        pred = np.argmax(arg_list)
        if pred == record[8]:
            correct_num += 1
    return correct_num/test_set.shape[0]

def compute_ten_avg_accuracy(df, deal_missing=False):
    total_accuracy = 0.0
    print('Computing avg accuracy...')
    for i in range(10):
        train, test = random_split(df)
        prior_dist = prior(train)
        if deal_missing:
            cls_dic = mean_variance_missing(train)
        else:
            cls_dic = mean_variance(train)
        total_accuracy += compute_prob_accuracy(test, cls_dic, prior_dist)
    print('Computing avg accuracy done')
    return total_accuracy/10

def svm_compute_and_classify(df):
    #initiate svm
    clf = svm.SVC(kernel='linear')
    accuracy = 0.0
    print('Computing avg accuracy...')
    for i in range(10): 
        correct_num = 0
        train_set, test_set = train_test_split(df, test_size=0.2)
        train_X = train_set[:,:-1]
        train_Y = train_set[:, -1]
        test_X = test_set[:,:-1]
        test_Y = test_set[:, -1]
        clf.fit(train_X, train_Y)
        pred_label = clf.predict(test_X)
        incorrect_num = np.sum(np.abs(test_Y - pred_label))
        accuracy += (1 - incorrect_num / len(test_set))
    accuracy /= 10
    print('Computing avg accuracy done')
    return accuracy

def main():
    dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw1/cs498_aml_hw1_dataset/'
    df = read_csv(dir + "pima-indians-diabetes.csv")
    part1a_accuracy = compute_ten_avg_accuracy(df, deal_missing=False)
    print('Part 1 A accuracy:{}'.format(part1a_accuracy))
    print('==='*12)
    part1b_accuracy = compute_ten_avg_accuracy(df, deal_missing=True)
    print('Part 1 B accuracy:{}'.format(part1b_accuracy))
    print('==='*12)
    print('SVM takes a little bit time...')
    part1d_accuracy = svm_compute_and_classify(df)
    print('Part 1 D accuracy:{}'.format(part1d_accuracy))
    print('==='*12)

if __name__ == '__main__':
    main()