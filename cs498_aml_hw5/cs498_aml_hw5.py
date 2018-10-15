import numpy as np
import os
import pandas as pd
import sys
import itertools
from matplotlib import pyplot as plt
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def read_data(dir):
    label = 0
    train_data = []
    class_names = []
    for dirs in [x[0] for x in os.walk(dir)][1:]:
        if "MODEL" not in dirs:
            label += 1
            class_names.append(dirs)
            for file in os.listdir(dirs):
                data = np.array(pd.read_csv(dirs + '/' + file, sep=" ", header=None))
                signal = list(np.reshape(data, (1,-1))[0]) + [label]
                train_data.append(signal)
    return train_data, class_names


def cut_segments(data, segment_size, overlap_percent, overlap=False):
    reconstruct_data = []
    segments = []
    if overlap:
        for signal in data:
            temp = []
            for start in range(0, len(signal), segment_size-int(overlap_percent*segment_size)):
                segment = signal[:-1][start:start+segment_size] #do not include label
                if len(segment) == segment_size:
                    temp.append(segment)
                    segments.append(segment)
            temp.append(signal[-1])
            reconstruct_data.append(temp)
    else:
        for signal in data:
            temp = []
            for start in range(0, len(signal), segment_size):
                segment = signal[:-1][start:start+segment_size] #do not include label
                if len(segment) == segment_size:
                    temp.append(segment)
                    segments.append(segment)
            temp.append(signal[-1])
            reconstruct_data.append(temp)

    print('Total segments number:{}'.format(len(segments)))
    return segments, reconstruct_data

def hierarchical_kmeans(segments, fst_level_cluster_num, snd_level_cluster_num, segment_size):
    temp_dic = {}
    dic = {}
    fst_kmeans = KMeans(n_clusters=fst_level_cluster_num, random_state=0).fit(segments)
    first_level = zip(fst_kmeans.labels_, segments)
    print('Building first level hierarchical kmeans...')
    for center, segment in tqdm(list(first_level)):
        if center in temp_dic:
            temp_dic[center].append(segment)
        else:
            temp_dic[center] = [segment]
    print('Building second level hierarchical kmeans...')
    for center in tqdm(range(fst_level_cluster_num)):
        snd_kmeans = KMeans(n_clusters=snd_level_cluster_num, random_state=0).fit(temp_dic[center])
        dic[center] = snd_kmeans
    return fst_kmeans, dic

def quantization(reconstruct_data, fst_kmeans, dic, fst_level_cluster_num, snd_level_cluster_num):
    new_data = []
    dimension = fst_level_cluster_num * snd_level_cluster_num
    print("Vector Quantization...")
    for sample in tqdm(reconstruct_data):
        new_sample = np.zeros(dimension+1) #+1 for label
        for segment in sample[:-1]:
            fst_center = fst_kmeans.predict([segment])[0]
            snd_center = dic[fst_center].predict([segment])[0]
            new_sample[fst_center*snd_level_cluster_num+snd_center] += 1
            new_sample[fst_center] += 1
        new_sample[-1] = sample[-1]
        new_data.append(new_sample)
    return np.array(new_data)

def plot_hist(data, name, dimension):
    plt.figure()
    plt.bar(range(dimension), data/sum(data))
    plt.ylim(top=0.2)
    plt.ylabel('Frequency')
    plt.xlabel('cluster index')
    plt.title(name[12:])
    plt.savefig('/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw5/report/' + name[12:] + '.png')


def classify_predict(train_data, test_data, class_names, classifier):
    # print(train_data[0,:])
    if classifier == 'svm':
        clf = svm.LinearSVC() 
    if classifier == 'rf':
        clf = RandomForestClassifier()  
    clf.fit(train_data[:,:-1], train_data[:,-1]) 
    y_pred = clf.predict(test_data[:,:-1])
    cnf_matrix = confusion_matrix(test_data[:,-1], y_pred)
    plt.figure()
    accuracy = accuracy_score(test_data[:,-1], y_pred)
    # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion Matrix')
    print('Accuracy: ', accuracy)
    print('=========='*10)
    return accuracy


'''
The code below is from Scikit-learn documentation
http://scikit-learn.org/stable/auto_examples/model_selection/\
plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def experiement(fst_level_cluster_num, snd_level_cluster_num, \
                segment_size, overlap, overlap_size, train_data, \
                test_data, clf, class_names):
    train_segments, reconstruct_train_data = cut_segments(train_data, segment_size, overlap_size, overlap=overlap)
    fst_kmeans, dic = hierarchical_kmeans(train_segments, fst_level_cluster_num, snd_level_cluster_num, 96)
    new_train_data = quantization(reconstruct_train_data, fst_kmeans, dic, fst_level_cluster_num, snd_level_cluster_num)
    test_segments, reconstruct_test_data = cut_segments(test_data, segment_size, overlap_size, overlap=overlap)
    new_test_data = quantization(reconstruct_test_data, fst_kmeans, dic, fst_level_cluster_num, snd_level_cluster_num)

    return classify_predict(new_train_data, new_test_data, class_names, clf)

def plot(classes, class_names, new_train_data, fst_level_cluster_num, snd_level_cluster_num):
    print('Saving figures...')
    for num in range(classes):
        sub_data = new_train_data[np.where(new_train_data[:,-1] == num+1)]
        plot_hist(np.mean(sub_data, axis=0)[:-1], class_names[num], fst_level_cluster_num*snd_level_cluster_num)

def main():
    classes = 14

    train_data, class_names = read_data("HMP_Dataset")
    print("Split dataset before quantization...")
    train_data, test_data = train_test_split(train_data, test_size=0.2)

    segment_sizes = [18, 48, 96, 192]
    overlap_percents = [0.1, 0.2, 0.5, 0.8]
    cluster_centers = [(5,2), (10,4), (20,6), (30, 8), (40, 12), (50, 20)]

    for overlap in [False, True]:
        for clf in ["rf", "svm"]:
            for k in cluster_centers:
                for segment_size in segment_sizes:
                    if overlap:
                        print('Overlap? :{0}'.format(overlap))
                        print('cluster number info: {0}'.format(k))
                        print('segment size info: {0}'.format(segment_size))
                        for overlap_percent in overlap_percents:
                            print('overlap percent info: {0}'.format(overlap_percent))
                            try:
                                accuracy = experiement(k[0], k[1], segment_size, overlap, overlap_percent, train_data, test_data, clf, class_names)
                                storing_info = np.array([overlap, clf, k, segment_size, overlap_percent, accuracy])
                            except:
                                continue
                    else:
                        print('Overlap? :{0}'.format(overlap))
                        print('cluster number info: {0}'.format(k))
                        print('segment size info: {0}'.format(segment_size))
                        try:
                            accuracy = experiement(k[0], k[1], segment_size, overlap, 0.1, train_data, test_data, clf, class_names)
                            storing_info = np.array([overlap, clf, k, segment_size, 'N/A', accuracy])
                        except:
                            continue


if __name__ == '__main__':
    main()