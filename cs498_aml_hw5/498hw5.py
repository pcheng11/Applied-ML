import pandas as pd
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier  
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split


def read_data(dir):
    label = 0
    train_data = []
    for dirs in [x[0] for x in os.walk(dir)][1:]:
        if "MODEL" not in dirs:
            label += 1
            for file in os.listdir(dirs):
                data = np.array(pd.read_csv(dirs + '/' + file, sep=" ", header=None))
                signal = list(data[:,0]) + list(data[:,1]) + list(data[:,2]) + [label]
                train_data.append(signal)
    return train_data

def cut_segments(train_data, segment_size):
    reconstruct_train_data = []
    segments = []
    for sample in train_data:
        temp = []
        for start in range(0, len(sample), segment_size):
            segment = sample[:-1][start:start+segment_size] #do not include label
            if len(segment) == segment_size:
                temp.append(segment)
                segments.append(segment)
        temp.append(sample[-1])
        reconstruct_train_data.append(temp)
    print('Total segments number:{}'.format(len(segments)))
    return segments, reconstruct_train_data

def hierarchical_kmeans(segments, fst_level_cluster_num, snd_level_cluster_num, segment_size):
    temp_dic = {}
    dic = np.zeros((fst_level_cluster_num*snd_level_cluster_num, segment_size))
    fst_kmeans = KMeans(n_clusters=fst_level_cluster_num, random_state=0).fit(segments)
    first_level = zip(fst_kmeans.labels_, segments)
    print('Building first level hierarchical kmeans...')
    for center, segment in tqdm(list(first_level)):
        if center*snd_level_cluster_num in temp_dic:
            temp_dic[center*snd_level_cluster_num].append(segment)
        else:
            temp_dic[center*snd_level_cluster_num] = [segment]

    print('Building second level hierarchical kmeans...')
    for center in tqdm(temp_dic.keys()):
        snd_kmeans = KMeans(n_clusters=snd_level_cluster_num, random_state=0).fit(temp_dic[center])
        snd_level = zip(range(snd_level_cluster_num), snd_kmeans.cluster_centers_)
        for snd_center, cluster_center in snd_level:
            dic[center + snd_center] = cluster_center
    return dic

def cluster(reconstruct_train_data, dic, dimension):
    new_train_data = []
    print("Vector Quantization...")
    for sample in tqdm(reconstruct_train_data):
        new_sample = np.zeros(dimension+1) #+1 for label
        for segment in sample[:-1]:
            center_idx = find_closest_cluster(segment, dic)
            new_sample[center_idx] += 1
        new_sample[-1] = sample[-1]
        new_train_data.append(new_sample)
    return np.array(new_train_data)

def find_closest_cluster(segment, dic):
    diff = sys.maxsize
    center_idx = 0
    for i in range(len(dic)):
        new_diff = difference(segment, dic[i])
        if new_diff < diff:
            diff = new_diff       
            center_idx = i
    return center_idx

def difference(segment, cluster_center):
    return sum((segment - cluster_center)**2)

def classify_predict(train_data):
    train_data, test_data = train_test_split(train_data, test_size=0.4)
    classifier = DecisionTreeClassifier()  
    classifier.fit(train_data[:,:-1], train_data[:,-1]) 
    y_pred = classifier.predict(test_data[:,:-1])
    print(confusion_matrix(test_data[:,-1], y_pred))  
    print(classification_report(test_data[:,-1], y_pred))  

def main():
    train_data = read_data("HMP_Dataset")
    segments, reconstruct_train_data = cut_segments(train_data, 96)
    dic = hierarchical_kmeans(segments, 40, 12, 96)
    new_train_data = cluster(reconstruct_train_data, dic, 40*12)
    classify_predict(new_train_data)



if __name__ == '__main__':
    main()