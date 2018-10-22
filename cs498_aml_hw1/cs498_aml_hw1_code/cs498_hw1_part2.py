import csv
import numpy as np
import pandas as pd
from scipy.misc import imsave
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier

def preprocess(train_file, val_file, test_file, stretched=False):
    """
    read in data
    """
    train_set = pd.read_csv(train_file)
    val_set = pd.read_csv(val_file)
    test_set = pd.read_csv(test_file, header=None)
    
    train_set = np.array(train_set)[:,1:] #to numpy matrix (speed up)
    val_set = np.array(val_set)
    test_set = np.array(test_set)

    return train_set, val_set, test_set

def stretch_img(dataset):
    """
    stretch img to 20x20
    """
    stretched_img = []
    imgs = dataset.reshape(-1, 28, 28)
    for img in imgs:
        left, right, top, bottom = np.min(np.where(img != 0)[1]), \
        np.max(np.where(img != 0)[1]), np.min(np.where(img != 0)[0]), \
        np.max(np.where(img != 0)[0])
        bounded_img = img[top:bottom+1, left:right+1]
        #stretch
        res_img = np.zeros((20, 20))
        new_height, new_width = bounded_img.shape
        h_ratio, w_ratio = float(new_height/20), float(new_width/20)
        for h in range(20):
            for w in range(20):
                new_h = int(h*h_ratio)
                new_w = int(w*w_ratio)
                res_img[h, w] = bounded_img[new_h, new_w]
        stretched_img.append(res_img.reshape(20*20))
    return stretched_img

def split_x_y(train_set, val_set):
    """
    split x and y
    """
    train_set_y = np.copy(train_set[:,0])
    train_set_x = np.copy(train_set[:,1:])
    val_set_y = np.copy(val_set[:,0])
    val_set_x = np.copy(val_set[:,1:])

    return train_set_x, train_set_y, val_set_x, val_set_y

def set_threshold(train_set, val_set, test_set, threshold):
    """
    set threshold for pixels
    """
    train_set_x = np.copy(train_set[:,1:])
    val_set_x = np.copy(val_set[:,1:])
    test_set = np.copy(test_set)

    train_set_x[train_set_x >= threshold] = 1
    train_set_x[train_set_x != 1] = 0
    val_set_x[val_set_x >= threshold] = 1
    val_set_x[val_set_x != 1] = 0
    test_set[test_set >= threshold] = 1
    test_set[test_set != 1] = 0
    
    return train_set_x, val_set_x, test_set
    
def naive_bayes_clf(X, Y, dist):
    """
    initialize naive bayes classifer
    """
    if dist == "Gaussian":
        clf = GaussianNB()
    elif dist == "Bernoulli":
        clf = BernoulliNB()
    else:
        print("Not implemented!")

    clf.fit(X, Y)
    return clf

def random_forest_clf(X, Y, n_tree, n_depth):
    """
    initialize random forest classifer
    
    """
    clf = RandomForestClassifier(max_depth=n_depth, n_estimators=n_tree)
    clf.fit(X, Y)
    return clf


def validate(val_set_x, labels, clf):
    """
    validate on val set
    """
    pred = clf.predict(val_set_x)
    accuracy = sum(labels == pred)/len(labels)
    return accuracy

def test(clf, test_set):
    """
    predict on the test set
    """
    pred = clf.predict(test_set)
    index = np.array(range(len(pred)))
    return np.column_stack((index,pred)), pred

def write_to_csv(res, num, dir):
    """
    write to csv file
    """
    with open(dir + 'pcheng11_'+str(num)+'.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID","Label"])
        writer.writerows(res)

def plot_img(pred, test_set, dist, dir, stretched=False):
    test_data = np.column_stack((test_set, pred))
    for i in range(10):
        if dist == "Gaussian":
            img = np.mean(test_data[test_data[:,-1] == i][:, :-1]/255, axis=0)
        else:
            img = np.mean(test_data[test_data[:,-1] == i][:, :-1], axis=0)
        if stretched:
            img = img.reshape((20, 20))
            imsave(dir+str(dist) + '_stretched_' + str(i) + '.png', img)
        else:
            img = img.reshape((28, 28))
            imsave(dir+str(dist) + '_untouched_' + str(i) + '.png', img)


def main():
    data_dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw1/cs498_aml_hw1_dataset/'
    write_dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw1/cs498_aml_hw1_report/'
    print("Preprocessing image...")
    train_set, val_set, test_set = preprocess(data_dir+'train.csv', data_dir+'val.csv', data_dir+'test.csv')
    train_set_x, train_set_y, val_set_x, val_set_y = split_x_y(train_set, val_set)
    thres_train_set_x, thres_val_set_x, thres_test_set = set_threshold(train_set, val_set, test_set, 50)
    print("Preprocessing image done!")

    print("Stretching image...")
    stre_train_set_x = stretch_img(train_set_x)
    stre_val_set_x = stretch_img(val_set_x)
    stre_test_set = stretch_img(test_set)
    print("Stretching image done")

    print("Stretching thresholded image...")
    stre_thres_train_set_x = stretch_img(thres_train_set_x)
    stre_thres_val_set_x = stretch_img(thres_val_set_x)
    stre_thres_test_set = stretch_img(thres_test_set)
    print("Stretching image done")

    print("Start validation...")
    #Gaussian + untouched used non-thresholded image
    clf = naive_bayes_clf(train_set_x, train_set_y, "Gaussian")
    accuracy_1 = validate(val_set_x, val_set_y, clf)
    res, pred = test(clf, test_set) 
    plot_img(pred, test_set, "Gaussian", write_dir)
    write_to_csv(res, 1, write_dir)
    print("1) Gaussian + untouched: " + str(accuracy_1))
    #Gaussian + stretched used non-thresholded image
    clf = naive_bayes_clf(stre_train_set_x, train_set_y, "Gaussian")
    accuracy_2 = validate(stre_val_set_x, val_set_y, clf)
    res, pred = test(clf, stre_test_set)
    print("2) Gaussian + stretched: " + str(accuracy_2))
    plot_img(pred, stre_test_set, "Gaussian", write_dir, True)
    write_to_csv(res, 2, write_dir)
    #Bernoulli + untouched used thresholded image
    clf = naive_bayes_clf(thres_train_set_x, train_set_y, "Bernoulli")
    accuracy_3 = validate(thres_val_set_x, val_set_y, clf)
    res, pred = test(clf, thres_test_set)
    print("3) Bernoulli + untouched " + str(accuracy_3))
    plot_img(pred, thres_test_set, "Bernoulli", write_dir)
    write_to_csv(res, 3, write_dir)
    #Bernoulli + stretched used thresholded image
    clf = naive_bayes_clf(stre_thres_train_set_x, train_set_y, "Bernoulli")
    accuracy_4 = validate(stre_thres_val_set_x, val_set_y, clf)
    res, pred = test(clf, stre_thres_test_set)
    print("4) Bernoulli + stretched " + str(accuracy_4))
    plot_img(pred, stre_thres_test_set, "Bernoulli", write_dir, True)
    write_to_csv(res, 4, write_dir)



    #decision_forest used un-thresholded image
    print("====="*10)
    print("Start validation...")
    index_rf = 4
    for n_tree in [10,30]:
        for n_depth in [4,16]:
            for kind in ["untouched", "stretched"]:
                index_rf += 1
                if kind == "stretched":
                    clf = random_forest_clf(stre_train_set_x, train_set_y, n_tree, n_depth)
                    accuracy_rf = validate(stre_val_set_x, val_set_y, clf)
                    res,pred = test(clf, stre_test_set)
                else:
                    clf = random_forest_clf(train_set_x, train_set_y, n_tree, n_depth)
                    accuracy_rf = validate(val_set_x, val_set_y, clf)
                    res, pred = test(clf, test_set)
                print(str(index_rf) + ": " + str(n_tree) + " trees + " + str(n_depth) + " depth + " + kind + ":" + str(accuracy_rf))
                write_to_csv(res, index_rf, write_dir)

if __name__ == '__main__':
    main()

