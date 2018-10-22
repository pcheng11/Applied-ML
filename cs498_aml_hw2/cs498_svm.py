import argparse
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser('SVM')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--steps', type=int)
    return parser.parse_args()

def set_logger():
    logger = logging.getLogger("SVM Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def read_data(test_file, train_file):
    test_data = pd.read_csv(test_file, header=None)
    test_set = np.array(test_data)[:,[0,2,4,10,11,12]]
    train_data = pd.read_csv(train_file, header=None)
    train_set = np.array(train_data)[:,[0,2,4,10,11,12, -1]]
    train_set[:, -1][train_set[:, -1] == ' >50K'] = 1
    train_set[:, -1][train_set[:, -1] == ' <=50K'] = -1
    return train_set, test_set

def split_train_val(train_set):
    train_set, val_set = train_test_split(train_set, test_size=0.1)
    return train_set, val_set

def initialize(mu, sigma):
    w = np.random.normal(mu, sigma, 6)
    b = np.random.normal(mu, sigma, 1)
    return w, b

def generate_mini_batch(train_set, batch_size):
    for batch_start in range(0, len(train_set), batch_size):
        yield [train_set[batch_start: batch_start+batch_size, :-1], train_set[batch_start: batch_start+batch_size,-1]]

def train_svm(epochs, steps, batch_size, train_set, lbda, logger):
    accuracy_list = []
    weight_list = []
    step_list = []
    w, b = initialize(0, 1)
    for epoch in tqdm(range(0, epochs)):
        np.random.shuffle(train_set)
        heldout_set = train_set[:50, :]
        sub_train_set = train_set[50:, :]
        learning_rate = 0.0001
        generator = generate_mini_batch(sub_train_set, batch_size)
        for step in range(1, steps+1):
            try:
                x, y = next(generator)
                w, b = update(learning_rate, w, b, lbda, x, y, batch_size)
            except StopIteration:
                continue
            if step%30 == 0:
                accuracy_list.append(validate(heldout_set, w, b))
                weight_list.append(np.linalg.norm(w))
                step_list.append(epoch*steps + step)
    logger.info('Avg accuracy for heldout set is {}'.format(np.mean(accuracy_list)))
    return w, b, accuracy_list, weight_list, step_list

def find_lbda(epochs, steps, batch_size, train_set, val_set, lbdas, logger):
    chosen_lbda = None
    cur_max_accuracy = 0
    for lbda in lbdas:
        w, b = initialize(0, 1)
        for epoch in tqdm(range(0, epochs)):
            np.random.shuffle(train_set)
            generator = generate_mini_batch(train_set, batch_size)
            learning_rate = 0.0001
            for step in range(0, steps):
                try:
                    x, y = next(generator)
                    w, b = update(learning_rate, w, b, lbda, x, y, batch_size)
                except StopIteration:
                    continue
        accuracy = validate(val_set, w, b)
        logger.info('Accuracy for lambda = {0} is {1}'.format(lbda, accuracy))
        if accuracy >= cur_max_accuracy:
            chosen_lbda = lbda
            cur_max_accuracy = accuracy
    return chosen_lbda


def normalize_split(train_set, test_set):
    for i in range(len(train_set[0])-1):
        col_mean_train = np.mean(train_set[:,i])
        col_std_train = np.std(train_set[:,i])
        train_set[:, i] = (train_set[:, i] - col_mean_train)/col_std_train
        test_set[:, i] = (test_set[:, i] - col_mean_train)/col_std_train

    train_set, val_set = split_train_val(train_set)

    return train_set, val_set, test_set


def update(learning_rate, w, b, lbda, x, y, batch_size):
    y_pred = np.dot(x, w) + b
    idx = y * y_pred
    smaller_idx = np.where(idx < 1)
    bigger_idx = np.where(idx >= 1)
    if len(smaller_idx[0]) == 0:
        w_batch = np.zeros(6)
        b_batch = 0
    else:
        w_batch = -1 * np.dot(x[smaller_idx].T, y[smaller_idx])
        b_batch = np.sum(-1 * y[smaller_idx])

    w  = w - learning_rate * (1/batch_size * w_batch + lbda*w)
    b = b - learning_rate * 1/batch_size * b_batch
    return w, b

def validate(val_set, w, b):
    num_correct = 0
    val_set_x = np.squeeze(val_set[:,:-1])
    val_set_y = val_set[:,-1]
    pred_y = np.dot(val_set_x, w)
    idx = pred_y * val_set_y
    num_correct = len(idx[idx > 0])
    return num_correct/len(val_set)

def predict(w, b, test_set, dir):
    idx = []
    preds = []
    with open(dir + 'labels_test.csv', 'w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["Example","Label"])
        for i in range(len(test_set)):
            y = np.dot(w, test_set[i,:]) + b
            if y >= 0:
                pred = '>50K'
            else:
                pred = '<=50K'
            idx = "'"+ str(i)+"'"
            writer.writerow([idx,pred])

def plot_accuracy_coefficient(lbdas, train_set, batch_size, epochs, steps, logger):
    accuracy_list = []
    weight_list = []
    for lbda in lbdas:
        _, _, sub_accuracy_list, sub_weight_list, step_list = train_svm(epochs, steps, batch_size, train_set, lbda, logger)
        accuracy_list.append(sub_accuracy_list)
        weight_list.append(sub_weight_list)
    
    #step_list = range(1, steps*epochs+1, 30)
    plt.figure(1)
    for i in range(len(accuracy_list)):
        plt.plot(step_list, accuracy_list[i], label="lambda = "+str(lbdas[i]))
    plt.xlabel('Number of steps')
    plt.ylabel('Heldout set Accuracy')
    plt.title('Accuracy Plot')
    plt.legend()
    plt.show()

    plt.figure(2)
    for i in range(len(weight_list)):
        plt.plot(step_list, weight_list[i], label="lambda = "+str(lbdas[i]))
    plt.xlabel('Number of steps')
    plt.ylabel('Magnitude of weight vector')
    plt.title('Weight Plot')
    plt.legend()
    plt.show()



def main():
    args = parse_args()
    logger = set_logger()
    data_dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw2/cs498_aml_data/'
    result_dir = '/Users/pengyucheng/Desktop/cs498_aml/cs498_aml_hw2/cs498_aml_result/'
    logger.info('Preprocessing data...')
    train_set, test_set = read_data(data_dir+'test.data', data_dir+'train.data')
    train_set, val_set, test_set = normalize_split(train_set, test_set)
    logger.info('Preprocessing done!')
    logger.info('Train set size: {0}, val set size: {1}, test set size: {2}'.format(len(train_set), len(val_set), len(test_set)))
    lbdas = [0.00025, 0.0025, 0.005, 0.0005, 0.0003] #1e-3, 1e-2, 1e-1, 1, 0.02, 0.03,
    logger.info("Epochs = {0}, Steps = {1}, Batch size = {2}".format(args.epochs, args.steps, args.batch_size))
    logger.info('Searching best regularization constant lambda...')
    lbda = find_lbda(args.epochs, args.steps, args.batch_size, train_set, val_set, lbdas, logger)
    logger.info('Searching done! lambda = ' + str(lbda))
    logger.info('Training svm using lamda = ' + str(lbda) + '...')
    w, b, _, _, _ = train_svm(args.epochs, args.steps, args.batch_size, train_set, lbda, logger)
    logger.info('SVM training done!')
    logger.info('Inferencing the model...')
    res = predict(w, b, test_set, result_dir)
    logger.info('Inferencing done!')
    logger.info('Plotting graphs')
    plot_accuracy_coefficient(lbdas, train_set, args.batch_size, args.epochs, args.steps, logger)

if __name__ == '__main__':
    main()