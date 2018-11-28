import numpy as np
from tqdm import tqdm
import struct
import matplotlib.pyplot as plt

class Boltzman_Machine:
    def __init__(self, dir):
        self.img_dir = dir
        self.noise_imgs = None
        self.imgs = None
        self.theta_hx = 0.2
        self.theta_hh = 0.2

        self.read_data()
        self.binarize()
        self.noise_flip()

    def read_data(self):
        with open(self.img_dir, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))
        # first 500 images
        self.imgs = data[:500]

    def binarize(self):
        self.imgs = self.imgs / 256
        self.imgs[np.where(self.imgs >= 0.5)] = 1
        self.imgs[np.where(self.imgs < 0.5)] = -1
        return self.imgs

    def noise_flip(self):
        self.noise_imgs = np.copy(np.reshape(self.imgs, (500, 28*28)))
        for i in range(500):
            # get non-repetitive index
            index = np.arange(28*28)
            np.random.shuffle(index)
            random_bits = index[:int(0.02*28*28)]
            self.noise_imgs[i][random_bits] *= -1
        self.noise_imgs = np.reshape(self.noise_imgs, (500, 28, 28))
        return self.noise_imgs

    def update(self):
        Qs = np.ones((500, 28, 28))/2 # Approximation distribution of P, initialize to 0.5
        for i in range(500):
            #update one imgae
            Qs[i] = self.update_img(self.noise_imgs[i], Qs[i])
            Qs[i] = np.where(Qs[i] > 0.5, 1, -1)
        return Qs

    def get_neighbors(self, row, col, length, height):
        neighbors = []
        up = [row-1, col]
        down = [row+1, col]
        left = [row, col-1]
        right = [row, col+1]
        if row == 0 and col == 0:
            neighbors.append(right)
            neighbors.append(down)
        elif row == height-1 and col == 0:
            neighbors.append(right)
            neighbors.append(up)
        elif row == 0 and col == length-1:
            neighbors.append(left)
            neighbors.append(down)
        elif row == height-1 and col == length-1:
            neighbors.append(left)
            neighbors.append(up)
        elif row == 0:
            neighbors.append(left)
            neighbors.append(down)
            neighbors.append(right)
        elif col == 0:
            neighbors.append(up)
            neighbors.append(down)
            neighbors.append(right)
        elif row == height-1:
            neighbors.append(left)
            neighbors.append(up)
            neighbors.append(right)
        elif col == length-1:
            neighbors.append(left)
            neighbors.append(up)
            neighbors.append(down)
        else:
            neighbors.append(left)
            neighbors.append(up)
            neighbors.append(down)
            neighbors.append(right)
        return np.array(neighbors)

    def update_img(self, noise_img, Q):
        for i in range(10):
            for row in range(28):
                for col in range(28):
                    neighbors = self.get_neighbors(row, col, 28, 28)
                    rows = neighbors[:,0]
                    cols = neighbors[:,1]
                    log_pos_q = self.theta_hh * \
                        np.sum(2*Q[rows, cols]-1) + self.theta_hx * noise_img[row, col]
                    log_neg_q = -1 * self.theta_hh * \
                        np.sum(2*Q[rows, cols]-1) - self.theta_hx * noise_img[row, col]
                    Q[row, col] = np.exp(log_pos_q) / (np.exp(log_pos_q) + np.exp(log_neg_q))
        return Q


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def main():
    BM = Boltzman_Machine(
        '/Users/pengyucheng/Desktop/Applied-ML/data/cs498_aml_hw9_dataset/train-images-idx3-ubyte')

    denoise_imgs = BM.update()
    show_img(BM.imgs[1])
    show_img(BM.noise_imgs[1])
    show_img(denoise_imgs[1])
if __name__ == '__main__':
    main()


    


