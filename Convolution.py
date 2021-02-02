import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

class CNN:
    def __init__(self):
        self.self = self

    def GaussianBlur_1(self):
        a = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]])
        k = a / np.sum(a)
        return k

    def GaussianBlur_2(self):
        a = (1 / 256) * np.array([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, 36, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]])
        return a

    def edge_detection(self):
        a = np.array([[1, 0, -1],
                      [0, 0, 0],
                      [-1, 0, 1]])
        return a

    def sharpen(self):
        a = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
        return a

    def sobel_edge_1(self):
        a = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
        return a

    def sobel_edge_2(self):
        a = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]).T
        return a

    def laplacian(self):
        a = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])
        return a

    def Laplacian_of_Gaussian(self):
        a = np.array([[0, 0, -1, 0, 0],
                      [0, -1, -2, -1, 0],
                      [-1, -2, 16, -2, -1],
                      [0, -1, -2, -1, 0],
                      [0, 0, -1, 0, 0]])
        return a

    def convolution(self, img, kernel):

        filter = np.flipud(np.fliplr(kernel))

        k_l = filter.shape[0]
        k_h = filter.shape[1]

        # Zero Padding
        pad = (k_l - 1) // 2
        total_pad = 2 * pad

        padded_image = np.zeros((img.shape[0] + total_pad, img.shape[1] + total_pad))
        padded_image[pad:-pad, pad:-pad] = img

        output_image = np.zeros(img.shape)

        for i in range(padded_image.shape[0] - total_pad):
            for j in range(padded_image.shape[1] - total_pad):
                output_image[i, j] = np.multiply(filter, padded_image[i: i + k_l, j: j + k_h]).sum()

        return output_image

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, classes = load_dataset()

    image = X_train[2]
    image = image / 255

    plt.imshow(image)
    plt.show()

    model = CNN()

    k = model.GaussianBlur_2()
    img = image[:, :, 0]

    filterned_Image = model.convolution(img, k)
    plt.imshow(filterned_Image)
    plt.show()


