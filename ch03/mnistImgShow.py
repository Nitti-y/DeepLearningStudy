import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def main():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten = True, normalize = False)

    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    imgShow(img)

def imgShow(img):
    pilImg = Image.fromarray(np.uint8(img))
    pilImg.show()