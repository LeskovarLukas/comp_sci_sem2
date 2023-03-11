import numpy as np
import scipy.ndimage
from PIL import Image

import utils


def read_img(inp:str) -> Image.Image:
    """
        Returns a PIL Image given by its input path.
    """
    img =  Image.open(inp)
    return img

def convert(img:Image.Image) -> np.ndarray:
    """
        Converts a PIL image [0,255] to a numpy array [0,1].
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    out = np.array(img) / 255

    ### END STUDENT CODE
    return out

def switch_channels(img:np.ndarray) -> np.ndarray:
    """
        Swaps the red and green channel of a RGB iamge given by a numpy array.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    out = np.zeros(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j,0] = img[i,j,1]
            out[i,j,1] = img[i,j,0]
            out[i,j,2] = img[i,j,2]

    ### END STUDENT CODE

    return out

def image_mark_green(img:np.ndarray) -> np.ndarray:
    """
        returns a numpy-array (HxW) with 1 where the green channel of the input image is greater or equal than 0.7, otherwise zero.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    mask = np.where(img[:, :, 1] >= 0.7, 1, 0)


    ### END STUDENT CODE

    return mask


def image_masked(img:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """
        sets the pixels of the input image to zero where the mask is 1.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    mask = np.atleast_3d(1 - mask)
    print(mask.shape)

    out = np.multiply(img, mask)

    ### END STUDENT CODE

    return out

def grayscale(img:np.ndarray) -> np.ndarray:
    """
        Returns a grayscale image of the input. Use utils.rgb2gray().
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    out = utils.rgb2gray(img)

    ### END STUDENT CODE

    return out

def cut_and_reshape(img_gray:np.ndarray) -> np.ndarray:
    """
        Cuts the image in half (x-dim) and stacks it together in y-dim.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    left = img_gray[:, :img_gray.shape[1]//2]
    right = img_gray[:, img_gray.shape[1]//2:]
    out = np.concatenate((right, left), axis=0)

    ### END STUDENT CODE

    return out

def filter_image(img:np.ndarray) -> np.ndarray:
    """
        filters the image with the gaussian kernel given below. 
    """
    gaussian = utils.gauss_filter(5, 2)

    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    filter_sum = np.sum(gaussian)

    out = np.zeros(img.shape)
    img = np.pad(img, ((2, 2), (2, 2), (0, 0)), 'constant')

    for x in range(0, out.shape[0]):
        for y in range(0, out.shape[1]):
            sub_img = np.multiply(img[x:x+5, y:y+5], gaussian[:, :, np.newaxis])
            out[x, y] = np.sum(sub_img, axis=(0, 1)) / filter_sum
    
    ### END STUDENT CODE

    return out

def horizontal_edges(img:np.ndarray) -> np.ndarray:
    """
        Defines a sobel kernel to extract horizontal edges and convolves the image with it.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    out = scipy.ndimage.correlate(img, kernel, mode='constant')

    #out = np.zeros(img.shape)

    ### END STUDENT CODE

    return out
