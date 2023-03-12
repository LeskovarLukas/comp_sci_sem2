from typing import List

import numpy as np
import matplotlib.pyplot as plt

def define_transformations() -> List[np.ndarray]:
    """
        Returns the four transformations t_1, .., t_4 to transform the quadrat. 
        The transformations are determined by using mscale, mrotate and mtranslate.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    t1, t2, t3, t4 = np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))

    t1 = mtranslate(-3, 0) @ mrotate(55)
    t2 = mrotate(55) @ mtranslate(-3, 0)
    t3 = mtranslate(3, 1) @ mrotate(70) @ mscale(3, 2)
    
    # t4 is a shear by theta (roughly 53.13 degrees) that is rotated 
    #define known properties
    diag_1 = 3 * np.sqrt(2)
    diag_2 = np.sqrt(2)

    #calculate side length
    side = np.sqrt((diag_1 / 2)**2 + (diag_2 / 2)**2)

    #calculate inner angle
    alpha = 2 * np.arctan((diag_2 / 2) / (diag_1 / 2))

    #calculate shear angle
    theta = (np.pi / 2 - alpha)

    #calculate shear matrix
    xShear = np.array([[1, np.tan(theta), 0], [0, 1, 0], [0, 0, 1]])
    
    # calculate long side after shear
    error = 1 / np.sin(alpha)

    #calculate translation matrix   
    t4 = mrotate(np.rad2deg(theta + alpha / 2))
    t4 = t4 @ xShear
    t4 = t4 @ mscale(1, 1 / error)
    t4 = t4 @ mscale(side, side)

    ### END STUDENT CODE

    return [t1, t2, t3, t4]

def mscale(sx : float, sy : float) -> np.ndarray:
    """
        Defines a scale matrix. The scales are determined by s_x in x and s_y in y dimension.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])                               

    ### END STUDENT CODE

    return m

def mrotate(angle : float) -> np.ndarray:
    """
        Defines a rotation matrix (z-axis) determined by the angle in degree (!).
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    angle = np.deg2rad(angle)
    m = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    ### END STUDENT CODE

    return m
    
def mtranslate(tx : float, ty : float) -> np.ndarray:
    """
        Defines a translation matrix. t_x in x, t_y in y direction.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    m = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    ### END STUDENT CODE

    return m

def transform_vertices(v : np.ndarray, m : np.ndarray) -> np.ndarray:
    """
        transform the (3xN) vertices given by v with the (3x3) transformation matrix determined by m.
    """
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    out = np.zeros(v.shape)

    out = m @ v

    print(out)

    ### END STUDENT CODE

    return out

def display_vertices(v : np.ndarray, title : str) -> None:
    """
        Plot the vertices in a matplotlib figure.
    """
    # create the figure and set the title
    plt.figure()
    plt.axis('square')

    plt.title(title)

    # x and y limits
    plt.xlim((-6,6))
    plt.ylim((-6,6))
    plt.xticks(range(-6,6))
    plt.yticks(range(-6,6))

    # plot coordinate axis
    plt.axvline(color='black')
    plt.axhline(color='black')
    plt.grid()
    
    # we just add the last element, so plot can do our job :)
    v_ = np.concatenate((v, v[:, 0].reshape(3,-1)), axis=1)

    plt.plot(v_[0, :], v_[1, :], linewidth=3)
    plt.show()
