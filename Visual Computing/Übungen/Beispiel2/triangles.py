from typing import List, Tuple

import numpy as np

def define_triangle() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    P1 = np.array([3, -2, -1])
    P2 = np.array([-5, -3, 4])
    P3 = np.array([-3, 9, -3])
    ### END STUDENT CODE

    return P1, P2, P3

def define_triangle_vertices(P1:np.ndarray, P2:np.ndarray, P3:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    P1P2 = P2 - P1
    P2P3 = P3 - P2
    P3P1 = P1 - P3
    ### END STUDENT CODE

    return P1P2, P2P3, P3P1

def compute_lengths(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> List[float]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    vectors = np.ndarray([P1P2, P2P3, P3P1])
    norms = [0., 0., 0.]

    for i in range(0, vectors.shape[0]):
        norms[i] = np.square(vectors[i])

    ### END STUDENT CODE

    return norms

def compute_normal_vector(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    n = np.zeros(3)
    n_normalized = np.zeros(3)
    ### END STUDENT CODE

    return n, n_normalized

def compute_triangle_area(n:np.ndarray) -> float:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    area = 0
    ### END STUDENT CODE

    return area

def compute_angles(P1P2:np.ndarray,P2P3:np.ndarray,P3P1:np.ndarray) -> Tuple[float, float, float]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    alpha, beta, gamma = 0., 0., 0.
    ### END STUDENT CODE

    return alpha, beta, gamma

