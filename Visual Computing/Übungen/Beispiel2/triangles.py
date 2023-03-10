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

    vectors = np.array([P1P2, P2P3, P3P1])
    norms = [0., 0., 0.]

    for i in range(0, vectors.shape[0]):
        norm = 0
        for j in range(0, vectors[i].shape[0]):
            norm += vectors[i][j] ** 2

        norms[i] = np.sqrt(norm)

    ### END STUDENT CODE

    return norms

def compute_normal_vector(P1P2:np.ndarray, P2P3:np.ndarray, P3P1:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    A = P1P2
    B = -P3P1

    n = np.cross(A, B)
    n_normalized = n / np.linalg.norm(n)
    ### END STUDENT CODE

    return n, n_normalized

def compute_triangle_area(n:np.ndarray) -> float:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    area = np.linalg.norm(n) / 2
    ### END STUDENT CODE

    return area

def compute_angles(P1P2:np.ndarray,P2P3:np.ndarray,P3P1:np.ndarray) -> Tuple[float, float, float]:
    ### STUDENT CODE
    # TODO: Implement this function.

	# NOTE: The following lines can be removed. They prevent the framework
    #       from crashing.

    alpha = np.arccos(np.dot(P1P2, -P3P1) / (np.linalg.norm(P1P2) * np.linalg.norm(-P3P1)))
    beta = np.arccos(np.dot(P2P3, -P1P2) / (np.linalg.norm(P2P3) * np.linalg.norm(-P1P2)))
    gamma = np.arccos(np.dot(P3P1, -P2P3) / (np.linalg.norm(P3P1) * np.linalg.norm(-P2P3)))

    alpha = np.degrees(alpha)
    beta = np.degrees(beta)
    gamma = np.degrees(gamma)

    ### END STUDENT CODE

    return alpha, beta, gamma

