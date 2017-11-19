import numpy as np

def get_blur_matrix(blur_rate, edge):
    matrix = [[0 for i in range(edge)] for j in range(edge)]
    matrix[0][0] = 1 - 2 * blur_rate
    matrix[0][1] = 2 * blur_rate
    for i in range(1, edge - 1):
        matrix[i][i - 1] = matrix[i][i + 1] = blur_rate
        matrix[i][i] = 1 - 2 * blur_rate
    matrix[edge - 1][edge - 2] = 2 * blur_rate
    matrix[edge - 1][edge - 1] = 1 - 2 * blur_rate
    return np.array(matrix)
