import pickle
import numpy as np
import pandas as pd

# load


def initialize(M1, N1, N2, K):
    X = np.random.rand(M1, K)
    Y = np.random.rand(N1, K)
    Z = np.random.rand(N2, K)
    return X, Y, Z


def main_func(matrix1,matrix2,matrix_B):
    # matrix1
    # matrix2
    # matrixB
    M1, N1 = matrix1.shape
    M2, N2 = matrix2.shape
    K = 100
    alpha = 0.5
    beta = 1
    X, Y, Z = initialize(M1, N1, N2, K)
    max_iteration = 100
    iteration = 0
    ERt = ER(X, Y, Z, matrix1, matrix2, matrix_B, alpha, beta)
    ER_history=0
    while (abs(ER_history-ERt)>100 and iteration < max_iteration):
        ER_history=ERt

        print ERt * 1.0 / np.sum(matrix_B)
        iteration += 1
        grad_X, grad_Y, grad_Z = gradient_XYZ(X, Y, Z, matrix1, matrix2, matrix_B, alpha, beta)
        gamma = 1.0
        while (ER((X - gamma * grad_X), (Y - gamma * grad_Y), (Z - gamma * grad_Z), matrix1, matrix2, matrix_B, alpha,
                  beta) > ERt):
            gamma /= 2.0

        X = X - gamma * grad_X
        Y = Y - gamma * grad_Y
        Z = Z - gamma * grad_Z
        ERt = ER(X, Y, Z, matrix1, matrix2, matrix_B, alpha, beta)
        np.savez(open("data/3_SVD_XYZ.npz", "w"), X=X, Y=Y, Z=Z)
        print "save ok"
    print "END"
    return X, Y, Z


def ER(X, Y, Z, matrix1, matrix2, matrix_B, alpha, beta):
    result = 0
    result += 0.5 * np.sum((matrix_B * (matrix1 - np.dot(X, Y.T))) ** 2)
    result += alpha * 0.5 * np.sum((matrix2 - np.dot(X, Z.T)) ** 2)
    result += beta * 0.5 * (np.sum(X ** 2) + np.sum(Y ** 2) + np.sum(Z ** 2))
    return result


def gradient_XYZ(X, Y, Z, matrix1, matrix2, matrix_B, alpha, beta):
    grad_X = np.dot(matrix_B * (np.dot(X, Y.T) - matrix1), Y)
    grad_X += alpha * (np.dot((np.dot(X, Z.T) - matrix2), Z)) + beta * X

    grad_Y = np.dot((matrix_B * (np.dot(X, Y.T) - matrix1)).T, X) + beta * Y
    grad_Z = alpha * np.dot((np.dot(X, Z.T) - matrix2).T, X) + beta * Z
    return grad_X, grad_Y, grad_Z

if __name__ == "__main__":
    npzfile=np.load("data/3_rating_matrix.npz")

    print "load finishing"

    matrix1=npzfile['arr_0']
    matrix_B=npzfile['arr_1']
    matrix2=npzfile['arr_2']

    M1, N1 = matrix1.shape
    M2, N2 = matrix2.shape
    K = 100
    print N1,N2,M1
    X, Y, Z = main_func(matrix1,matrix2,matrix_B)
    np.savez(open("data/3_SVD_XYZ.npz", "w"),X=X,Y=Y,Z=Z)
