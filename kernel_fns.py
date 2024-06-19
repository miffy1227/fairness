import numpy as np

def rbf_kernel(X, Y=None, gamma=None):
    if gamma in [None, 'scale']:
        gamma = 1.0 / X.shape[1]
    if Y is None:
        Y = X
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)[:, np.newaxis]
    XY = np.dot(X, Y.T)
    K = -2 * XY
    K += XX
    K += YY.T
    np.maximum(K, 0, out=K)
    if X is Y:
        np.fill_diagonal(K, 0)
    K = K.astype(float)  # 행렬의 자료형을 float로 변환
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K

def linear_kernel(x, y):
    if y is None:
        y = x
    return np.dot(x, y.T)

def poly_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = np.dot(X, Y.T)
    gamma = float(gamma)  # gamma를 부동소수점으로 변환
    K = K.astype(float)  # K를 부동소수점으로 변환
    K *= gamma
    K += coef0
    K **= degree
    return K

def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    if gamma in [None, 'scale']:
        gamma = 1.0 / X.shape[1]
    K = np.dot(X, Y.T)
    gamma = float(gamma)  # gamma를 부동소수점으로 변환
    K = K.astype(float)  # K를 부동소수점으로 변환
    K *= gamma
    K += coef0
    np.tanh(K, K)  # compute tanh in-place
    return K

def calculate_kernel(X, Y=None, kernel='linear', **kwds):
    assert kernel in ['linear', 'rbf', 'poly', 'sigmoid']
    metric = eval(kernel + '_kernel')
    if kernel == 'linear':
        out = linear_kernel(X, Y)
    else:
        out = metric(X, Y, **kwds)
    return out

# 간단한 데이터 생성
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

# 각 커널 함수 테스트
print("Linear Kernel:\n", calculate_kernel(X, Y, kernel='linear'))
print("RBF Kernel:\n", calculate_kernel(X, Y, kernel='rbf', gamma=0.1))
print("Polynomial Kernel:\n", calculate_kernel(X, Y, kernel='poly', degree=2, gamma=0.1, coef0=1))
print("Sigmoid Kernel:\n", calculate_kernel(X, Y, kernel='sigmoid', gamma=0.1, coef0=1))
