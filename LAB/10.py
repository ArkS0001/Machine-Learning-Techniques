import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Regression function
def locally_weighted_regression(X, y, tau):
    def kernel(x, xi):
        return np.exp(-np.sum((x - xi)**2) / (2 * tau**2))

    m, n = X.shape
    theta = np.zeros(n)
    y_pred = np.zeros(m)
    
    for i in range(m):
        weights = np.array([kernel(X[i], X[j]) for j in range(m)])
        W = np.diag(weights)
        theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
        y_pred[i] = X[i] @ theta

    return y_pred

# Generate sample data
X = np.array([[1, 1.1], [1, 2.2], [1, 3.3], [1, 4.4], [1, 5.5]])
y = np.array([1, 2, 3, 4, 5])

# Fit and predict
tau = 0.5
y_pred = locally_weighted_regression(X, y, tau)

# Plotting
plt.scatter(X[:, 1], y, color='blue', label='Data Points')
plt.plot(X[:, 1], y_pred, color='red', label='LWR Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
