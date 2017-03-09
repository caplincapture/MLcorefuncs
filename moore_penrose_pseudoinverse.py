'''
Compute the (Moore-Penrose) pseudo-inverse of a matrix.
The following example checks that a * a+ * a == a and a+ * a * a+ == a+:
'''

a = np.random.randn(9, 6)
B = np.linalg.pinv(a)
np.allclose(a, np.dot(a, np.dot(B, a)))

np.allclose(B, np.dot(B, np.dot(a, B)))