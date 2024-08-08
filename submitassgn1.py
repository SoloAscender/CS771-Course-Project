import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y0_train, y1_train ):
################################
#  Non Editable Region Ending  #
################################

    X_mapped = my_map(X_train)

    # Initialize and train the model4
    model0 = LogisticRegression(C=100, tol=1, max_iter=1000)
    model1 = LogisticRegression(C=1, tol=1e-2, max_iter=1000)

    model0.fit(X_mapped, y0_train)
    model1.fit(X_mapped, y1_train)

    # Get the weights and biases
    W0 = model0.coef_[0]
    b0 = model0.intercept_[0]
    W1 = model1.coef_[0]
    b1 = model1.intercept_[0]

    return W0, b0, W1, b1


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

    n = X.shape[0]  # Number of challenges
    d = X.shape[1]  # Dimensionality of each challenge (32)
    feat = np.empty((X.shape[0], 2*X.shape[1]-1))

    for i in range(n):
        c = X[i]
        mapped_features = np.empty(2 * d - 1)

        # Original challenge bits
        mapped_features[:d] = c

        dd = 1 - 2 * c[:32]
        # dd = np.append(dd, 1 - 2 * c[31])
        cp = np.cumprod(dd[::-1])[::-1]
        
        mapped_features[d:] = cp[:31]
        feat[i] = mapped_features

    return np.array(feat)