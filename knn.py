"""
Implementation of the knn algorithm for our dataset using the scikit-learn library. Saves graphical output to the graph folder 
For the purpose of this project, we used different K sizes for our implementation. The ouputs can be found in the graph folder 
with each file labeled as 'k'-nn.pdf e.g. 5-nn.pdf for the 5 nearest neighbors output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

