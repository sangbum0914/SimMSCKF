import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(10)
a = np.random.randn(3, 1000)
b = np.random.randn(3, 1000)

diff = b - a

