import os
import sys
import pickle 

#Third party
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

#Project
from ismetlib import sampler
from ismetlib import statistics
from ismetlib import plots

model = sampler.meteoroid_perihelion_model()

x = np.empty((6,100))
for ind in range(100):
    x[:, ind] = model(*np.random.randn(2).tolist())

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x[0,:], x[1,:], x[2,:])

ax = fig.add_subplot(122, projection='3d')
ax.scatter(x[3,:], x[4,:], x[5,:])


plt.show()