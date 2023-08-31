import numpy as np
import matplotlib.pyplot as plt
from Bigobject import Object, gene_par, trajectory
np.random.seed(2003)

H = Object(np.array([[0, 0, 0]]), 60, np.array([[0, 0, 0]]))
list0 = [H]
gene_par(list0)
print(list0[1].position)
print(trajectory(list0[1], 0, 1, np.random.uniform(-1e-22, 1e-22)))
