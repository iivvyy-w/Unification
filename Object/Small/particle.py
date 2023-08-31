import numpy as np
from scipy.integrate import odeint
from Bigobject import Object

def gene_par(list0):
    position = np.random.uniform(0, 1, (1, 3))
    mass = 3.35e-23
    velocity = np.random.uniform(-1, 1, (1, 3))
    new = Object(position, mass, velocity)
    list0.append(new)

