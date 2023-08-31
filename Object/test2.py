import numpy as np
import matplotlib.pyplot as plt
from Bigobject import Object, gene_par, solve


H = Object(np.array([[0, 0, 0]]), 60, np.array([[0, 0, 0]]))
list0 = [H]


ax = plt.subplot(projection = '3d')

for n in range(10):
    gene_par(list0)
    ax.scatter(list0[-1].position[0][0],
               list0[-1].position[0][1],
               list0[-1].position[0][2], c='blue', marker='.',)
    plt.show

for r in range(2):
    for n in range(10):
        result = solve(list0, n+1, dt=np.linspace(0, 0.5, 20)).T
        ax.plot(result[0], result[1], result[2], c='orange', marker = '.')

x, y, z = np.indices((2, 2, 2))
filled = np.ones((1, 1, 1))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.voxels(x, y, z, filled = filled, alpha = 0.1)

ax.scatter(0, 0, 0, c='red', marker='x')
plt.show()
