from Bigobject import Object, generate, solve
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2000)


E = Object(np.array([[0, 0, 0]]), 5.97e24, np.array([[0, 0, 0]]))
list0 = [E]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for n in range(10):
    generate(list0)
    ax.scatter(list0[-1].position[0][0],
               list0[-1].position[0][1],
               list0[-1].position[0][2], c='red', marker='x',)
    plt.show

for r in range(8):
    for n in range(10):
        result = solve(list0, n+1, dt=np.linspace(0, 1, 20)).T
        ax.plot(result[0], result[1], result[2], c='orange', marker = '.')

# result = solve(list0, 1, dt=np.linspace(0, 10, 300)).T
# ax.plot(result[0], result[1], result[2], marker = '.')

ax.scatter(list0[0].position[0][0],
           list0[0].position[0][1],
           list0[0].position[0][2], c='blue', marker='x')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# plt.savefig("test.jpg", dpi=150, bbox_inches='tight')
