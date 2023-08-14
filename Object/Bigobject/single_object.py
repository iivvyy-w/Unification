import numpy as np
from scipy.integrate import odeint


class Object:

    def __init__(self, position, mass, velocity):
        self.position = position
        self.mass = mass
        self.velocity = velocity

    def distance(self, other):
        pos1 = self.position
        pos2 = other.position
        return np.linalg.norm(pos1 - pos2)

    def graviforce(self, other):
        G = 6.674e-11
        r = self.distance(other)
        F = G * self.mass * other.mass / (r**2)
        Fsplit = F / r * (other.position - self.position)
        return (F, Fsplit)


def generate(list0):
    position = np.random.uniform(-200000, 200000, (1, 3))
    for r in range(3):
        if position[0][r] < 0:
            position[0][r] -= np.random.uniform(100000, 300000)
        else:
            position[0][r] += np.random.uniform(100000, 300000)
    mass = 200000 + np.random.uniform(0, 200000)
    velocity = np.random.uniform(-2000, 2000, (1, 3))
    new = Object(position, mass, velocity)
    list0.append(new)


def solve(list0, n, dt=np.linspace(0, 1, 10)):
    target = list0[n]
    pos = target.position[0]
    vel = target.velocity[0]

    def model(current, t):
        x, y, z, dxdt, dydt, dzdt = current

        d2xdt2 = np.random.uniform(-1000, 1000)  # psychic force
        d2ydt2 = np.random.uniform(-1000, 1000)
        d2zdt2 = np.random.uniform(-1000, 1000)
        for order in range(len(list0)):
            if order != n:
                d2xdt2 += Object.graviforce(target, list0[order])[1][0][0]
                d2ydt2 += Object.graviforce(target, list0[order])[1][0][1]
                d2zdt2 += Object.graviforce(target, list0[order])[1][0][2]
        d2xdt2 = d2xdt2 / target.mass
        d2ydt2 = d2ydt2 / target.mass
        d2zdt2 = d2zdt2 / target.mass
        return [dxdt, dydt, dzdt, d2xdt2, d2ydt2, d2zdt2]

    initial = [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]
    result = odeint(model, initial, dt)
    new = result[-1]
    list0[n] = Object(np.array([[new[0], new[1], new[2]]]),
                      target.mass, np.array([[new[3], new[4], new[5]]]))
    return result
