import numpy as np
from scipy.integrate import odeint
from math import sqrt
np.random.seed(2023)


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


def gene_par(list0):
    position = np.random.uniform(0, 1, (1, 3))
    mass = 3.35e-23
    velocity = np.random.uniform(-1, 1, (1, 3))
    new = Object(position, mass, velocity)
    list0.append(new)


def solve(list0, n, dt=np.linspace(0, 1, 10)):
    target = list0[n]
    pos = target.position[0]
    vel = target.velocity[0]

    def model(current, t):
        x, y, z, dxdt, dydt, dzdt = current

        d2xdt2 = np.random.uniform(-1, 1)  # psychic force
        d2ydt2 = np.random.uniform(-1, 1)
        d2zdt2 = np.random.uniform(-1, 1)

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


def newsolve(list0, dt=np.linspace(0, 1, 10)):
    len0 = len(list0)

    def newmodel(current, t):
        variables = {}
        resultlist = []
        r = int(len(current)/2)
        for i in range(r):
            resultlist.append(current[r+i])

        for i in range(len0):
            var_dxdt2 = f"dx{i}dt2"
            var_dydt2 = f"dy{i}dt2"
            var_dzdt2 = f"dz{i}dt2"
            variables[var_dxdt2] = np.random.uniform(-1, 1)  # psychic force
            variables[var_dydt2] = np.random.uniform(-1, 1)
            variables[var_dzdt2] = np.random.uniform(-1, 1)

        for all in range(len0):
            target = list0[all]

            for order in range(len0):
                if order != all:
                    variables[f"dx{all}dt2"] += Object.graviforce(target, list0[order])[1][0][0]
                    variables[f"dy{all}dt2"] += Object.graviforce(target, list0[order])[1][0][1]
                    variables[f"dz{all}dt2"] += Object.graviforce(target, list0[order])[1][0][2]

            variables[f"dx{all}dt2"] = variables[f"dx{all}dt2"] / target.mass
            variables[f"dy{all}dt2"] = variables[f"dy{all}dt2"] / target.mass
            variables[f"dz{all}dt2"] = variables[f"dz{all}dt2"] / target.mass

            resultlist.append(variables[f"dx{all}dt2"])
            resultlist.append(variables[f"dy{all}dt2"])
            resultlist.append(variables[f"dz{all}dt2"])

        return resultlist

    initial = []
    for i in range(len0):
        target2 = list0[i]
        pos = target2.position[0]
        initial = initial + [pos[0], pos[1], pos[2]]
    for i in range(len0):
        target2 = list0[i]
        vel = target2.velocity[0]
        initial = initial + [vel[0], vel[1], vel[2]]

    result = odeint(newmodel, initial, dt, mxstep=1000)
    return result


def trajectory(object, axis, t, psy): # axis x, y, z = 0, 1, 2
    p = object.position[0][axis]
    v = object.velocity[0][axis]
    f = psy
    m = object.mass
    traj = f / (2*m) * t**2 + v * t + p
    vel = f/m*t + v
    if traj > 1:
        ht = (-2 * m * v + sqrt((2*m*v)**2 - 8*f*m*(p-1))) / (2*f)
        object.position[0][axis] = 1
        object.velocity[0][axis] = -(f/m * ht + v)
        trajectory(object, axis, t-ht, psy)
    if traj < 0:
        ht = (-2 * m * v - sqrt((2*m*v)**2 - 8*f*m*(p))) / (2*f)
        object.position[0][axis] = 0
        object.velocity[0][axis] = -(f/m * ht + v)
        trajectory(object, axis, t-ht, psy)
    object.position[0][axis] = traj
    object.velocity[0][axis] = vel
    return traj