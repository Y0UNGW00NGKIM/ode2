import numpy as np
import matplotlib.pyplot as plt

def RK1Solve(f, y0, nsteps, x0, xmax) -> np.array:
    h = (xmax - x0) / nsteps  # step size
    x = x0                     # independent variable
    y = y0                     # dependent variable to plot vs x
    points = [(x0, y0)]        # store points for plotting

    for i in range(nsteps - 1):
        k1 = h * f(x, y)
        y = y + k1
        x += h
        points.append((x, y))
    return np.array(points)

def RK2Solve(f, y0, nsteps, x0, xmax) -> np.array:
    h = (xmax - x0) / nsteps  # step size
    x = x0                     # independent variable
    y = y0                     # dependent variable to plot vs x
    points = [(x0, y0)]        # store points for plotting

    for i in range(nsteps - 1):
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        y = y + k2
        x += h
        points.append((x, y))
    return np.array(points)

def RK4Solve(f, y0, nsteps, x0, xmax) -> np.array:
    h = (xmax - x0) / nsteps
    x = x0
    y = y0
    points = [(x0, y0)]

    for i in range(nsteps - 1):
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x = x + h
        points.append((x, y))
    return np.array(points)

def deq_241(x, y):
    return x - y

def exact_241(x):
    return x - 1.0 + 4.0 * np.exp(-x)

def deq_250(x, y):
    return 2.0 * y + x * np.exp(x)

def exact_250(x):
    return -(x + 1.0) * np.exp(x)

nsteps = 30
x0 = 0.0
xmax = 3.0

tg1 = RK1Solve(deq_241, 3.0, nsteps, x0, xmax)
tg2 = RK2Solve(deq_241, 3.0, nsteps, x0, xmax)
tg4 = RK4Solve(deq_241, 3.0, nsteps, x0, xmax)

x_exact = np.linspace(x0, xmax, 300)
y_exact = exact_241(x_exact)

x_nodes = tg1[:, 0]
y_exact_nodes = exact_241(x_nodes)

diff_rk1 = tg1[:, 1] - y_exact_nodes
diff_rk2 = tg2[:, 1] - y_exact_nodes
diff_rk4 = tg4[:, 1] - y_exact_nodes

max_rk1 = np.max(np.abs(diff_rk1))
max_rk2 = np.max(np.abs(diff_rk2))
max_rk4 = np.max(np.abs(diff_rk4))

rms_rk1 = np.sqrt(np.mean(diff_rk1 * diff_rk1))
rms_rk2 = np.sqrt(np.mean(diff_rk2 * diff_rk2))
rms_rk4 = np.sqrt(np.mean(diff_rk4 * diff_rk4))

textstr = (
    "Max |error|:\n"
    f"RK1 = {max_rk1:.2e}\n"
    f"RK2 = {max_rk2:.2e}\n"
    f"RK4 = {max_rk4:.2e}\n"
    "RMS error:\n"
    f"RK1 = {rms_rk1:.2e}\n"
    f"RK2 = {rms_rk2:.2e}\n"
    f"RK4 = {rms_rk4:.2e}"
)

plt.figure(figsize=(10, 6))
plt.plot(tg1[:, 0], tg1[:, 1], 'r^', markersize=8, label='RK1 Solution')
plt.plot(tg2[:, 0], tg2[:, 1], 'g^', markersize=8, label='RK2 Solution')
plt.plot(tg4[:, 0], tg4[:, 1], 'bo', markersize=4, label='RK4 Solution')
plt.plot(x_exact, y_exact, 'k--', label='Exact Solution')

plt.title("ODE 241: y' + y = x, y(0)=3")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

ax = plt.gca()
ax.text(
    0.02,
    0.98,
    textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.savefig("RK4.pdf")
print("close plot window to exit")
plt.show()
