from ast import Index
from inspect import trace
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

import FindBoundary as fb


def f(x):
    return -3 * x**3 + 4 * x**2 + 5 * x + 2 * np.exp(x)


def isInside(Point):
    x, y = Point
    if y > f(x) and x ** 2 + y**2 < 20**2:
        return 1
    return 0


In = np.array([0, 15])
Out = np.array([0, 0])
StepWidth = 0.1
StepLength = 0.5
PairTol = 0.1
tracer = fb.BoundaryTracer(In, Out, isInside, StepWidth, StepLength, PairTol)
Start = tracer.GetNewestMiddle()
fig = plt.figure()
ax = fig.gca()
xx = np.linspace(-20, 20, 10001)
# ax.plot(xx, f(xx), linestyle=":")
(InPlot,) = ax.plot([0, 0], [0, 1], linewidth=1, linestyle="--")
(OutPlot,) = ax.plot([0, 0], [0, 1], linewidth=1, linestyle="--")
InPlot.set_data(np.array(tracer.In).T)
OutPlot.set_data(np.array(tracer.Out).T)
ax.set_xlim([-10, 10])
ax.set_ylim([0, 20])
LeftStart = False


def autoscale(padding_percentage=5):
    InX, InY = InPlot.get_data()
    OutX, OutY = OutPlot.get_data()
    xmax = max(max(InX), max(OutX))
    ymax = max(max(InY), max(OutY))
    xmin = min(min(InX), min(OutX))
    ymin = min(min(InY), min(OutY))

    w = ax.get_window_extent(None).width
    h = ax.get_window_extent(None).height

    if (xmax - xmin) / w > (ymax - ymin) / h:
        ymax = (ymax + ymin) / 2 + (h / w) * (xmax - xmin) / 2
        ymin = (ymax + ymin) / 2 - (h / w) * (xmax - xmin) / 2
    else:
        xmax = (xmax + xmin) / 2 + (w / h) * (ymax - ymin) / 2
        xmin = (xmax + xmin) / 2 - (w / h) * (ymax - ymin) / 2

    r = padding_percentage / 100
    ax.set_xlim(xmin - r * (xmax - xmin), xmax + r * (xmax - xmin))
    ax.set_ylim(ymin - r * (ymax - ymin), ymax + r * (ymax - ymin))


i = 0
filenames = []


def save_frame():
    global i
    plt.pause(1e-9)
    # create file name and append it to a list
    filename = f"{i}.png"
    filenames.append(filename)

    # save frame
    plt.savefig(filename)
    i = i + 1


while True:
    save_frame()
    # plt.pause(0.2)
    tracer.TakeStep(Tighten=False)
    InPlot.set_data(np.array(tracer.In).T)
    OutPlot.set_data(np.array(tracer.Out).T)
    autoscale()
    # plt.pause(0.2)
    save_frame()
    tracer.TightenNewestPair()
    tracer.UpdateDirection()
    InPlot.set_data(np.array(tracer.In).T)
    OutPlot.set_data(np.array(tracer.Out).T)
    autoscale()

    M = tracer.GetNewestMiddle()
    if LeftStart and np.linalg.norm(Start - M) < 0.6 * StepLength:
        print("Tracing done, back to the start")
        tracer.In.append(tracer.In[0])
        tracer.Out.append(tracer.Out[0])
        InPlot.set_data(np.array(tracer.In).T)
        OutPlot.set_data(np.array(tracer.Out).T)
        autoscale()
        save_frame()
        break
    elif not LeftStart and np.linalg.norm(Start - M) > 5:
        LeftStart = True


# build gif
with imageio.get_writer("compl.gif", mode="I") as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)
    # repeat the last image
    for k in range(50):
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)
