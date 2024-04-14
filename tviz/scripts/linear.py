# %%
from typing import Any, Literal

import numpy as np
import plotly.io as pio
from plotly import graph_objs as go
from scipy import linalg

pio.renderers.default = "browser"


# %%
def create_matrix_family(
    matrix: np.ndarray[float, Any], timesteps: int = 100
) -> np.ndarray[float, Any]:
    result = np.zeros((timesteps, matrix.shape[0], matrix.shape[1]))
    log_M = linalg.logm(matrix)
    for i, a in enumerate(np.linspace(0, 1, timesteps)):
        result[i] = linalg.expm(a * log_M)
    return result


def apply_to_points(points: np.ndarray[float, Any], function: Any) -> np.ndarray[float, Any]:
    return np.array([function(point) for point in points])


def create_axes(
    xvals: np.ndarray[float, Any], yvals: np.ndarray[float, Any], npoints: int = 21
) -> np.ndarray[float, Any]:
    # return an array of size (len(xvals) + len(yvals), npoints, 2)
    # containing all the points in each line of the grid. There should be a line from ymin to ymax
    # for each xval, and a line from xmin to xmax for each yval.
    xmin = np.min(xvals)
    xmax = np.max(xvals)
    xticks = np.linspace(xmin, xmax, npoints)

    ymin = np.min(yvals)
    ymax = np.max(yvals)
    yticks = np.linspace(ymin, ymax, npoints)

    out = []
    for x in xvals:
        out.append(np.array([[x, y] for y in yticks]))
    for y in yvals:
        out.append(np.array([[x, y] for x in xticks]))
    return np.array(out)


def matrix_on_axes(
    matrix: np.ndarray[float, Any],
    axes: np.ndarray[float, Any],
    timesteps: int = 101,
) -> np.ndarray[float, Any]:
    # matrix_family is an output of smooth_matrix_application
    # axes is an output of create_axes
    # output is an array of size (len(matrix_family), len(axes), 2)
    # The first index is timestep, the second is line index
    matrix_family = create_matrix_family(matrix, timesteps=timesteps)
    return np.einsum("tij, lnj -> tlni", matrix_family, axes)


def relu_family_on_axes(
    axes: np.ndarray[float, Any], timesteps: int = 101
) -> np.ndarray[float, Any]:
    gradients = np.linspace(1, 0, timesteps)
    out = np.zeros((timesteps, axes.shape[0], axes.shape[1], axes.shape[2]))
    out[:, axes >= 0] = axes[axes > 0]
    out[:, axes < 0] = gradients[:, None, None] * axes[axes < 0]
    return out


def generate_axes_timeseries(
    operations: list[np.ndarray[float, Any] | Literal["relu"]],
    axes: np.ndarray[float, Any],
    timesteps_per_operation: int = 101,
) -> np.ndarray[float, Any]:
    # operations is a list of matrices and "relu" strings. Each matrix is applied to the axes, and
    # each "relu" string is applied to the axes with a ReLU nonlinearity. The result is a timeseries
    # of axes.
    out = [axes]
    for operation in operations:
        if operation == "relu":
            out.append(relu_family_on_axes(out[-1], timesteps=timesteps_per_operation))
        else:
            out.append(matrix_on_axes(operation, out[-1], timesteps=timesteps_per_operation))
    return np.array(out)


def plot_axes_timeseries(axes_timeseries: np.ndarray[float, Any]) -> None:
    # axes_timeseries is an output of generate_axes_timeseries for each timestep (first axis), plot
    # all the lines (second axis) as defined by their points (third axis). Use plotly with an
    # interactive slider labeled by the timestep.
    fig = go.Figure()
    num_lines = len(axes_timeseries[0])
    for matrix in axes_timeseries:
        for line in matrix:
            x, y = line.T
            fig.add_trace(
                go.Scatter(visible=False, x=x, y=y, mode="lines", line=dict(width=2, color="black"))
            )
    for i in range(num_lines):
        fig.data[i].visible = True

    # add slider
    steps = []
    for i in range(len(axes_timeseries)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}, {"title": f"Step {i}"}],
        )
        for j in range(num_lines):
            step["args"][0]["visible"][j + i * num_lines] = True
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": "Step: "}, steps=steps)]
    fig.update_layout(sliders=sliders)

    fig.show()


# %%
matrix = np.array([[1, 2], [-1, 1]])
matrix_family = smooth_matrix_application(matrix, timesteps=200)
xvals = np.linspace(-2, 2, 11)
yvals = np.linspace(-2, 2, 11)
axes = create_axes(xvals, yvals)
axes_timeseries = matrix_on_axes(matrix_family, axes)
plot_matrix_on_axes(axes_timeseries)

# %%
print(pio.renderers)
# %%
