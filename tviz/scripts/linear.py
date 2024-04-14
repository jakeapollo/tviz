# %%
from typing import Literal

import numpy as np
import plotly.io as pio
import torch
from plotly import graph_objs as go
from scipy import linalg

pio.renderers.default = "browser"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def create_matrix_family(matrix: torch.Tensor, timesteps: int = 100) -> torch.Tensor:
    result = torch.zeros((timesteps, matrix.shape[0], matrix.shape[1]))
    log_M = torch.tensor(linalg.logm(matrix.detach().numpy()))
    for i, a in enumerate(np.linspace(0, 1, timesteps)):
        result[i] = torch.linalg.matrix_exp(a * log_M)
    return result


# def apply_to_points(points: np.ndarray[float, Any], function: Any) -> np.ndarray[float, Any]:
#     return np.array([function(point) for point in points])


def create_axes(xvals: torch.Tensor, yvals: torch.Tensor, npoints: int = 21) -> torch.Tensor:
    # return an array of size (len(xvals) + len(yvals), npoints, 2)
    # containing all the points in each line of the grid. There should be a line from ymin to ymax
    # for each xval, and a line from xmin to xmax for each yval.
    xmin = xvals.min()
    xmax = xvals.max()
    xticks = torch.linspace(xmin, xmax, npoints)

    ymin = yvals.min()
    ymax = yvals.max()
    yticks = torch.linspace(ymin, ymax, npoints)

    out = []
    for x in xvals:
        out.append([[x, y] for y in yticks])
    for y in yvals:
        out.append([[x, y] for x in xticks])

    out = torch.tensor(out, device=device)
    assert out.shape == (len(xvals) + len(yvals), npoints, 2)
    return out


def matrix_on_axes(
    matrix: torch.Tensor,
    axes: torch.Tensor,
    timesteps: int = 101,
) -> torch.Tensor:
    # matrix_family is an output of smooth_matrix_application
    # axes is an output of create_axes
    # output is an array of size (len(matrix_family), len(axes), 2)
    # The first index is timestep, the second is line index
    matrix_family = create_matrix_family(matrix, timesteps=timesteps).to(device)
    return torch.einsum("tij, lnj -> tlni", matrix_family, axes)


def relu_family_on_axes(axes: torch.Tensor, timesteps: int = 101) -> torch.Tensor:
    assert len(axes.shape) == 3
    assert axes.shape[2] == 2
    gradients = torch.linspace(1, 0, timesteps, device=device)
    # out = torch.zeros((timesteps, axes.shape[0], axes.shape[1], axes.shape[2]), device=device)
    # print(out.shape, axes.shape, gradients.shape)
    # out[:, axes >= 0] = axes[axes > 0]
    negative_mask = (axes < 0).float()
    gradient_scale = torch.einsum("g, lnj -> glnj", gradients, negative_mask)
    out = axes * gradient_scale
    out[:, axes >= 0] = axes[axes >= 0]
    out = out.to(device)
    return out


def generate_axes_timeseries(
    operations: list[torch.Tensor | Literal["relu"]],
    axes: torch.Tensor,
    timesteps_per_operation: int = 101,
) -> torch.Tensor:
    # operations is a list of matrices and "relu" strings. Each matrix is applied to the axes, and
    # each "relu" string is applied to the axes with a ReLU nonlinearity. The result is a timeseries
    # of axes.
    assert len(axes.shape) == 3
    assert axes.shape[2] == 2
    print(axes.shape)
    out = axes.unsqueeze(0)
    print(out.shape)
    for operation in operations:
        if operation == "relu":
            print("out-1", out[-1].shape)
            out = torch.concatenate(
                (out, relu_family_on_axes(out[-1], timesteps=timesteps_per_operation))
            )
            print(out.shape)
        else:
            assert isinstance(operation, torch.Tensor)
            out = torch.concatenate(
                (out, matrix_on_axes(operation, out[-1], timesteps=timesteps_per_operation))
            )
            print(out.shape)
    return torch.tensor(out)


def plot_axes_timeseries(axes_timeseries: torch.Tensor) -> None:
    # axes_timeseries is an output of generate_axes_timeseries for each timestep (first axis), plot
    # all the lines (second axis) as defined by their points (third axis). Use plotly with an
    # interactive slider labeled by the timestep.
    fig = go.Figure()
    num_lines = len(axes_timeseries[0])
    axes_timeseries = axes_timeseries.cpu().detach().numpy()
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
    fig.update_layout(sliders=sliders, xaxis=dict(range=[-2, 2]), yaxis=dict(range=[-2, 2]))

    fig.show()


# %%
xvals = torch.linspace(-10, 10, 11)
yvals = torch.linspace(-10, 10, 11)
axes = create_axes(xvals, yvals)
matrix1 = torch.randn(2, 2)
matrix2 = torch.randn(2, 2)
axes_timeseries = generate_axes_timeseries([matrix1, "relu", matrix2], axes)

# %%
plot_axes_timeseries(axes_timeseries)

# %%
