---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="oVtJNlNr0N5L" -->
# Plotting with Python

This notebook is written in parallel to an article for my website. We will
introduce the basics of creating plots in Python using Matplotlib .
<!-- #endregion -->

<!-- #region id="ZTQqcPtuhffL" -->
## Initial Setup

We'll begin with the line `%matplotlib inline`. This is specific to notebooks
and tells the notebook to render matplotlib plots inline. We then import
the libraries we'll use throughout our examples. In this case, `numpy` and `matplotlib.pyplot`.
<!-- #endregion -->

```python id="CtKnXemx0lBt"
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region id="D2c8KdHU0Zpw" -->
## A Simple Plot

Our first plot is a simple sine plot using `np.sin`. First we use `np.linspace` to
create a list (or NumPy array in this case) of all our X points. In this case, an evenly spaced list from $0$
to $4\pi$ with 100 points. We then generate our Y points by calling `np.sin` on
the X list. Finally, we can use `plt.plot(x, y)` to plot the results.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="ux71wOC_0m32" outputId="9c5d65e2-6714-4a6d-8520-d5698eff6182"
x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

<!-- #region id="7TD17qWA1qcC" -->
## Multiple Lines

Now we'll plot multiple lines on a single chart. In this case,
- $y_1 = 0.5 \sin{(x)}$
- $y_2 = 0.5 \cos{(x)}$
- $y_3 = 2 \sin{(x)}$
- $y_4 = 2 \cos{(x)}$.

We have the option of calling `plt.plot` once as `plt.plot(x, y1, x, y2, x, y3, x, y4)` or once for each plot (shown below).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="dZvOI1vs1vaL" outputId="35e45ca5-8545-49eb-c645-bf5517e932c9"
x = np.linspace(0, 4*np.pi, 100)
y1 = 0.5*np.sin(x)
y2 = 0.5*np.cos(x)
y3 = 2*np.sin(x)
y4 = 2*np.cos(x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.show()
```

<!-- #region id="-CcQCh1VhffO" -->
## Line Styles
We can pass additional arguments to `plot()` to specify the line style. The first way to provide a *format string*. That might look something like `'-b'` or `'--sy'`. We can specify the line style, marker style, and color with this format string. For example `-` tells matplotlib to make the line solid, `--` is dashed, and `:` is dotted. We can also define the marker style. In our `'--sy'` example, `s` declares that the marker should be square. The full list of marker codes can be found [here](https://matplotlib.org/stable/api/markers_api.html). Finally, we can specify the color. The following color codes are available:

- `b` is blue
- `r` is red
- `g` is green
- `c` is cyan
- `m` is magenta
- `y` is yellow
- `k` is black

If we want to customize our plot styles further, we can use a variety of keyword arguments such as `markersize` and `linewidth` to modify the plot style. The full list of options is available [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).
<!-- #endregion -->

```python id="F1-DzeV6hffP" outputId="5cd0d928-a0bf-43a0-8212-f2d3bd8789cc"
# Solid, blue line
plt.plot(x, y1, '-b')

# Red, dashed line
plt.plot(x, y2, '--r')

# Dotted, green line
plt.plot(x, y3, ':g', linewidth=1.5)

# We call also use keyword arguments
plt.plot(
    x, y4, '-ok',
    markersize=6,
    markeredgewidth=0.75,
    markeredgecolor=[0.1, 0.1, 0.3, 0.9],
    markerfacecolor=[0.5, 0.5, 0.6, 0.5]
)
plt.show()
```

<!-- #region id="ivW4QeLL1hNi" -->
## Using Stylesheets

If we want to change a lot more about our plot with a lot less code, we can use stylesheets. Matplotlib comes with several predefined stylesheets. We can use `plt.style.available` to see the list of available style sheets.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LaLOQZhY0LKW" outputId="f8b9abef-a3eb-46ab-bd26-bba3541a06de"
print(plt.style.available)
```

<!-- #region id="zXHvDoC0hffR" -->
In this case, we'll combine a few style sheets that set the plot size, grid colors, and line colors to create a graph with a clean style without having to specify the style of each line.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 441} id="JaSDxGmT1nHc" outputId="6bbd59c4-038e-4e28-f987-d3b4a0cd45d8"
plt.style.use('seaborn-talk')
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-deep')

x = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = 2*np.sin(x)
y4 = 2*np.cos(x)

plt.plot(x, y1, x, y2, x, y3, x, y4)
plt.show()
```

<!-- #region id="rldLwsOphffR" -->
We can also create our own style sheet. The one for this example can be found at [./mystyle.mplstyle](./mystyle.mplstyle). This stylesheet defines the plto size, custom colors, and a bit more.
<!-- #endregion -->

```python id="6vS3I1Xd5xr2" outputId="e663f5bf-a09b-46c9-af2f-736d7abac882"
plt.style.use('mystyle.mplstyle')
plt.plot(x, y1, x, y2, x, y3, x, y4)
plt.show()
```

<!-- #region id="qTnwZAfDhffS" -->
## Scatter Plots
Now we can use `plt.scatter` to plot some noisy data. According to [Jake VanderPlas](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/04.02-Simple-Scatter-Plots.ipynb), `plt.plot` is much more efficient than `plt.scatter` for larger data sets.
<!-- #endregion -->

```python id="Tn6F8EUHhffS" outputId="fe05bdb8-c3f7-436a-84f8-6d8fdee2e8e4"
x = np.linspace(0, 8, 100)
y = 2*x

# Add noise
noisy = [point + 5*np.random.random() - 5*np.random.random() for point in y]

plt.scatter(x, noisy, marker='o', s=2)
plt.show()
```

<!-- #region id="iN6rTanzhffT" -->
## Best Fit Line
Now we can use NumPy's [polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) to generate a polynomial fit line. We'll also add some text to the plot to show the equation of the line and the R-squared value.
<!-- #endregion -->

```python id="9AoBjUEshffT" outputId="5857b3f2-ae7a-4882-d941-b9e3658f3046"
# Fit line
degree = 1
fit = np.polyfit(x, y, degree)
bfline = fit[0]*x + fit[1]

# R-squared
correlation_matrix = np.corrcoef(x, y)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

# R-squared
p = np.poly1d(fit)
yhat = p(x)
ybar = np.sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)
sstot = np.sum((y - ybar)**2)
r_squared = ssreg / sstot

# Plot data points and line fit
plt.scatter(x, noisy, marker='o', s=2)
plt.plot(x, bfline)

# Generate labels and show plot
m = f'{fit[0]:.3f}'
b = f'{fit[1]:.3f}'
op = '' if b.startswith('-') else '+'
eq_label = f'$y = {m}x {op} {b}$'
r_label = f'$R^2 = {r_squared:.4f}$'
plt.text(0.5, 14.5, eq_label, fontsize=8)
plt.text(0.5, 12, r_label, fontsize=8)
plt.show()
```

<!-- #region id="qTlC2zWNhffT" -->
## Subplots
We can also generate and plot multiple graphs in a single figure using subplots. The simplest way to do this is call `plt.subplot()`. Subplot takes three arguments: the number of rows, the number of columns, and the position of the next plot.
<!-- #endregion -->

```python id="5I-0_cp-hffT" outputId="ff5caeb6-f58b-478d-c2c5-924fbd112338"
x = np.linspace(0.1, 10, 100)

plt.subplot(2,1,1)
plt.plot(x, x)

plt.subplot(2,1,2)
plt.plot(x, np.log(x))

plt.show()
```

<!-- #region id="RbtvqGuohffU" -->
### FFT Example
Let's look at another example using a Fast-Fourier Transform (FFT). This example is based on [this page from UC Berkeley](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html). First we need to generate an aggregate signal. In this case, our signal consists of three sine waves of varying frequencies and amplitudes.
<!-- #endregion -->

```python id="Zr6MDIzOhffU"
# Complex signal
sr = 2000 # sampling rate
ts = 1.0/sr # sampling interval
t = np.arange(0,2,ts)

freq = 1
A = 3
x = A*np.sin(2*np.pi*freq*t)

freq = 3.5
A = 1.5
x += A*np.sin(2*np.pi*freq*t)

freq = 6
A = 0.5
x += A* np.sin(2*np.pi*freq*t)

freq = 9.5
A = 1.5
x += A* np.sin(2*np.pi*freq*t)

freq = 0.5
A = 1
x += A* np.sin(2*np.pi*freq*t)
```

<!-- #region id="Lg4HWgYyhffV" -->
Then, we can compute the fast-fourier transform (FFT) of the plot using NumPy's `fft` module.
<!-- #endregion -->

```python id="qaHVOK-khffV" outputId="71f77811-31d5-416d-9230-637fd6daa00f"
X = np.fft.fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T
F = np.abs(X)
print(F)
```

<!-- #region id="UpzExk3-hffV" -->
Then we will create a figure with subplots (2 rows and 1 column) and plot the time series signal on the top set of axes and the frequency domain on the bottom.
<!-- #endregion -->

```python id="gAOo1bH0hffV" outputId="b8f0f6b5-9d76-4337-a4b1-e5079efbd7b6"
fig, axs = plt.subplots(2, 1)

axs[0].plot(t, x)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

axs[1].stem(freq, F, 'r', markerfmt=" ", basefmt="-r")
axs[1].set_xlim(0, 10)
axs[1].set_xlabel('Freq (Hz)')
axs[1].set_ylabel('X(freq)')
axs[1].grid(True)

plt.tight_layout()
plt.show()

```

<!-- #region id="AU7AufmthffV" -->
## Formatting
With some basics of plotting covered, its worth introducing some formatting basics to make your plots a bit more professional and detailed.

### Titles and Axes Labels
Matplotlib provides `.title()`, `.xlabel()`, and `.ylabel()` functions to add plot titles and axes labels.
<!-- #endregion -->

```python id="TXVy6oIhhffW" outputId="ff150437-a71e-40c5-d124-d6d118bc60b2"
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('A Sine Wave')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.show()
```

<!-- #region id="92q8XsIYhffW" -->
### Legends
The `.legend()` function allows you to add a legend to the plot.
<!-- #endregion -->

```python id="ALBdNpmuhffW" outputId="e40435f9-1df4-47ed-8e5c-8f49a346423e"
#plt.style.use('classic')
#plt.style.use('seaborn')
#plt.style.use('seaborn-paper')

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, x, y2)
plt.title('Sine and Cosine Waves')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.legend(['Sin', 'Cos'], fontsize=10)
plt.show()
```

<!-- #region id="9-wdru_ThffX" -->
## Saving Figures
We can use `plt.savefig()` to save the current figure. In the example below, we style and generate a plot, then call `plt.gcf()` to **g**et the **c**urrent **f**igure, then adjust its size, and use `savefig()` to save the figure as a JPEG image.
<!-- #endregion -->

```python id="geHwdNaehffX" outputId="ada49ca7-323b-444e-b043-df39f50f859e"
plt.style.use('classic')
plt.style.use('seaborn')
plt.style.use('seaborn-paper')

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, x, y2)
plt.title('Sine and Cosine Waves')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.legend(['Sin', 'Cos'], fontsize=10)


# Ge the current figure and update its size
fig = plt.gcf()
fig.set_size_inches(6, 4)

# Save the figure
plt.savefig('output.jpg', dpi=300)
plt.show()
```

<!-- #region id="T8GiMi2DhffX" -->
## Additional Resources

This barely scratches the surface of what can be done with Matplotlib. For more examples, check out the [matplotlib example gallery](https://matplotlib.org/stable/gallery/index.html).
<!-- #endregion -->
