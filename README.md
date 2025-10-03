# Pearson Correlation Analysis

## Dataset
We are given 8 pairs of (X, Y):

```
(-5,  2)
(-1,  1)
( 3,  1)
(-3, -1)
(-4, -4)
( 1, -2)
( 5, -3)
( 7, -2)
```

The goal is to compute **Pearson’s correlation coefficient** between X and Y, visualize the relationship, and interpret the result.

---

## Method

Pearson’s correlation coefficient \( r \) is defined as:

\[
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}
         {\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \; \sum_{i=1}^{n} (y_i - \bar{y})^2}}
\]

Where:
- \( \bar{x} \) is the mean of X,
- \( \bar{y} \) is the mean of Y,
- n is the number of data points.

We computed r in two ways:
1. **Manual calculation** using the formula above.
2. **`scipy.stats.pearsonr`** function (which also gives a p-value).

We also generated a scatter plot with the regression line for visualization.

---

## Python Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import pearsonr, linregress

# Data
data = [(-5, 2), (-1, 1), (3, 1), (-3, -1),
        (-4, -4), (1, -2), (5, -3), (7, -2)]
x = np.array([d[0] for d in data], dtype=float)
y = np.array([d[1] for d in data], dtype=float)

# Manual calculation
mean_x, mean_y = x.mean(), y.mean()
num = ((x - mean_x) * (y - mean_y)).sum()
den = sqrt(((x - mean_x)**2).sum() * ((y - mean_y)**2).sum())
r_manual = num / den

# Scipy calculation
r, p_value = pearsonr(x, y)

# Regression line
slope, intercept = np.polyfit(x, y, 1)

# Results
print("Manual Pearson r:", r_manual)
print("Scipy Pearson r:", r)
print("p-value:", p_value)
print(f"Regression line: Y = {slope:.3f} * X + {intercept:.3f}")

# Scatter plot
plt.scatter(x, y, label="Data points")
xx = np.linspace(x.min() - 1, x.max() + 1, 200)
yy = slope * xx + intercept
plt.plot(xx, yy, 'r-', label="Regression line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter plot of X vs Y")
plt.legend()
plt.grid(True)
plt.show()
```

---

## Results

- **Manual Pearson r**: -0.2597  
- **Scipy Pearson r**: -0.2597  
- **p-value**: 0.5345  
- **Regression line**: \( Y = -0.127 X - 0.952 \)

---

## Interpretation

- The correlation coefficient (r ≈ -0.26) indicates a **weak negative linear relationship** between X and Y.  
- The p-value (> 0.05) shows the result is **not statistically significant**, so we cannot conclude there is a reliable linear relationship.  
- The regression line has a slight downward slope, which matches the negative correlation.  
- **Conclusion:** This dataset shows only a weak, non-significant negative correlation. Correlation does not imply causation, and with such a small sample size, results should be interpreted cautiously.

---

## Visualization

The scatter plot shows the data points and the fitted regression line:

![Scatter Plot](corr_scatter.png)

---
