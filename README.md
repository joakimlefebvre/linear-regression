# Linear Regression with Gradient Descent

## Overview

This Python script implements **Linear Regression** using **Gradient Descent**. It reads a dataset from a CSV file (`data.csv`), trains a model to fit a line (`y = mx + b`), and outputs the optimized values of `m` (slope) and `b` (intercept), along with the error at different stages.

## How It Works

The script follows these steps:

1. **Load Data**: Reads a CSV file containing two columns: `x` (input) and `y` (output).
2. **Define Cost Function**: Calculates Mean Squared Error (MSE) to evaluate model performance.
3. **Implement Gradient Descent**: Iteratively updates `m` and `b` using their partial derivatives.
4. **Train the Model**: Runs gradient descent for a fixed number of iterations.
5. **Output Results**: Prints the initial and final values of `m`, `b`, and error.

---

## Code Breakdown

### 1. Compute Error Function

```python
def compute_error(b, m, df):
    totalError = 0
    for i in range(len(df)):
        totalError += (df.y[i] - (m * df.x[i] + b)) ** 2
    return totalError / float(len(df))
```

- Computes the **Mean Squared Error (MSE)** between predicted and actual values.
- Formula: $f(b,m) = \frac{1}{N} \sum (y_i - (m x_i + b))^2$

### 2. Step Gradient Function
<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="https://github.com/user-attachments/assets/36851807-1371-4a52-81e0-91d2d4d93a40" width="400"/>
    <img src="https://github.com/user-attachments/assets/42e62417-3cc2-40e9-a1fd-25e9ba6808e7" width="400"/>
</div>


```python
def step_gradient(b, m, df, learning_rate):
    b_grad = 0
    m_grad = 0
    N = len(df)
    for i in range(N):
        x = df.x[i]
        y = df.y[i]
        b_grad += -(2 / N) * (y - ((m * x) + b))
        m_grad += -(2 / N) * x * (y - ((m * x) + b))
    new_b = b - (learning_rate * b_grad)
    new_m = m - (learning_rate * m_grad)
    return [new_b, new_m]
```

- Computes **partial derivatives** of the cost function with respect to `m` and `b`.
- $\frac{\partial f}{\partial b} = -\frac{2}{N} \sum \left( y_i - (mx_i + b) \right)$
- $\frac{\partial f}{\partial m} = -\frac{2}{N} \sum x \left( y_i - (mx_i + b) \right)$

- Updates `m` and `b` in the opposite direction of the gradient.

### 3. Gradient Descent Function

```python
def gradient_descent(df, initial_b, initial_m, learning_rate, max_iterations):
    b = initial_b
    m = initial_m
    for i in range(max_iterations):
        b, m = step_gradient(b, m, df, learning_rate)
    return [b, m]
```

- Runs **gradient descent** for a fixed number of iterations.
- Calls `step_gradient()` in each iteration to update parameters.

### 4. Main Execution

```python
def run():
    df = pd.read_csv('data.csv', names=['x', 'y'])
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    max_iterations = 1000
    print(f'starting gradient descent at b={initial_b}, m={initial_m}, error = {compute_error(initial_b, initial_m, df)}')
    [b, m] = gradient_descent(df, initial_b, initial_m, learning_rate, max_iterations)
    print(f'ending point at b={b}, m={m}, error = {compute_error(b, m, df)}')

if __name__ == '__main__':
    run()
```

- Reads `data.csv` and initializes **hyperparameters** (`learning_rate`, `initial_m`, `initial_b`, `max_iterations`).
- Runs **gradient descent** to optimize `m` and `b`.
- Prints the **initial** and **final** error values.

---

## How to Use

### 1. Install Dependencies

Ensure you have Python installed, then install the required libraries:

```bash
pip install pandas
```

### 2. Run the Script

```bash
python script.py
```

---

## Example Output

```
starting gradient descent at b=0, m=0, error = 5565.107834483211
ending point at b=0.08893651993741347, m=1.4777440851894448, error = 112.61481011613472
```

---

## Adjustments & Improvements

- **Modify Learning Rate**: Experiment with different values for `learning_rate`.
- **Increase Iterations**: Try a higher `max_iterations` for better convergence.
- **Use Different Data**: Replace `data.csv` with your own dataset.
- **Plot Results**: Use `matplotlib` to visualize the regression line.

---

## License

This project is open-source and free to use for educational purposes.

