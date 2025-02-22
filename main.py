import pandas as pd


def compute_error(b, m, df):
    # Initialize it at 0
    totalError = 0
    for i in range(0, len(df)):
        # Get the difference, square it, add it to the total
        totalError += (df.y[i] - (m * df.x[i] + b)) ** 2
    # Get the average
    # 1/N sum(y_i - (m*x_i + b))^2
    return totalError / float(len(df))


def gradient_descent(df, initial_b, initial_m, learning_rate, max_iterations):
    b = initial_b
    m = initial_m

    # Gradient descent
    for i in range(max_iterations):
        # Update b and m with the new b and m by performing GD
        b, m = step_gradient(b, m, df, learning_rate)

    return [b, m]


def step_gradient(b, m, df, learning_rate):
    b_grad = 0
    m_grad = 0
    N = len(df)
    for i in range(0, N):
        x = df.x[i]
        y = df.y[i]
        # Get the direction
        # Partial derivative b
        b_grad += -(2 / N) * (y - ((m * x) + b))
        # Partial derivative m
        m_grad += -(2 / N) * x * (y - ((m * x) + b))
    # Update b and m
    new_b = b - (learning_rate * b_grad)
    new_m = m - (learning_rate * m_grad)
    return [new_b, new_m]


def run():
    # 1.Collect our data
    df = pd.read_csv('data.csv', names=['x', 'y'])

    # 2.Hyperparameters
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    max_iterations = 1000
    compute_error(0, 0, df)
    # 3.Train our model
    print(
        f'starting gradient descent at b={initial_b}, m={initial_m}, error = {compute_error(initial_b, initial_m, df)}')
    [b, m] = gradient_descent(df, initial_b, initial_m, learning_rate, max_iterations)
    print(f'ending point at b={b}, m={m}, error = {compute_error(b, m, df)}')


if __name__ == '__main__':
    run()
