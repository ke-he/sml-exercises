import numpy as np
import time


def slow_code(x):
    # check if x is a square 2d-matrix, otherwise return None
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return None

    n = x.shape[0]

    # calc average
    total = 0
    for i in range(n):
        for j in range(n):
            total += x[i, j]
    avg = total / (n * n)

    # add diagonal elements if value >= avg
    total = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                if x[i, j] >= avg:
                    total += x[i, j]
    return total


def fast_code(x):
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        return None

    avg = np.mean(x)

    diagonal_elements = np.diagonal(x)

    return np.sum(diagonal_elements[diagonal_elements >= avg])


def main():
    np.random.seed(1)
    n = 200
    x_test = np.random.rand(n, n)

    start_time = time.time()
    solution1 = slow_code(x_test)
    end_time = time.time()
    slow_time = end_time - start_time

    start_time = time.time()
    solution2 = fast_code(x_test)
    end_time = time.time()
    fast_time = end_time - start_time

    print(f"Execution time of slow_code: {slow_time:.10f} seconds")
    print(f"Result of slow_code: {solution1:.10f}")
    print()
    print(f"Execution time of fast_code: {fast_time:.10f} seconds")
    print(f"Result of fast_code: {solution2:.10f}")
    print()

    if np.isclose(solution1, solution2, atol=1e-10):
        print("Your solution seems correct")
    else:
        print("Results are too different")


if __name__ == "__main__":
    main()
