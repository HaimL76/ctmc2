from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(x0: int = 0, t_range: int = 99999, num_simulations: int = 8):
    arr: list[list[int]] = [[]] * num_simulations

    for k in range(num_simulations):
        xs: list[int] = [0] * t_range
        x: int = x0

        now = datetime.now()
        ts = int(now.timestamp())
        np.random.seed(ts)

        for t in range(t_range):
            arr_step = np.random.choice([1, -1], 1)
            step = arr_step[0]
            x += step
            xs[t] = x
            print(f"[{k}], step[{t}]: {step}, x = {x}")

        arr[k] = xs

    plt.style.use('ggplot')

    counter: int = 1

    for arr0 in arr:
        xpoints = np.arange(0, t_range)
        ypoints = np.array(arr0)

        plt.plot(xpoints, ypoints, label=f"simulation {counter}")
        counter += 1

    plt.legend(loc="upper left")
    plt.show()


def main():
    run_simulation()

main()