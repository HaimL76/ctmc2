import copy
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

max_int: int = 2 ** 63 - 1


def run_simulation(bm0: int = 0, t_max: int = 99, delta_t: float = 0.01, num_simulations: int = 1):
    #cmap = mpl.colormaps['plasma']

    # Take colors at regular intervals spanning the colormap.
    #colors = cmap(np.linspace(0, 1, 256))
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(9999)
    limit = sys.getrecursionlimit()
    colors = ["red", "yellow", "orange", "green", "blue", "pink", "purple", "lightgreen", "lightblue"]

    arr: list[list[float]] = [[]] * num_simulations

    dict_boxes = {}

    box_row_min = max_int
    box_row_max = max_int * -1

    num_steps: int = int(t_max / delta_t)

    for k in range(num_simulations):
        xs: list[float] = [0] * num_steps
        bm: float = bm0

        now = datetime.now()
        ts = int(now.timestamp())
        np.random.seed(ts)

        for index in range(num_steps):
            arr_step = np.random.choice([1, -1], 1)

            arr_step = np.random.normal(0, delta_t, 1)

            step = arr_step[0]
            bm += step

            box_row_min = min(box_row_min, bm)
            box_row_max = max(box_row_max, bm)

            xs[index] = bm
            print(f"[{k}], index: {index}, bm: {bm}")

        epsilon = 0.9

        calculate_box_counting(xs, ((0, box_row_min), (num_steps, box_row_max)),
                               epsilon=epsilon, delta_t=delta_t, dict_boxes=dict_boxes, level=0)

        arr[k] = xs

    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    counter: int = 1

    xpoints = np.array([index * delta_t for index in range(num_steps)])

    for arr0 in arr:
        ypoints = np.array(arr0)

        ax.plot(xpoints, ypoints, label=f"simulation {counter}")
        counter += 1

    for level in dict_boxes.keys():
        list0 = dict_boxes[level]

        for box0 in list0:
            box = copy.deepcopy(box0[0])
            left_bottom: (int, int) = box[0]
            right_top: (int, int) = box[1]

            box_left: int = int(left_bottom[0])
            box_bottom: int = int(left_bottom[1])
            box_right: int = int(right_top[0])
            box_top: int = int(right_top[1])

            height = box_top - box_bottom
            width = box_right - box_left

            color = colors[level % len(colors)] if box0[1] else "none"

            ax.add_patch(Rectangle((box_left, box_bottom), width, height, facecolor=color,
                                   edgecolor='black', lw=0.7))

            #print(f"({box_left}, {box_bottom}), {width}, {height}")
        #matplotlib.patches.Rectangle(xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)[source]

    plt.legend(loc="upper left")
    plt.show()


def calculate_box_counting(xs: list[float], box: ((float, float), (float, float)), epsilon, delta_t,
                           dict_boxes: [int, list[((int, int), (int, int))]], level):
    print(f"level: {level}, box: {box}, epsilon: {epsilon}")

    coords_min = box[0]
    coords_max = box[1]

    row_min = coords_min[1]
    row_max = coords_max[1]

    col_min = coords_min[0]
    col_max = coords_max[0]

    epsilon_size_in_cols = epsilon / delta_t

    row_index = row_min

    while row_index <= row_max:
        row_start = row_index
        row_end = row_start + epsilon
        row_index = row_end

        col_index = col_min

        is_inside = False

        while col_index <= col_max:
            col_start = col_index
            col_end = col_start + epsilon_size_in_cols
            col_index = col_end

            box = ((col_start, row_start), (col_end, row_end))

            is_inside = False

            col = int(col_index)

            while not is_inside and col <= col_end:
                val = xs[col]
                col += 1

                if row_start <= val <= row_end:
                    is_inside = True

            if is_inside:
                if epsilon > 0.1:
                    epsilon_new = epsilon - 0.1
                    calculate_box_counting(xs, box, epsilon_new, delta_t,
                                           dict_boxes=dict_boxes, level=level + 1)
                else:
                    if level not in dict_boxes:
                        dict_boxes[level] = []

                    list0 = dict_boxes[level]

                    list0.append()


def main():
    run_simulation()


main()
