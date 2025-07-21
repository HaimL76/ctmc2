import copy
import math
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

max_int: int = 2 ** 63 - 1


def run_simulation(bm0: int = 0, t_max: int = 1, initial_epsilon: float = 0.9,
                   downscale_factor: float = 0.5, delta_t: float = 0.00005, num_simulations: int = 1):
    colors = ["red", "yellow", "orange", "green", "blue", "pink", "purple", "lightgreen", "lightblue"]

    arr: list[list[float], dict] = [[]] * num_simulations

    box_row_min = max_int
    box_row_max = max_int * -1

    num_steps: int = int(t_max / delta_t)

    current_timestamp = 0

    for k in range(num_simulations):
        bms: list[float] = [0] * num_steps
        bm: float = bm0

        now = datetime.now()
        ts = int(now.timestamp())

        if ts != current_timestamp:
            np.random.seed(ts)
            current_timestamp = ts

        for index in range(num_steps):
            arr_step = np.random.normal(0, delta_t, 1)

            step = arr_step[0]
            bm += step

            box_row_min = min(box_row_min, bm)
            box_row_max = max(box_row_max, bm)

            bms[index] = bm
            print(f"[{k}], index: {index}, bm: {bm}")

        epsilon = initial_epsilon

        dict_boxes: dict = {}

        count = calculate_box_counting(bms, ((0, box_row_min), (num_steps, box_row_max)),
                                       epsilon=epsilon, downscale_factor=downscale_factor,
                                       delta_t=delta_t, dict_boxes=dict_boxes, level=0, count=0,
                                       simulation_index=k)

        arr[k] = (bms, dict_boxes)

    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    counter: int = 1

    xpoints = np.array([index * delta_t for index in range(num_steps)])

    for tup_bms in arr:
        bms = tup_bms[0]

        if isinstance(bms, list):
            ypoints = np.array(bms)

            if len(ypoints) == len(xpoints):
                ax.plot(xpoints, ypoints, label=f"simulation {counter}")
        counter += 1

    plt.legend(loc="upper left")
    plt.title("Box Counting Dimension")
    plt.xlabel("One Over Epsilon")
    plt.ylabel("Box Counting Dimension")
    plt.savefig('brownian_motion_simulation.png', bbox_inches='tight')
    plt.close(fig=fig)

    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    counter: int = 1

    for tup_bms in arr:
        draw_box_counting = False

        if isinstance(dict_boxes, dict):
            xarr: list[float] = []
            yarr: list[float] = []

            for level in dict_boxes.keys():
                tup = dict_boxes[level]

                num_boxes = tup[1]
                epsilon = tup[2]

                log_num_boxes = math.log(num_boxes)
                log_1_over_epsilon = math.log(1 / epsilon)

                dimension = log_num_boxes / log_1_over_epsilon

                print(f"{epsilon},{num_boxes},{dimension}")

                xarr.append(log_1_over_epsilon)
                yarr.append(dimension)

                list0 = tup[0]

                if list0 is not None and len(list0) > 0:
                    for tup in list0:
                        box0 = tup[1]
                        box = copy.deepcopy(box0)

                        num_boxes += 1

                        #print(f"num boxes: {num_boxes}")

                        if draw_box_counting:
                            coords_min: (float, float) = box[0]
                            coords_max: (float, float) = box[1]

                            col_min: float = coords_min[0]
                            col_max: float = coords_max[0]
                            row_min: float = coords_min[1]
                            row_max: float = coords_max[1]

                            t_min = col_min * delta_t
                            t_max = col_max * delta_t

                            height = row_max - row_min
                            width = t_max - t_min

                            color = colors[level % len(colors)] if box0[1] else "none"

                            ax.add_patch(Rectangle((t_min, row_min), width, height, facecolor=color,
                                           edgecolor='black', lw=0.7))

            xpoints_boxes = np.array(xarr)
            ypoints_boxes = np.array(yarr)

            ax.plot(xpoints_boxes, ypoints_boxes, label=f"box counting {counter}")

        counter += 1

    plt.legend(loc="upper left")
    plt.title("Box Counting Dimension")
    plt.xlabel("One Over Epsilon")
    plt.ylabel("Box Counting Dimension")
    plt.savefig('box_counting_dimensions.png', bbox_inches='tight')


def calculate_box_counting(xs: list[float], box: ((float, float), (float, float)), epsilon, downscale_factor,
                           delta_t, dict_boxes: [int, list[((int, int), (int, int))]], level, count,
                           simulation_index):
    if (count % 1000) == 0:
        print(f"[{simulation_index}] count: {count}, level: {level}, epsilon: {epsilon}")

    coords_min = box[0]
    coords_max = box[1]

    row_min = coords_min[1]
    row_max = coords_max[1]

    col_min = coords_min[0]
    col_max = coords_max[0]

    epsilon_size_in_cols = epsilon / delta_t

    row_index = row_min

    draw = False

    while row_index <= row_max:
        row_start = row_index
        row_end = row_start + epsilon
        row_index = row_end

        col_index = col_min

        while col_index <= col_max:
            col_start = col_index
            col_end = col_start + epsilon_size_in_cols
            col_index = col_end

            box = ((col_start, row_start), (col_end, row_end))

            is_inside = False

            col = int(col_index)

            while not is_inside and col <= col_end:
                if col < len(xs):
                    val = xs[col]

                    if row_start <= val <= row_end:
                        is_inside = True

                col += 1

            if is_inside:
                if level not in dict_boxes:
                    dict_boxes[level] = ([], 0, epsilon)

                tup = dict_boxes[level]

                dict_boxes[level] = (tup[0], tup[1] + 1, epsilon)

                if epsilon > delta_t:
                    epsilon_new = epsilon * downscale_factor
                    count = calculate_box_counting(xs, box, epsilon_new, downscale_factor=downscale_factor,
                                                   delta_t=delta_t, dict_boxes=dict_boxes, level=level + 1,
                                                   count=count, simulation_index=simulation_index)
                else:
                    count += 1

                    if draw:
                        tup = dict_boxes[level]

                        list0 = tup[0]

                        if list0 is None:
                            list0 = []

                            dict_boxes[level] = (list0, tup[1], epsilon)

                        list0.append((epsilon, copy.deepcopy(box)))

    return count


def main():
    run_simulation(num_simulations=22)


main()
