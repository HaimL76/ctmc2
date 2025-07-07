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


def calculate_box_counting(xs: list[float], area: ((float, float), (float, float)), epsilon, delta_t,
                           dict_boxes: [int, list[((int, int), (int, int))]], level):
    print(f"level: {level}")

    coords_min = area[0]
    coords_max = area[1]

    row_min = coords_min[1]
    row_max = coords_max[1]

    row_index = row_min

    while row_index <= row_max:
        print(row_index)
        row_index += epsilon

    for box in grid:
        box0 = copy.deepcopy(box)

        left_bottom: (int, int) = box0[0]
        right_top: (int, int) = box0[1]

        box_left: int = int(left_bottom[0])
        box_bottom: int = int(left_bottom[1])
        box_right: int = int(right_top[0])
        box_top: int = int(right_top[1])

        #print(f"{[level]}, ({box_left}, {box_bottom}, {box_right}, {box_top})")

        col = box_left

        inside_box: bool = False

        while not inside_box and col < box_right:
            val = xs[col]
            col += 1

            if box_bottom <= val <= box_top:
                inside_box = True
            else:
                _ = 0  #print(f"{box_bottom}, {val}, {box_top}")

        if not inside_box:
            _ = 0

        if inside_box:
            height = box_top - box_bottom
            width = box_right - box_left

            if height > 1 or width > 1:
                row_start = box_bottom
                col_start = box_left

                row_middle = (box_bottom + box_top) / 2
                col_middle = (box_left + box_right) / 2

                row_end = box_top
                col_end = box_right

                list_boxes_new: list[((int, int), (int, int))] = []

                if height > 1 and width > 1:
                    list_boxes_new.append(((col_start, row_start), (col_middle, row_middle)))
                    list_boxes_new.append(((col_middle, row_start), (col_end, row_middle)))
                    list_boxes_new.append(((col_start, row_middle), (col_middle, row_end)))
                    list_boxes_new.append(((col_middle, row_middle), (col_end, row_end)))
                elif height > 1:
                    list_boxes_new.append(((col_start, row_start), (col_end, row_middle)))
                    list_boxes_new.append(((col_start, row_middle), (col_end, row_end)))
                elif width > 1:
                    list_boxes_new.append(((col_start, row_start), (col_middle, row_end)))
                    list_boxes_new.append(((col_middle, row_start), (col_end, row_end)))

                calculate_box_counting(xs, area=list_boxes_new, epsilon=epsilon, delta_t=delta_t,
                                       dict_boxes=dict_boxes, level=level + 1)
            else:
                if level not in dict_boxes:
                    dict_boxes[level] = []

                list0: list[(((int, int), (int, int)), bool)] = dict_boxes[level]

                list0.append((box0, inside_box))


def main():
    run_simulation()


main()
