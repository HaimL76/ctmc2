from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(x0: int = 0, t_range: int = 999, num_simulations: int = 8):
    arr: list[list[int]] = [[]] * num_simulations

    box_row_min = 999999
    box_row_max = -999999

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

            if x < box_row_min:
                box_row_min = x

            if x > box_row_max:
                box_row_max = x

            xs[t] = x
            print(f"[{k}], step[{t}]: {step}, x = {x}")

        dict_boxes = {}

        calculate_box_counting(xs, [((0, box_row_min), (t_range, box_row_max))], dict_boxes=dict_boxes, level=0)

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


def calculate_box_counting(xs: list[int], list_boxes: list[((int, int), (int, int))], dict_boxes: [int, list[((int, int), (int, int))]], level):
    list_boxes_new: list[((int, int), (int, int))] = []

    max_int: int = 2 ** 63 - 1

    for box in list_boxes:
        left_bottom: (int, int) = box[0]
        right_top: (int, int) = box[1]

        box_left: int = int(left_bottom[0])
        box_bottom: int = int(left_bottom[1])
        box_right: int = int(right_top[0])
        box_top: int = int(right_top[1])

        line_row_max: int = -999999# max_int * -1
        line_row_min: int = 999999# max_int

        for col in range(box_left, box_right):
            x = int(xs[col])

            if x < line_row_min:
                line_row_min = x

            if x > line_row_max:
                line_row_max = x

        outside: bool = line_row_max < box_bottom or line_row_min > box_top

        if not outside:
            if level not in dict_boxes:
                dict_boxes[level] = []

            list0: list[(int, int, int, int)] = dict_boxes[level]

            list0.append(box)

            if box_top - box_bottom < 2 or box_right - box_left < 2:
                return

            row_start = box_bottom
            col_start = box_left

            row_middle = (box_bottom + box_top) / 2
            col_middle = (box_left + box_right) / 2

            row_end = box_top
            col_end = box_right

            list_boxes_new.append(((col_start, row_start), (col_middle, row_middle)))
            list_boxes_new.append(((col_middle, row_start), (col_end, row_middle)))
            list_boxes_new.append(((col_start, row_middle), (col_middle, row_end)))
            list_boxes_new.append(((col_middle, row_middle), (col_end, row_end)))

            calculate_box_counting(xs, list_boxes_new, dict_boxes=dict_boxes, level=level + 1)




def main():
    run_simulation()

main()