
from collections import deque
from os import stat
from PIL import Image
import numpy as np


QT_THRESHOLD = 16
QT_DEPTH_LIMIT = 12


class QuadTree:
    # [start] represents the top-left corner of the rectangle.
    start: tuple[int, int]

    # [end] represents the bottom-right corner of the rectangle.
    end: tuple[int, int]

    # The depth of the current node in the QuadTree.
    depth: int

    # The children of the current node.
    #   If the node is a leaf, then this is None.
    children: list["QuadTree"] | None

    # The average color of the current node.
    #   If the node is not a leaf, then this is None.
    value: tuple[int, int, int] | None

    def __init__(self, start, end, depth, value):
        self.start = start
        self.end = end
        self.depth = depth
        self.value = value
        self.children = None


def encode(image: Image.Image) -> Image.Image:
    print("Encoding start")

    # Get the image as an array of signed 16-bit integers.
    #   The reason it is 16-bit signed instead of 8-bit unsigned
    #   is to prevent overflow when calculating the MAD.
    image_array = np.array(image).astype(np.int16)

    # Grabs the average color of the entire image.
    image_average = image_array.mean(axis=(0, 1)).astype(np.uint8)
    root = QuadTree(
        start=(0, 0),
        end=(image.width, image.height),
        depth=0,
        value=image_average,
    )

    queue = deque([root])
    while len(queue) > 0:
        current = queue.popleft()

        # Compute the MAD (Mean Average Deviation) of
        #   the current node and the pixels it covers.
        # MAD = Σ|pixel - average| / size

        assert current.value is not None
        area = image_array[current.start[1]:current.end[1],
                           current.start[0]:current.end[0]]
        sum_of_difference = np.abs(area - current.value).sum()

        size = (current.end[1] - current.start[1]) \
            * (current.end[0] - current.start[0])
        mean_average_deviation = sum_of_difference / size

        # If the MAD is greater than the set threshold,
        #   and the depth is less than the limit (i.e. we can still split),
        #   Then we do.
        if mean_average_deviation > QT_THRESHOLD \
                and current.depth + 1 < QT_DEPTH_LIMIT:
            # Find the midpoints of the current boundaries.
            mid_x = (current.end[0] + current.start[0]) // 2
            mid_y = (current.end[1] + current.start[1]) // 2

            current.value = None
            current.children = []

            # The bounds are defined as follows:
            #   (start, end), forming a rectangle.
            children_bounds = [
                (current.start,             (mid_x, mid_y)),
                ((mid_x, current.start[1]), (current.end[0], mid_y)),
                ((current.start[0], mid_y), (mid_x, current.end[1])),
                ((mid_x,            mid_y), current.end)
            ]

            for (x_start, y_start), (x_end, y_end) in children_bounds:
                if y_end - y_start > 0 and x_end - x_start > 0:
                    child_value = image_array[y_start:y_end, x_start:x_end]\
                        .mean(axis=(0, 1))\
                        .astype(np.uint8)

                    # We create a new QuadTree node for each quadrant.
                    qt_subnode = QuadTree(
                        start=(x_start, y_start),
                        end=(x_end, y_end),
                        depth=current.depth + 1,
                        value=child_value,
                    )
                    current.children.append(qt_subnode)

            # We add all of the nodes to the stack to be processed.
            queue.extend(current.children)

    print("QuadTree building finished")

    # Make an output image to draw the QuadTree on.
    output_image = Image.new(image.mode, (image.width, image.height))

    # Depth-first algorithm to draw the QuadTree on the output image.
    output_stack = deque([root])
    while len(output_stack) > 0:
        current = output_stack.pop()

        if (children := current.children) is not None:
            output_stack.extend(reversed(children))
        elif (child_value := current.value) is not None:
            for y in range(current.start[1], current.end[1]):
                for x in range(current.start[0], current.end[0]):
                    output_image.putpixel((x, y), tuple(child_value))

    print("Image painting finished")

    return output_image


# Runner Code
if __name__ == "__main__":
    files = [
        ("png_1.png", "qt_output_1/output.png"),
        ("png_3.png", "qt_output_3/output.png"),
        ("png_4.png", "qt_output_4/output.png"),
    ]

    for input_path, output_path in files:
        print(f"Start {input_path}")
        with Image.open(input_path).convert("RGB") as image:
            down_sampled = encode(image)
            down_sampled.save(output_path)

            # Compare the disk-size of the original image
            #   and the down-sampled image.
            input_size = stat(input_path).st_size
            output_size = stat(output_path).st_size

            print(f"Original size: {input_size} bytes")
            print(f"Down-sampled size: {output_size} bytes")

        print(f"End {input_path}")
        print()
