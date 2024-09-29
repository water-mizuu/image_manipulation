# cspell: disable

from collections import deque
import contextlib
import glob
from PIL import Image
from typing import cast
import numpy as np


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class ImageDownscaling:
    def encode(self, image: Image.Image) -> Image.Image:
        file = image.convert("RGB")
        down_sampled = Image.new(image.mode, (image.width, image.height))

        for y in range(0, image.height, 2):
            print("Sampling row ", y // 2)
            for x in range(0, image.width, 2):
                r, g, b = 1, 1, 1

                for i in range(2):
                    for j in range(2):
                        r_, g_, b_ = cast(
                            tuple[int, int, int],
                            file.getpixel((x + i, y + j))
                        )
                        r += r_
                        g += g_
                        b += b_

                r = clamp(r // 4, 0, 255)
                g = clamp(g // 4, 0, 255)
                b = clamp(b // 4, 0, 255)

                down_sampled.putpixel((x, y), (r, g, b))
                down_sampled.putpixel((x + 1, y), (r, g, b))
                down_sampled.putpixel((x, y + 1), (r, g, b))
                down_sampled.putpixel((x + 1, y + 1), (r, g, b))

        return down_sampled


QT_THRESHOLD = 8
QT_DEPTH_LIMIT = 10


class QuadtreeOptimization:
    @staticmethod
    class QuadTree:
        start: tuple[int, int]
        end: tuple[int, int]
        depth: int

        children: list["QuadtreeOptimization.QuadTree"] | None
        value: tuple[int, int, int] | None

        def __init__(self, start, end, depth, value):
            self.start = start
            self.end = end
            self.depth = depth
            self.children = None
            self.value = value

    @staticmethod
    def encode(image: Image.Image, step_images=False) -> tuple[Image.Image, list[Image.Image] | None]:
        print("Encoding start!")

        output_step_images: list[Image.Image] | None = \
            [] if step_images else None
        image_array = np.array(image).astype(np.int16)

        # Grabs the average color of the entire image.
        image_average = image_array.mean(axis=(0, 1)).astype(np.uint8)
        root = QuadtreeOptimization.QuadTree(
            start=(0, 0),
            end=(image.width, image.height),
            depth=0,
            value=image_average,
        )

        tracking = -1
        stack = deque([root])
        while len(stack) > 0:
            current = stack.popleft()

            # If we want to output images at each depth change:
            if step_images and current.depth != tracking and output_step_images is not None:
                print("Depth change at", (tracking, current.depth, len(stack)))
                output_image = Image.new(image.mode, (image.width, image.height))
                output_stack = deque([root])

                while len(output_stack) > 0:
                    current_output = output_stack.pop()
                    if (children := current_output.children) is not None:
                        output_stack.extend(reversed(children))
                    elif (value := current_output.value) is not None:
                        for y in range(current_output.start[1], current_output.end[1]):
                            for x in range(current_output.start[0], current_output.end[0]):
                                output_image.putpixel((x, y), tuple(value))

                                if (y == current_output.end[1] - 1 or x == current_output.end[0] - 1):
                                    output_image.putpixel((x, y), (0, 0, 0))

                output_step_images.append(output_image)
                tracking = current.depth

            # Compute the MAD (Mean Average Deviation) of
            #   the current node and the pixels it covers.
            # MAD = Σ|pixel - average| / size

            assert current.value is not None
            area = image_array[current.start[1]:current.end[1], current.start[0]:current.end[0]]
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
                    (current.start, (mid_x, mid_y)),
                    ((mid_x, current.start[1]), (current.end[0], mid_y)),
                    ((current.start[0], mid_y), (mid_x, current.end[1])),
                    ((mid_x,            mid_y), current.end)
                ]

                for (x_start, y_start), (x_end, y_end) in children_bounds:
                    if y_end - y_start > 0 and x_end - x_start > 0:
                        value = image_array[y_start:y_end, x_start:x_end]\
                            .mean(axis=(0, 1))\
                            .astype(np.uint8)

                        # We create a new QuadTree node for each quadrant.
                        qt = QuadtreeOptimization.QuadTree(
                            start=(x_start, y_start),
                            end=(x_end, y_end),
                            depth=current.depth + 1,
                            value=value,
                        )
                        current.children.append(qt)

                # We add all of the nodes to the stack to be processed.
                stack.extend(current.children)

        print("QuadTree building finished")

        # Make an output image to draw the QuadTree on.
        output_image = Image.new(image.mode, (image.width, image.height))

        # Depth-first search to draw the QuadTree on the output image.
        output_stack = deque([root])
        while len(output_stack) > 0:
            current = output_stack.pop()

            if (children := current.children) is not None:
                output_stack.extend(reversed(children))
            elif (value := current.value) is not None:
                for y in range(current.start[1], current.end[1]):
                    for x in range(current.start[0], current.end[0]):
                        output_image.putpixel((x, y), tuple(value))

        print("Output finished")
        output_image.save("output.png")

        return (output_image, output_step_images)


if __name__ == "__main__":
    # use exit stack to automatically close opened images
    # with contextlib.ExitStack() as stack:

    #     # lazily load images
    #     imgs = (stack.enter_context(Image.open(f))
    #             for f in sorted(glob.glob("ds_example/*.png")))

    #     # extract  first image from iterator
    #     img = next(imgs)

    #     # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    #     img.save(fp="ds_example/output.gif", format='GIF', append_images=imgs,
    #              save_all=True, duration=1000, loop=0)

    with Image.open("png_1.png") as image:
        down_sampled, step_images = QuadtreeOptimization().encode(image, step_images=True)
        down_sampled.save("output.png")

        if step_images is not None:
            for i, step_image in enumerate(step_images):
                step_image.save(f"output/output_{i:2}.png")

    # up_sampled = ImageDownscaling().decode(down_sampled)
    # up_sampled.save("png_5.png")

    # with Image.open("png_4.png") as image:
    #     image = image.convert("RGB")
    #     compressed = QuadtreeOptimization.encode(image)
