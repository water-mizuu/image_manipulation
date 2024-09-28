# cspell: disable

from PIL import Image, ImageFile
from typing import cast
import numpy as np


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class ImageDownsampling:
    def encode(self, image: Image.Image) -> Image.Image:
        file = image.convert("RGB")
        down_sampled = Image.new(
            image.mode, (image.width // 2, image.height // 2))

        for x in range(0, image.width, 2):
            for y in range(0, image.height, 2):
                r, g, b = 1, 1, 1
                for i in range(2):
                    for j in range(2):
                        r_, g_, b_ = cast(
                            tuple[int, int, int], file.getpixel((x + i, y + j)))
                        r += r_
                        g += g_
                        b += b_

                r = clamp(r // 4, 0, 255)
                g = clamp(g // 4, 0, 255)
                b = clamp(b // 4, 0, 255)

                down_sampled.putpixel((x // 2, y // 2), (r, g, b))

        return down_sampled

    def decode(self, image: Image.Image) -> Image.Image:
        file = image.convert("RGB")
        up_sampled = Image.new(image.mode, (image.width * 2, image.height * 2))

        for x in range(image.width):
            for y in range(image.height):
                r, g, b = cast(tuple[int, int, int], file.getpixel((x, y)))
                for i in range(2):
                    for j in range(2):
                        up_sampled.putpixel((x * 2 + i, y * 2 + j), (r, g, b))

        return up_sampled


QT_THRESHOLD = 12
QT_DEPTH_LIMIT = 24


class QuadtreeOptimization:
    @staticmethod
    class QuadTree:
        start: tuple[int, int]
        end: tuple[int, int]
        depth: int

        children: list["QuadtreeOptimization.QuadTree"]
        value: tuple[int, int, int] | None

        def __init__(self, start, end, depth, value):
            self.start = start
            self.end = end
            self.depth = depth
            self.value = value

    @staticmethod
    def encode(image: Image.Image) -> Image.Image:
        print("Encoding start!")

        image_array = np.array(image)
        image_average = image_array.mean(axis=(0, 1)).astype(np.uint8)

        root = QuadtreeOptimization.QuadTree(
            start=(0, 0),
            end=(image.width, image.height),
            depth=0,
            value=image_average)
        stack = [root]

        while len(stack) > 0:
            current = stack.pop(0)

            # Compute the MAD (Mean Average Deviation) of the current node and the pixels it covers.
            current_mad = 0
            for x in range(current.start[0], current.end[0]):
                for y in range(current.start[1], current.end[1]):
                    pixel = np.array(image.getpixel((x, y)))
                    current_mad += np.abs(pixel - current.value).sum()

            size = (current.end[1] - current.start[1]) \
                * (current.end[0] - current.start[0])
            mean_average_deviation = current_mad / size

            if mean_average_deviation > QT_THRESHOLD and current.depth + 1 < QT_DEPTH_LIMIT:
                mid_x = (current.end[0] + current.start[0]) // 2
                mid_y = (current.end[1] + current.start[1]) // 2

                bounds = [
                    (current.start, (mid_x, mid_y)),
                    ((mid_x, current.start[1]), (current.end[0], mid_y)),
                    ((current.start[0], mid_y), (mid_x, current.end[1])),
                    ((mid_x,             mid_y), current.end)
                ]
                current.value = None
                current.children = []

                for (x_start, y_start), (x_end, y_end) in bounds:
                    if y_end - y_start > 0 and x_end - x_start > 0:
                        value = image_array[y_start:y_end, x_start:x_end]\
                            .mean(axis=(0, 1))\
                            .astype(np.uint8)

                        qt = QuadtreeOptimization.QuadTree(
                            start=(x_start, y_start),
                            end=(x_end, y_end),
                            depth=current.depth + 1,
                            value=value)
                        current.children.append(qt)

                stack.extend(current.children)

        print("QuadTree finished")
        output_image = Image.new(image.mode, (image.width, image.height))
        output_stack = [root]

        while len(output_stack) > 0:
            current = output_stack.pop()
            if current.value is not None:
                for y in range(current.start[1], current.end[1]):
                    for x in range(current.start[0], current.end[0]):
                        try:
                            output_image.putpixel((x, y), tuple(current.value))
                        except IndexError:
                            print("ERROR!", (x, y), (image.width, image.height),
                                  current.value)
            else:
                output_stack.extend(reversed(current.children))

        output_image.save("output.png")
        print("Output finished")

        return output_image

    @staticmethod
    def decode(image: Image.Image) -> Image.Image:
        return image


def main():
    with Image.open("png_3.png") as image:
        down_sampled = ImageDownsampling().encode(image)
        down_sampled.save("down_sampled.png")

        up_sampled = ImageDownsampling().decode(down_sampled)
        up_sampled.save("up_sampled.png")

    with Image.open("png_2.png") as image:
        image = image.convert("RGB")
        QuadtreeOptimization.encode(image)


if __name__ == "__main__":
    main()
