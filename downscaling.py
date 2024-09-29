from PIL import Image
from os import stat


BLOCK_SIZE = 2


def encode(image: Image.Image) -> Image.Image:
    """
    Produces a redundancy-filled down-sampled image from the input.
    The added redundancy allows for other lossless compression
        algorithms to be more effective.
    """
    image = image.convert("RGB")

    # The output image is initialized with
    #   the same size as the input image
    down_sampled = Image.new(image.mode, (image.width, image.height))

    print("Image size: ", image.size)

    # We iterate [y, x] for every pixel in
    #   the image by the [BLOCK_SIZE].
    # The current implementation states that
    #   [BLOCK_SIZE] is 2, so we iterate by 2.
    for y in range(0, image.height, BLOCK_SIZE):
        for x in range(0, image.width, BLOCK_SIZE):
            r, g, b = 0, 0, 0

            # The [domain] refers to the pixels that
            #   are within the bounds of the current block.
            # Since the block size is 2, the domain is:
            #  (0, 0), (0, 1), (1, 0), (1, 1).
            domain = [(i, j) for i in range(BLOCK_SIZE)
                      for j in range(BLOCK_SIZE)
                      if x + i < image.width and y + j < image.height]

            # For every block in the domain, we add the RGB values.
            for i, j in domain:
                r_, g_, b_ = image.getpixel((x + i, y + j))
                r += r_
                g += g_
                b += b_

            # And divide the sum by the number of pixels in the block.
            #   This is the arithmetic mean of the block.
            r = max(0, min(255, r // (BLOCK_SIZE * BLOCK_SIZE)))
            g = max(0, min(255, g // (BLOCK_SIZE * BLOCK_SIZE)))
            b = max(0, min(255, b // (BLOCK_SIZE * BLOCK_SIZE)))

            # We then set the RGB values of the block in the output to the mean.
            for i, j in domain:
                down_sampled.putpixel((x + i, y + j), (r, g, b))

    return down_sampled


# Runner Code
if __name__ == "__main__":
    files = [
        ("png_1.png", "ds_output_1/output.png"),
        ("png_3.png", "ds_output_3/output.png"),
        ("png_4.png", "ds_output_4/output.png"),
    ]

    for input_path, output_path in files:
        print(f"Start {input_path}")
        with Image.open(input_path).convert("RGB") as image:
            down_sampled = encode(image)
            down_sampled.save(output_path)

            # Compare the disk-size of the original image and the down-sampled image.
            input_size = stat(input_path).st_size
            output_size = stat(output_path).st_size

            print(f"Original size: {input_size} bytes")
            print(f"Down-sampled size: {output_size} bytes")

        print(f"End {input_path}")
        print()
