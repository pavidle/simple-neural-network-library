from main import cut_image_array_on_frames, ImageBuilder


def test_cut_image_on_frame():
    builder = ImageBuilder(
        "../image.png",
        (4, 4),
        (1, 1, 5, 5)
    )
    image = builder.build()
    pixels = image.load()
    cut_image = cut_image_array_on_frames(pixels, (2, 2), (2, 2))
    result = [
        [81 / 255, 118 / 255, 137 / 255, 188 / 255],
        [42 / 255, 0 / 255, 52 / 255, 0 / 255],
        [108 / 255, 120 / 255, 0 / 255, 0 / 255],
        [75 / 255, 0 / 255, 0 / 255, 0 / 255]
    ]
    assert cut_image == result
