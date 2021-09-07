from PIL import Image


class ImageBuilder:

    def __init__(self, file: str, size: tuple, crop_size: tuple):
        self.__file = Image.open(file)
        self.__size = size
        self.__crop_size = crop_size

    def build(self):
        gray_image = self.__to_gray_scale()
        resized_image = self.__resize(gray_image)
        cropped_image = self.__crop(resized_image)
        return cropped_image

    def __to_gray_scale(self):
        return self.__file.convert("L")

    def __crop(self, image: Image):
        left_x, left_y, right_x, right_y = self.__crop_size
        return image.crop((left_x, left_y, right_x, right_y))

    def __resize(self, image: Image):
        old_height, old_width = image.size
        width, height = self.__size
        if width and height:
            max_size = (width, height)
        elif width:
            max_size = (width, old_height)
        elif height:
            max_size = (old_width, height)
        else:
            raise RuntimeError()
        image.thumbnail(max_size, Image.ANTIALIAS)
        return image


class NeuralNetwork:

    def __init__(self, file: str):
        self.__image_builder = ImageBuilder(
            file,
            (500, 500),  # 500 x 500
            (144, 144, 400, 400)  # 144; 144; 400; 400
        )

    def get_brightness_array(self, width, height):
        image = self.__image_builder.build()
        size_x, size_y = image.size
        size_frame_x = int(size_x / width)
        size_frame_y = int(size_y / height)
        pixels = image.load()
        array_of_brightness = list()
        x_step = 0
        y_step = 0
        for frame_index_y in range(size_frame_y):
            frame_brightness_array = []
            y_step += height
            for frame_index in range(size_frame_x):
                x_step += width
                for y in range(height):
                    for x in range(width):
                        frame_brightness_array.append(pixels[x + x_step, y + y_step] / 255)
                array_of_brightness.append(frame_brightness_array)
        return array_of_brightness


if __name__ == "__main__":
    a = NeuralNetwork("test2.png")
    print(a.get_brightness_array(4, 4))
