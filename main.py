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


def cut_image_array_on_frames(image_array, count_of_frames: tuple, size_of_frame: tuple):
    array_of_brightness = list()
    count_of_frame_by_width, count_of_frame_by_height = count_of_frames
    frame_width, frame_height = size_of_frame
    x_step = 0
    y_step = 0
    for frame_y in range(count_of_frame_by_width):
        for frame_x in range(count_of_frame_by_height):
            frame_brightness = list()
            for y in range(frame_height):
                for x in range(frame_width):
                    frame_brightness.append(image_array[x + x_step, y + y_step] / 255)
            array_of_brightness.append(frame_brightness)
            x_step += frame_width
        y_step += frame_height
        x_step = 0
    return array_of_brightness


class ImageConverter:

    def __init__(self, file: str):
        self.__image = ImageBuilder(
            file,
            (500, 500),  # 500 x 500
            (144, 144, 400, 400)  # 144; 144; 400; 400
        ).build()

    def __get_framed_image_size(self, frame_width: int, frame_height: int) -> tuple:
        size_x, size_y = self.__image.size
        size_frame_x = int(size_x / frame_width)
        size_frame_y = int(size_y / frame_height)
        return size_frame_x, size_frame_y

    def __cut_image_array_on_frames(self, count_of_frames: tuple, size_of_frame: tuple):
        return cut_image_array_on_frames(self.__image.load(), count_of_frames, size_of_frame)

    def get_brightness_array(self, width, height):
        size_frame_x, size_frame_y = self.__get_framed_image_size(width, height)
        return self.__cut_image_array_on_frames(
            (size_frame_x, size_frame_y),
            (width, height)
        )


class NeuralNetwork:

    pass


if __name__ == "__main__":
    converter = ImageConverter(
        "image.png",
    )
