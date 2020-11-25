import os

from data_loader import DataLoader

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.layers import InstanceNormalization


# MODEL = "../model_zebra_2_horse/image_generator_A_B_model.h5"
MODEL = "../model_zebra_2_horse/image_generator_B_A_model.h5"

# FILE_INPUT_PATH = "/Users/karin/programming/data/image_pairs/horse2zebra/val_A"
FILE_INPUT_PATH = "/Users/karin/programming/data/image_pairs/horse2zebra/val_B"

FILE_OUTPUT_PATH = "../test_predictions/model_predictions"

# SAVE_DIR = "A_2_B"
SAVE_DIR = "B_2_A"

IMAGE_SIZE = 128  # dimensions (squared) of incoming images /try 256, 512, ...
USE_IMG_RATIO = 1.0  # 0.6715  # square img: 1.0  # <--- the ratio of image dimensions (in == out)


if __name__ == "__main__":

    def resolve_single_image(file_name):
        input_image = data_loader.load_single_image(
            f"{FILE_INPUT_PATH}/{file_name}", IMAGE_SIZE
        )
    
        output_image = model.predict(input_image)  # predict image

        output_image = 0.5 * output_image + 0.5
        output_image = Image.fromarray((np.uint8(output_image * 255)[0]))

        if USE_IMG_RATIO < 1.0:
            output_image = output_image.resize(
                (IMAGE_SIZE, int(IMAGE_SIZE * USE_IMG_RATIO))
            )

        return output_image

    # load the trained model (generator)
    # Regarding compile=False: https://stackoverflow.com/a/57054106
    model = keras.models.load_model(MODEL, compile=False)

    data_loader = DataLoader(load_training_set=False)
    save_img_dir = f"{FILE_OUTPUT_PATH}/{SAVE_DIR}"

    file_names = os.listdir(FILE_INPUT_PATH)

    # ***
    # predict each image
    # ***
    for file_name in file_names:
        try:
            output_image = resolve_single_image(file_name)
            output_image.save(f"{save_img_dir}/{file_name}")
        except Exception as e:
            print(f"ERROR ---> {e}")
