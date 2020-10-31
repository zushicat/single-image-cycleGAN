from glob import glob
import os

import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm


BASE_PATH = "/Users/karin/programming/data/image_pairs/horse2zebra"
IMG_A_DIR = "trainA"
IMG_B_DIR = "trainB"


class DataLoader():
    def __init__(self, image_size=128, percent_of_training_set=0.5):
        self.datagen = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode="reflect"
        )

        self.image_size = image_size
        
        self.percent_of_training_set = percent_of_training_set

        self.images_path_A = glob(f'{BASE_PATH}/{IMG_A_DIR}/*.jpg') 
        self.images_path_B = glob(f'{BASE_PATH}/{IMG_B_DIR}/*.jpg') 

        self.images = []


    def load_images(self, images_path):
        num_images_from_training_set = int(len(images_path)*self.percent_of_training_set)
        print(f"Selected number of high res images from training set: {num_images_from_training_set} ({int(self.percent_of_training_set*100)}%)")
        
        border = 25  # trying to take augemtation (i.e. rotation) in load_data into account
        tmp_resize_value = self.image_size + (2*border)

        tmp_image_list = []

        random_img_path_selection = np.random.choice(images_path, size=num_images_from_training_set)

        for img_path in tqdm(random_img_path_selection, desc="Loading training set"): 
            img = load_img(img_path)  # type: PIL image in RGB
            tmp_image_list.append(img.resize((tmp_resize_value, tmp_resize_value), Image.BICUBIC))

        return tmp_image_list

    
    def crop_image(self, img):
        dist_to_image_border = (img.width-self.image_size)//2
        upper_left = dist_to_image_border
        lower_right = upper_left + self.image_size

        return img.crop((upper_left, upper_left, lower_right, lower_right))


    def load_data(self):
        # ***
        # initial image load
        # ***
        if len(self.images) == 0:
            self.images.append(self.load_images(self.images_path_A))
            self.images.append(self.load_images(self.images_path_B))

        
        # ***
        # augment and store the crops of this batch
        # ***
        for i, img_set in enumerate(self.images):
            for j, img in enumerate(img_set):
                # ***
                # image augmentation
                # ***
                data = np.expand_dims(img_to_array(img), 0)
                it = self.datagen.flow(data, batch_size=1)
                augmented_img_np_array = it.next()[0].astype('uint8')
                
                img = Image.fromarray(augmented_img_np_array)
                img = self.crop_image(img)
                # img.show()  # debug
                
                # ***
                # store augmented image
                # ***
                self.images[i][j] = np.asarray(img)

            # ***
            # normalize: [0, 255] -> [-1, 1]
            # ***
            self.images[i] = np.array(self.images[i]) / 127.5 - 1.
        
        return self.images


    def load_single_image(self, file_path, size=None):
        img = Image.open(file_path)
        if size is not None:
            img = img.resize((size, size))
        img_np_array = [np.asarray(img)]
        return np.array(img_np_array) / 127.5 - 1.



if __name__ == "__main__":
    data_loader = DataLoader(image_size=256, percent_of_training_set=0.01)
    imgs_set = data_loader.load_data()
    print('Loaded', imgs_set[0].shape, imgs_set[1].shape)

