from glob import glob
import os

import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm


BASE_PATH = "/Users/karin/programming/data/image_pairs/tree2no_tree"
IMG_A_DIR = "train_A"
IMG_B_DIR = "train_B/*"


class DataLoader():
    def __init__(self, image_size=128, percent_of_training_set=0.5, load_training_set=True, use_augmentation=True):
        if load_training_set is True:
            if use_augmentation is True:
                self.datagen = ImageDataGenerator(
                    rotation_range=90,
                    horizontal_flip=True,
                    vertical_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    fill_mode="reflect"
                )
            else:
                self.datagen = ImageDataGenerator()

            self.image_size = image_size
            
            self.percent_of_training_set = percent_of_training_set

            self.images_path_A = glob(f'{BASE_PATH}/{IMG_A_DIR}/*.jpg') 
            self.images_path_B = glob(f'{BASE_PATH}/{IMG_B_DIR}/*.jpg') 

            # ***
            # get number of images from % of smallest set of both sets
            # ***
            self.num_images_from_training_set = self.set_image_number_to_load()
            
            # ***
            # get image sets
            # ***
            self.images = []  # List[List, List]
            self.images.append(self.load_images(self.images_path_A))
            self.images.append(self.load_images(self.images_path_B))
        

    def set_image_number_to_load(self):
        '''
        Get number of images from % of smaller of both sets
        '''
        min_img_set_len = len(self.images_path_A) if len(self.images_path_A) < len(self.images_path_B) else len(self.images_path_B)
        num_images = int(min_img_set_len * self.percent_of_training_set)

        print(f"Selected number of high res images from training set: {num_images} ({int(self.percent_of_training_set*100)}%)")

        return num_images


    def get_train_set_size(self):
        return self.num_images_from_training_set


    def load_images(self, images_path):
        border = 25  # trying to take augemtation (i.e. rotation) in load_data into account
        tmp_resize_value = self.image_size + (2*border)

        tmp_image_list = []

        random_img_path_selection = np.random.choice(images_path, size=self.num_images_from_training_set)
        
        for img_path in tqdm(random_img_path_selection, desc="Loading training set"): 
            img = load_img(img_path)  # type: PIL image in RGB
            tmp_image_list.append(img.resize((tmp_resize_value, tmp_resize_value), Image.BICUBIC))

        return tmp_image_list

    
    def crop_image(self, img):
        dist_to_image_border = (img.width-self.image_size)//2
        upper_left = dist_to_image_border
        lower_right = upper_left + self.image_size

        return img.crop((upper_left, upper_left, lower_right, lower_right))


    def load_data(self, batch_size=1):
        batch = []
        for i in range(2):
            batch.append([])
        # ***
        # assign random batches
        # ***
        random_indices = np.random.choice(self.num_images_from_training_set, size=batch_size)
        for i in random_indices:
            batch[0].append(self.images[0][i])
            batch[1].append(self.images[1][i])

        
        # ***
        # augment and return batch
        # ***
        for i, img_set in enumerate(batch):
            for j, img in enumerate(img_set):
                # ***
                # image augmentation
                # ***
                data = np.expand_dims(img_to_array(img), 0)
                it = self.datagen.flow(data, batch_size=1)
                augmented_img_np_array = it.next()[0].astype('uint8')
                
                img = Image.fromarray(augmented_img_np_array)
                img = self.crop_image(img)
                img.show()  # debug

                # ***
                # store augmented image
                # ***
                batch[i][j] = np.asarray(img)
            
            
            # ***
            # normalize: [0, 255] -> [-1, 1]
            # ***
            batch[i] = np.array(batch[i]) / 127.5 - 1.
        
        return batch


    def load_single_image(self, file_path, size=None):
        img = Image.open(file_path)
        if size is not None:
            img = img.resize((size, size))
        img_np_array = [np.asarray(img)]
        return np.array(img_np_array) / 127.5 - 1.



if __name__ == "__main__":
    data_loader = DataLoader(image_size=128, percent_of_training_set=0.05, use_augmentation=False)
    # print(len(data_loader.images[0]))
    # print(len(data_loader.images[1]))
    batch = data_loader.load_data(batch_size=1)
    print('Loaded', batch[0].shape, batch[1].shape)

