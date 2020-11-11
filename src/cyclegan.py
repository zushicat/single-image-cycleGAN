
'''
See also: https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
'''
import datetime
import os
import pickle

from data_loader import DataLoader

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, Conv2DTranspose, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
from PIL import Image


GENERATOR_MODEL_A_B = "../model/image_generator_A_B_model.h5"
GENERATOR_MODEL_B_A = "../model/image_generator_B_A_model.h5"

CHECKPOINT_DIR = "../model/checkpoints"

IMG_DIR_VAL_A = "/Users/karin/programming/data/image_pairs/horse2zebra/val_A_horse"
IMG_DIR_VAL_B = "/Users/karin/programming/data/image_pairs/horse2zebra/val_B_zebra"

IMG_DIR_PREDICTED = "../test_predictions"

TRAIN_IMG_SIZE = 128
CHANNELS = 3

PERCENT_OF_TRAINING_SET=0.2  # check out actual training set (size) and if everything would fit into memory


# **************************************
#
# **************************************
class Utils():
    # ***************
    # inference
    # ***************
    def test_predict(self, generator, data_loader, val_img_dir, trained_steps, sub_dir):
        ''' Create prediction example of selected epoch '''
        def create_img_dir(trained_steps):
            ''' My default test dump '''
            save_img_dir = f"{IMG_DIR_PREDICTED}/{sub_dir}/{trained_steps}"
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
            return save_img_dir
        
        def resolve_single_image(val_img_file_path):
            val_img = data_loader.load_single_image(val_img_file_path, size=128)
            generated_img = generator.predict(val_img)
            generated_img = 0.5 * generated_img + 0.5
            return Image.fromarray((np.uint8(generated_img*255)[0])) 

        save_img_dir = create_img_dir(trained_steps)
        file_names = os.listdir(val_img_dir)
        
        for i, file_name in enumerate(file_names):  # here: 5 prediction examples
            val_img_file_path = f"{val_img_dir}/{file_name}"
            img = resolve_single_image(val_img_file_path)
            img.save(f"{save_img_dir}/{file_name}")


# **************************************
#
# **************************************
class CycleModel(): 
    def __init__(self, img_shape, number_resnet_blocks):
        self.img_shape = img_shape
        self.number_resnet_blocks = number_resnet_blocks  # 6 or 9

    
    def build_generator(self):
        '''
        See also: https://keras.io/examples/generative/cyclegan/
        n_resnet = 9 -> 256x256
        n_resnet = 6 -> 128x128
        InstanceNormalization: replacement for batch normalization (https://www.tensorflow.org/addons/tutorials/layers_normalizations)
        '''
        def resnet_block(filters, x_in):
            kernel_initializer = RandomNormal(stddev=0.02)
            
            # first convolutional layer
            x = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=kernel_initializer)(x_in)
            x = InstanceNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            
            # second convolutional layer
            x = Conv2D(filters=filters, kernel_size=(3,3), padding='same', kernel_initializer=kernel_initializer)(x)
            x = InstanceNormalization(axis=-1)(x)
            
            # concatenate input layer
            x = Concatenate()([x, x_in])

            return x


        def down_sampling(x_in, filters, kernel_size, kernel_initializer, use_strides=True):
            if use_strides:
                x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x_in)
            else:
                x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
            
            x = InstanceNormalization(axis=-1)(x)
            x = Activation('relu')(x)

            return x

        
        kernel_initializer = RandomNormal(stddev=0.02)
        
        x_in = Input(shape=self.img_shape)
        
        # ***
        # downsampling using conv2d
        # ***
        x = down_sampling(x_in=x_in, filters=64, kernel_size=(7,7), kernel_initializer=kernel_initializer, use_strides=False)
        x = down_sampling(x_in=x, filters=128, kernel_size=(3,3), kernel_initializer=kernel_initializer)
        x = down_sampling(x_in=x, filters=256, kernel_size=(2,2), kernel_initializer=kernel_initializer)
       
        # ***
        # transform hidden representation using resnet blocks
        # ***
        for _ in range(self.number_resnet_blocks):
            x = resnet_block(256, x)

        # ***
        # upsampling: recover transformed image
        # ***
        x = Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        x = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters=3, kernel_size=(7,7), padding='same', kernel_initializer=kernel_initializer)(x)
        x = InstanceNormalization(axis=-1)(x)

        x_out = Activation('tanh')(x)  # normalize output image with tanh activation
        
        return Model(x_in, x_out)


    def build_discriminator(self):
        def discriminator_block(x_in, filters, kernel_initializer, normalization=True, use_strides=True):
            if use_strides is True:
                x = Conv2D(filters=filters, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x_in)
            else:
                x = Conv2D(filters=filters, kernel_size=(4,4), padding='same', kernel_initializer=kernel_initializer)(x_in)
            
            if normalization:
                x = InstanceNormalization(axis=-1)(x)
            x = LeakyReLU(alpha=0.2)(x)

            return x


        kernel_initializer = RandomNormal(stddev=0.02)

        x_in = Input(shape=self.img_shape)

        x = discriminator_block(x_in, filters=64, normalization=False, kernel_initializer=kernel_initializer)
        x = discriminator_block(x, filters=128, kernel_initializer=kernel_initializer)
        x = discriminator_block(x, filters=256, kernel_initializer=kernel_initializer)
        x = discriminator_block(x, filters=512, kernel_initializer=kernel_initializer)

        x = discriminator_block(x, filters=512, use_strides=False, kernel_initializer=kernel_initializer)

        x_out = Conv2D(filters=1, kernel_size=(4,4), padding='same', kernel_initializer=kernel_initializer)(x)

        model = Model(x_in, x_out)
        model.compile(
            loss='mse', 
            optimizer=Adam(lr=0.0002, beta_1=0.5), 
            loss_weights=[0.5]
        )

        return model


    def build_composite(self, generator_a, generator_b, discriminator):
        generator_a.trainable = True
        generator_b.trainable = False

        discriminator.trainable = False
        
        x_in = Input(shape=self.img_shape)
        
        output_generator_a = generator_a(x_in)
        output_discriminator = discriminator(output_generator_a)
        
        # identity element
        input_id = Input(shape=self.img_shape)
        output_id = generator_a(input_id)
        
        # forward cycle
        output_forward = generator_b(output_generator_a)
        
        # backward cycle
        output_generator_b = generator_b(input_id)
        output_backward = generator_a(output_generator_b)
        
        # define model graph
        model = Model(
            [x_in, input_id], 
            [output_discriminator, output_id, output_forward, output_backward]
        )
        
        model.compile(
            loss=['mse', 'mae', 'mae', 'mae'], 
            loss_weights=[1, 5, 10, 10], 
            optimizer=Adam(lr=0.0002, beta_1=0.5)
        )
        
        return model


# **************************************
#
# **************************************
class Trainer():
    def __init__(self):
        # ***
        # Input shape
        # ***
        self.img_shape = (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, CHANNELS)
        print(self.img_shape)
        
        # ***
        # data loader and utils
        # ***
        self.data_loader = DataLoader(TRAIN_IMG_SIZE, PERCENT_OF_TRAINING_SET)
        self.utils = Utils()

        # ***
        # create cycleModel instancce
        # number_resnet_blocks in generator = 9 -> 256x256
        # number_resnet_blocks in generator = 6 -> 128x128
        # ***
        number_resnet_blocks = 9 if TRAIN_IMG_SIZE > 128 else 6
        self.cycle_model = CycleModel(self.img_shape, number_resnet_blocks)

        # ***
        # generators
        # ***
        self.generator_A_B = self.cycle_model.build_generator()  # A -> B
        self.generator_B_A = self.cycle_model.build_generator()  # B -> A

        # ***
        # discriminators
        # ***
        self.discriminator_A = self.cycle_model.build_discriminator()  # A -> [real/fake]
        self.discriminator_B = self.cycle_model.build_discriminator()  # B -> [real/fake]

        # ***
        # composites
        # ***
        self.composite_A_B = self.cycle_model.build_composite(self.generator_A_B, self.generator_B_A, self.discriminator_B)  # A -> B -> [real/fake, A]
        self.composite_B_A = self.cycle_model.build_composite(self.generator_B_A, self.generator_A_B, self.discriminator_A)  # B -> A -> [real/fake, B]

        # ***
        # get output square shape of the discriminator
        # ***
        self.patch = self.discriminator_A.output_shape[1]
       
        # ***
        # save training in checkpoint
        # ***
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            generator_A_B=self.generator_A_B,
            generator_B_A=self.generator_B_A,
            discriminator_A=self.discriminator_A,
            discriminator_B=self.discriminator_B,
            composite_A_B=self.composite_A_B,
            composite_B_A=self.composite_B_A
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=CHECKPOINT_DIR,
            max_to_keep=1
        )

        self.restore_checkpoint()


    # ***
    # get the latest checkpoint (if existing)
    # ***
    def restore_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restore checkpoint at step {self.checkpoint.step.numpy()}.")


    # ***
    #
    # ***
    def generate_fake_images(self, generator, batch_real_images):
        X = generator.predict(batch_real_images)
        y = np.zeros((len(X), self.patch, self.patch, 1))  # create 0 labels
	    
        return X, y

    
    def update_image_pool(self, fake_images, max_size=10):
        selected = []
        pool = []

        for image in fake_images:
            if len(pool) < max_size:
                pool.append(image)  # stock the pool
                selected.append(image)
            elif numpy.random.uniform(0.0, 1.0) < 0.5:
                selected.append(image)  # use image, but don't add it to the pool
            else:
                ix = np.random.randint(0, len(pool)) # replace an existing image and use replaced image
                selected.append(pool[ix])
                pool[ix] = image

        return np.asarray(selected)


    # ***
    #
    # ***
    def train_step(self, batch_size, X_real_A, X_real_B):
        y_real_A = np.ones((batch_size, self.patch, self.patch, 1))  # create 1 labels
        y_real_B = np.ones((batch_size, self.patch, self.patch, 1))

        # ***
        #
        # ***
        X_fake_A, y_fake_A = self.generate_fake_images(self.checkpoint.generator_B_A, X_real_B)
        X_fake_B, y_fake_B = self.generate_fake_images(self.checkpoint.generator_A_B, X_real_A)

        # ***
        #
        # ***
        X_fake_A = self.update_image_pool(X_fake_A)
        X_fake_B = self.update_image_pool(X_fake_B)
        
        # ***
        # train generator B->A only (use generator A->B and discriminator_A)
        # ***
        g_B_A_loss, _, _, _, _  = self.checkpoint.composite_B_A.train_on_batch(
            [X_real_B, X_real_A], 
            [y_real_A, X_real_A, X_real_B, X_real_A]
        )

        # ***
        # update discriminator for A -> [real/fake]
        # ***
        d_A_loss_real = self.checkpoint.discriminator_A.train_on_batch(X_real_A, y_real_A)
        d_A_loss_fake = self.checkpoint.discriminator_A.train_on_batch(X_fake_A, y_fake_A)
        
        # ***
        # train generator A->B only (use generator B->A and discriminator_B)
        # ***
        g_A_B_loss, _, _, _, _ = self.checkpoint.composite_A_B.train_on_batch(
            [X_real_A, X_real_B],
            [y_real_B, X_real_B, X_real_A, X_real_B]
        )

        # ***
        # update discriminator for B -> [real/fake]
        # # ***
        d_B_loss_real = self.checkpoint.discriminator_B.train_on_batch(X_real_B, y_real_B)
        d_B_loss_fake = self.checkpoint.discriminator_B.train_on_batch(X_fake_B, y_fake_B)

        return g_B_A_loss, d_A_loss_real, d_A_loss_fake, g_A_B_loss, d_B_loss_real, d_B_loss_fake


    # ***
    #
    # ***
    def train(self, epochs=10, batch_size=1, sample_interval=2):
        batch_per_epoch = int(self.data_loader.get_train_set_size() / batch_size)
        steps = batch_per_epoch * epochs

        print("*********")
        print(f"steps to train: {steps} | train set size: {self.data_loader.get_train_set_size()} image pairs")
        print("*********")

        start_time = datetime.datetime.now()  # for controle dump only

        for step in range(steps):
            self.checkpoint.step.assign_add(1)  # update steps in checkpoint
            trained_steps = self.checkpoint.step.numpy()  # overall trained steps: for controle dump only
            
            # ***
            # get real images and create image label (y)
            # ***
            X_real_A, X_real_B = self.data_loader.load_data(batch_size=batch_size)
            
            # ***
            # train model
            # ***
            g_B_A_loss, d_A_loss_real, d_A_loss_fake, g_A_B_loss, d_B_loss_real, d_B_loss_fake = self.train_step(batch_size, X_real_A, X_real_B)

            elapsed_time = datetime.datetime.now() - start_time  # for controle dump only

            # ***
            # save and/or dump
            # ***
            if (step+1) % 1 == 0:
                print(f"steps {trained_steps} | g B_A {round(g_B_A_loss, 3)} | d_A real {round(d_A_loss_real, 3)} fake {round(d_A_loss_fake, 3)} | g A_B {round(g_A_B_loss, 3)} | d_B real {round(d_B_loss_real, 3)} fake {round(d_B_loss_fake, 3)} | time: {elapsed_time}")
            if (step+1) % sample_interval == 0:
                print("   |---> save and make image sample")
                self.checkpoint_manager.save()  # save checkpoint
                self.checkpoint.generator_A_B.save(GENERATOR_MODEL_A_B)  # save generator model for usage
                self.checkpoint.generator_B_A.save(GENERATOR_MODEL_B_A)

                # controle dump of predicted images: save in dirs named by trained_steps
                self.utils.test_predict(self.checkpoint.generator_A_B, self.data_loader, IMG_DIR_VAL_A, trained_steps, "A_2_B")
                self.utils.test_predict(self.checkpoint.generator_B_A, self.data_loader, IMG_DIR_VAL_B, trained_steps, "B_2_A")



if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(epochs=1, batch_size=1, sample_interval=2)  # set_interval refers to steps
