
'''
See also: https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
'''
import datetime
import os
import pickle

from data_loader import DataLoader

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Add, Concatenate, Conv2D, Conv2DTranspose, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
from PIL import Image


PRETRAINED_GENERATOR_WEIGHTS = "../model/weights/pretrained_generator.h5"
GENERATOR_MODEL = "../model/image_generator_model.h5"

PRETRAINED_GENERATOR_CHECKPOINT_DIR = "../model/checkpoints/pre_train"
FINE_TUNE_CHECKPOINT_DIR = "../model/checkpoints/fine_tune"

#IMG_DIR_VAL_LR = "/Users/karin/programming/data/ortho-images/default_test_images/lr"
#IMG_DIR_PREDICTED = "../test_predictions"

TRAIN_IMG_SIZE = 256
CHANNELS = 3

PERCENT_OF_TRAINING_SET=0.05  # check out actual training set (size) and if everything would fit into memory


# **************************************
#
# **************************************
# class Utils():
#     # ***************
#     # inference
#     # ***************
#     def test_predict(self, model, data_loader, lr_dir_path, trained_steps, sub_dir):
#         pass
#         # ''' Create prediction example of selected epoch '''
#         # def create_img_dir(trained_steps):
#         #     ''' My default test dump '''
#         #     save_img_dir = f"{IMG_DIR_PREDICTED}/{sub_dir}/{trained_steps}"
#         #     if not os.path.exists(save_img_dir):
#         #         os.makedirs(save_img_dir)
#         #     return save_img_dir
        
#         # def resolve_single_image(lr_file_path):
#         #     lr_img = data_loader.load_single_image(lr_file_path, size=100)
#         #     generated_hr = model.generator.predict(lr_img)
#         #     generated_hr = 0.5 * generated_hr + 0.5
#         #     return Image.fromarray((np.uint8(generated_hr*255)[0])) 

#         # save_img_dir = create_img_dir(trained_steps)
#         # file_names = os.listdir(lr_dir_path)
        
#         # for i, file_name in enumerate(file_names[:5]):  # here: 5 prediction examples
#         #     lr_file_path = f"{lr_dir_path}/{file_name}"
#         #     img = resolve_single_image(lr_file_path)
#         #     img.save(f"{save_img_dir}/{file_name}")


# **************************************
#
# **************************************
class CycleModel(tf.Module):  # regarding parameter: https://stackoverflow.com/a/60509193
    def __init__(self, img_shape):
        self.img_shape = img_shape
 
    
    def build_generator(n_resnet=9):
        '''
        n_resnet = 9 -> 256x256
        n_resnet = 6 -> 128x128
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


        def add_layer(x_in, filters, kernel_size, kernel_initializer, use_strides=True):
            if use_strides:
                x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x_in)
            else:
                x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
            
            x = InstanceNormalization(axis=-1)(x)
            x = Activation('relu')(x)

            return x

        
        kernel_initializer = RandomNormal(stddev=0.02)
        
        x_in = Input(shape=self.img_shape)
        
        x = add_layer(x_in=x_in, filters=64, kernel_size=(7,7), kernel_initializer=kernel_initializer, use_strides=False)
        x = add_layer(x_in=x, filters=128, kernel_size=(3,3), kernel_initializer=kernel_initializer)
        x = add_layer(x_in=x, filters=256, kernel_size=(2,2), kernel_initializer=kernel_initializer)
       
        for _ in range(n_resnet):
            x = resnet_block(256, x)

        x = Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        x = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        
        x = Conv2D(filters=3, kernel_size=(7,7), padding='same', kernel_initializer=kernel_initializer)(x)
        x = InstanceNormalization(axis=-1)(x)
        x_out = Activation('tanh')(x)
        
        return Model(x_in, x_out)


    def build_discriminator(self):
        def discriminator_block(x_in, filters, kernel_initializer, strides=1, normalization=True, use_strides=True):
            if use_kernel_size:
                x = Conv2D(filters, filter=filters, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x_in)
            else:
                x = Conv2D(filters, filter=filters, kernel_size=(4,4), padding='same', kernel_initializer=kernel_initializer)(x_in)
            
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

        x_out = Conv2D(filter=1, kernel_size=(4,4), padding='same', kernel_initializer=kernel_initializer)(d)

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
    def __init__(self, use_pretrain_weights=False):
        # ***
        # Input shape
        # ***
        self.img_shape = (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, CHANNELS)
        
        # ***
        # data loader and utils
        # ***
        self.data_loader = DataLoader(TRAIN_IMG_SIZE, PERCENT_OF_TRAINING_SET)
        # self.utils = Utils()

        # ***
        #
        # ***
        self.cycle_model = CycleModel(self.img_shape)

        # ***
        # generators
        # ***
        self.generator_A_B = self.cycle_model.build_generator()  # A -> B
        self.generator_B_A = self.cycle_model.build_generator()  # B -> A

        # ***
        # discriminators
        # ***
        self.discriminator_A = self.cycle_model.build_discriminator(image_shape)  # A -> [real/fake]
        self.discriminator_B = self.cycle_model.build_discriminator(image_shape)  # B -> [real/fake]

        # ***
        # composites
        # ***
        self.composite_A_B = self.cycle_model.build_composite(generator_A_B, generator_B_A, discriminator_B)  # A -> B -> [real/fake, A]
        self.composite_B_A = self.cycle_model.build_composite(generator_B_A, generator_A_B, discriminator_A)  # B -> A -> [real/fake, B]

        # ***
        # get output shape of the discriminator
        # ***
        self.n_patch = self.discriminator_A.output_shape[1]

        # # ***
        # # save training in checkpoint
        # # necessary to keep all values (i.e. optimizer) when interrupting & resuming training
        # # ***
        # self.checkpoint = tf.train.Checkpoint(
        #     step=tf.Variable(0),
        #     # optimizer_generator=Adam(learning_rate=LEARNING_RATE),
        #     # optimizer_discriminator=Adam(learning_rate=LEARNING_RATE),
        #     # model=CycleModel(self.hr_shape, self.lr_shape, self.channels)
        # )

        # self.checkpoint_manager = tf.train.CheckpointManager(
        #     checkpoint=self.checkpoint,
        #     #directory=FINE_TUNE_CHECKPOINT_DIR,
        #     max_to_keep=1
        # )

        # # ***
        # # either: start training with pretrained generator weights a) if parameter is True and b) if weights are available
        # # or: get latest checkpoint of training  a) if parameter is False and b) if checkpoint is available
        # # ***
        # self.restore_checkpoint()


    # ***
    # get the latest checkpoint (if existing)
    # ***
    def restore_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"Restore checkpoint at step {self.checkpoint.step.numpy()}.")

    # ***
    # loss functions, used in training step
    # ***
    

    # ***
    #
    # ***
    @tf.function
    def train_step(self, lr_img, hr_img):
        pass
        # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #     hr_generated = self.checkpoint.model.generator(lr_img, training=True)

        #     # ***
        #     # discriminator
        #     # ***
        #     hr_output = self.checkpoint.model.discriminator(hr_img, training=True)
        #     hr_generated_output = self.checkpoint.model.discriminator(hr_generated, training=True)

        #     fake_logit, real_logit = self.relativistic_loss(hr_output, hr_generated_output)

        #     discriminator_loss = self.discriminator_loss(fake_logit, real_logit)

        #     # ***
        #     # generator
        #     # ***
        #     content_loss = self.content_loss(hr_img, hr_generated)
        #     generator_loss = self.generator_loss(real_logit, fake_logit)
        #     perceptual_loss = content_loss + 0.001 * generator_loss
            

        # gradients_generator = gen_tape.gradient(perceptual_loss, self.checkpoint.model.generator.trainable_variables)
        # gradients_discriminator = disc_tape.gradient(discriminator_loss, self.checkpoint.model.discriminator.trainable_variables)

        # self.checkpoint.optimizer_generator.apply_gradients(zip(gradients_generator, self.checkpoint.model.generator.trainable_variables))
        # self.checkpoint.optimizer_discriminator.apply_gradients(zip(gradients_discriminator, self.checkpoint.model.discriminator.trainable_variables))

        # return perceptual_loss, discriminator_loss

    
    # ***
    #
    # ***
    def train(self, epochs=10, batch_size=1, sample_interval=2, blur_lr_images=False):
        start_time = datetime.datetime.now()  # for controle dump only

        for epoch in range(epochs):
            # self.checkpoint.step.assign_add(batch_size)  # update steps in checkpoint
            # trained_steps = self.checkpoint.step.numpy() # overall trained steps: for controle dump only
            
            # ***
            # train on batch
            # ***
            # imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size, blur_lr_images=blur_lr_images)
            # perceptual_loss, discriminator_loss = self.train_step(imgs_lr, imgs_hr)

            elapsed_time = datetime.datetime.now() - start_time  # for controle dump only

            # ***
            # save and/or dump
             # ***
            if (epoch + 1) % 10 == 0:
                print("xxx")
                # print(f"{epoch + 1} | steps: {trained_steps} | g_loss: {perceptual_loss} | d_loss: {discriminator_loss} | time: {elapsed_time}")
            if (epoch + 1) % sample_interval == 0:
                print("   |---> save and make image sample")
                # self.checkpoint_manager.save()  # save checkpoint
                # self.checkpoint.model.generator.save(GENERATOR_MODEL)  # save complete model for actual usage

                # # controle dump of predicted images: save in dirs named by trained_steps
                # self.utils.test_predict(self.checkpoint.model, self.data_loader, IMG_DIR_VAL_LR, trained_steps, "fine_tune")
        


if __name__ == '__main__':
    trainer = Trainer(use_pretrain_weights=True)  # use this parameter on very first training run (default: False)
    # trainer = Trainer()  # use this if you continue training (i.e. after interruption)
    # trainer.train(epochs=4000, batch_size=4, sample_interval=1000, blur_lr_images=True)  # blur_lr_images default: False
