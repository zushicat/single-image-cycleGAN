
'''
See also: https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py
'''
import datetime
import os

from data_loader import DataLoader

import tensorflow as tf

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import UpSampling2D, Activation, Add, Concatenate, Conv2D, Input, LeakyReLU
from tensorflow.keras.models import Model, Sequential
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

PERCENT_OF_TRAINING_SET=0.05  # check out actual training set (size) and if everything would fit into memory


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
    def __init__(self, img_shape, loss_cycle, loss_identity, optimizer):
        self.img_shape = img_shape
        self.channels = self.img_shape[2]
        self.loss_cycle = loss_cycle
        self.loss_identity = loss_identity
        self.optimizer = optimizer # ? use the same optimizer for both discr. & combined network ?


    def build_generator(self):
        def down_sampling(x_in, filters, kernel_size):
            '''
            conv2d
            '''
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(2,2), padding="same")(x_in)
            x = InstanceNormalization(axis=-1)(x)
            x = Activation('relu')(x)

            return x


        def up_sampling(x_in, skip_input, filters, kernel_size, dropout_rate=0):
            '''
            deconv2d
            '''
            x = UpSampling2D(size=2)(x_in)
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding="same", activation="relu")(x)
            if dropout_rate:  # ?
                x = Dropout(dropout_rate)(x)
            x = InstanceNormalization()(x)
            x = Concatenate()([x, skip_input])
            return x

        
        x_in = Input(shape=self.img_shape)
        
        # ***
        # downsampling: conv2d
        # ***
        d_1 = down_sampling(x_in=x_in, filters=32, kernel_size=(4,4))
        d_2 = down_sampling(x_in=d_1, filters=64, kernel_size=(4,4))
        d_3 = down_sampling(x_in=d_2, filters=128, kernel_size=(4,4))
        d_4 = down_sampling(x_in=d_3, filters=256, kernel_size=(4,4))

        # ***
        # upsampling: deconv2d
        # ***
        u_1 = up_sampling(x_in=d_4, skip_input=d_3, filters=128, kernel_size=(4,4))
        u_2 = up_sampling(x_in=u_1, skip_input=d_2, filters=64, kernel_size=(4,4))
        u_3 = up_sampling(x_in=u_2, skip_input=d_1, filters=32, kernel_size=(4,4))

        u_4 = UpSampling2D(size=2)(u_3)
        
        x_out = Conv2D(self.channels, kernel_size=4, strides=(1,1), padding="same", activation="tanh")(u_4)

        return Model(x_in, x_out)


    def build_discriminator(self):
        def discriminator_block(x_in, filters, normalization=True):
            x = Conv2D(filters=filters, kernel_size=(4,4), strides=(2,2), padding="same")(x_in)
            x = LeakyReLU(alpha=0.2)(x)
            if normalization:
                x = InstanceNormalization()(x)

            return x


        x_in = Input(shape=self.img_shape)

        x = discriminator_block(x_in=x_in, filters=64, normalization=False)
        x = discriminator_block(x_in=x, filters=128)
        x = discriminator_block(x_in=x, filters=256)
        x = discriminator_block(x_in=x, filters=512)

        x_out = Conv2D(filters=1, kernel_size=(4,4), padding="same")(x)

        model = Model(x_in, x_out)
        
        model.compile(
            loss="mse",
            optimizer=self.optimizer,
            metrics=["accuracy"]
        )

        return model


    def build_combined(self, generator_a_b, generator_b_a, discriminator_a, discriminator_b):
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_B = generator_a_b(img_A)
        fake_A = generator_b_a(img_B)

        reconstr_A = generator_b_a(fake_B)
        reconstr_B = generator_a_b(fake_A)

        img_A_id = generator_b_a(img_A)
        img_B_id = generator_a_b(img_B)

        discriminator_a.trainable = False
        discriminator_b.trainable = False

        valid_A = discriminator_a(fake_A)
        valid_B = discriminator_b(fake_B)

        model = Model(
            inputs=[img_A, img_B],
            outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id ]
        )

        model.compile(
            loss=["mse", "mse", "mae", "mae", "mae", "mae"],
            loss_weights=[1, 1, self.loss_cycle, self.loss_cycle, self.loss_identity, self.loss_identity],
            optimizer=self.optimizer
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
        
        # ***
        # data loader and utils
        # ***
        self.data_loader = DataLoader(TRAIN_IMG_SIZE, PERCENT_OF_TRAINING_SET)
        self.utils = Utils()

        # ***
        # loss weights
        # ***
        self.loss_cycle = 10.0  # cycle consistency loss
        self.loss_identity = 0.1 * self.loss_cycle  # identity loss

        # ***
        # optimizer
        # ***
        self.optimizer = Adam(0.0002, 0.5)

        # ***
        # create cycleModel instancce
        # ***
        self.cycle_model = CycleModel(self.img_shape, self.loss_cycle, self.loss_identity, self.optimizer)

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
        self.combined = self.cycle_model.build_combined(self.generator_A_B, self.generator_B_A, self.discriminator_A, self.discriminator_B)  # A -> B -> [real/fake, A]
        
        # ***
        # get output square shape of the discriminator
        # ***
        self.discriminator_patch = (
            self.discriminator_A.output_shape[1], 
            self.discriminator_A.output_shape[1], 
            1
        )

        # ***
        # adversarial loss ground truths
        # set valid and fake with used batch size in train()
        # ***
        self.valid = np.ones((1,) + self.discriminator_patch)
        self.fake = np.zeros((1,) + self.discriminator_patch)
        
        # ***
        # save training in checkpoint
        # ***
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            generator_A_B=self.generator_A_B,
            generator_B_A=self.generator_B_A,
            discriminator_A=self.discriminator_A,
            discriminator_B=self.discriminator_B,
            combined=self.combined
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
    def train_step(self, X_real_A, X_real_B):
        X_fake_B = self.checkpoint.generator_A_B.predict(X_real_A)
        X_fake_A = self.checkpoint.generator_B_A.predict(X_real_B)

        # ***
        # train discriminators
        # ***
        dA_loss_real = self.checkpoint.discriminator_A.train_on_batch(X_real_A, self.valid)
        dA_loss_fake = self.checkpoint.discriminator_A.train_on_batch(X_fake_A, self.fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = self.checkpoint.discriminator_B.train_on_batch(X_real_B, self.valid)
        dB_loss_fake = self.checkpoint.discriminator_B.train_on_batch(X_fake_B, self.fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # total disciminator loss
        d_loss = 0.5 * np.add(dA_loss, dB_loss)


        # ***
        # train generators in combined network
        # ***
        g_loss = self.checkpoint.combined.train_on_batch(
            [X_real_A, X_real_B],
            [self.valid, self.valid, X_real_A, X_real_B, X_real_A, X_real_B]
        )

        return d_loss, g_loss


    # ***
    #
    # ***
    def train(self, epochs=10, batch_size=1, sample_interval=2):
        batch_per_epoch = int(self.data_loader.get_train_set_size() / batch_size)
        steps = batch_per_epoch * epochs
        sample_interval_steps = int(steps/epochs)*sample_interval

        print("*********")
        print(f"steps to train: {steps} | train set size: {self.data_loader.get_train_set_size()} image pairs")
        print(f"save every {sample_interval_steps} steps")
        print("*********")

        start_time = datetime.datetime.now()  # for controle dump only

        # ***
        # adversarial loss ground truths
        # ***
        self.valid = np.ones((batch_size,) + self.discriminator_patch)
        self.fake = np.zeros((batch_size,) + self.discriminator_patch)

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
            d_loss, g_loss = self.train_step(X_real_A, X_real_B)

            elapsed_time = datetime.datetime.now() - start_time  # for controle dump only

            # ***
            # save and/or dump
            # ***
            if (step+1) % 50 == 0:
                print(f"{step+1} steps {trained_steps} | g loss {round(g_loss[0],4)} | d loss {round(d_loss[0],4)} | time: {elapsed_time}")
            if (step+1) % sample_interval_steps == 0:
                print("   |---> save and make image sample")
                self.checkpoint_manager.save()  # save checkpoint
                self.checkpoint.generator_A_B.save(GENERATOR_MODEL_A_B)  # save generator model for usage
                self.checkpoint.generator_B_A.save(GENERATOR_MODEL_B_A)

                # controle dump of predicted images: save in dirs named by trained_steps
                self.utils.test_predict(self.checkpoint.generator_A_B, self.data_loader, IMG_DIR_VAL_A, trained_steps, "A_2_B")
                self.utils.test_predict(self.checkpoint.generator_B_A, self.data_loader, IMG_DIR_VAL_B, trained_steps, "B_2_A")



if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(epochs=100, batch_size=4, sample_interval=20)  # set_interval refers to epochs
