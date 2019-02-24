#%%
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#%%
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# adapt this if using `channels_first` image data format


#%%
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# x_train_noisy 被限制在0-1之间
# 修剪后的数组存入到x_train_noisy中
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#%%
def train_model():
    # adapt this if using `channels_first` image data format
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16,
               (3, 3),
               activation='relu',
               padding='same')(input_img)
    x = MaxPooling2D((2, 2),
                     padding='same')(x)
    x = Conv2D(8,
               (3, 3),
               activation='relu',
               padding='same')(x)
    x = MaxPooling2D((2, 2),
                     padding='same')(x)
    x = Conv2D(8,
               (3, 3),
               activation='relu',
               padding='same')(x)
    encoded = MaxPooling2D((2, 2),
                           padding='same',
                           name='encoder')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8,
               (3, 3),
               activation='relu',
               padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8,
               (3, 3),
               activation='relu',
               padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16,
               (3, 3),
               activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1,
                     (3, 3),
                     activation='sigmoid',
                     padding='same')(x)

    # 训练开始
    autoencoder = Model(input_img,
                        decoded)
    autoencoder.compile(optimizer='adadelta',
                        loss='binary_crossentropy')

    autoencoder.fit(x_train_noisy,
                    x_train,
                    epochs=20,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/tb',
                                           histogram_freq=0,
                                           write_graph=False)])

    # save
    autoencoder.save('autoencoder.h5')

train_model()
# the code is followed: https://blog.keras.io/building-autoencoders-in-keras.html
# https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511
# and the paper





