import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import glob

from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia

from deepposekit.models import (StackedDenseNet, DeepLabCut, StackedHourglass, LEAP)
from deepposekit.models import load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from deepposekit.callbacks import Logger, ModelCheckpoint


import time
from os.path import expanduser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_generator = DataGenerator("/media/storage2/open_monkey/train_data.h5")

image, keypoints = data_generator[0]

plt.figure(figsize=(5,5))
image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
cmap = None if image.shape[-1] is 3 else 'gray'
plt.imshow(image, cmap=cmap, interpolation='none')
for idx, jdx in enumerate(data_generator.graph):
        if jdx > -1:
                    plt.plot(
                        [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
                        [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
                        'r-'
                        )

plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)
plt.show()


augmenter = []

augmenter.append(FlipAxis(data_generator, axis=0))  # flip image up-down
augmenter.append(FlipAxis(data_generator, axis=1))  # flip image left-right

sometimes = []
#sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
#                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
#                shear=(-8, 8),
#                order=ia.ALL,
#                cval=ia.ALL,
#                mode=ia.ALL)
#                                  )
#sometimes.append(iaa.Affine(scale=(0.8, 1.2),
#                mode=ia.ALL,
#                order=ia.ALL,
#                cval=ia.ALL)
#                 )
#augmenter.append(iaa.Sometimes(0.75, sometimes))
#augmenter.append(iaa.Affine(rotate=(-180, 180),
#                mode=ia.ALL,
#                order=ia.ALL,
#                cval=ia.ALL)
#                 )
augmenter = iaa.Sequential(augmenter)


#image, keypoints = data_generator[0]
#image, keypoints = augmenter(images=image, keypoints=keypoints)
#plt.figure(figsize=(5,5))
#image = image[0] if image.shape[-1] is 3 else image[0, ..., 0]
#cmap = None if image.shape[-1] is 3 else 'gray'
#plt.imshow(image, cmap=cmap, interpolation='none')
#for idx, jdx in enumerate(data_generator.graph):
#        if jdx > -1:
#                    plt.plot(
#                        [keypoints[0, idx, 0], keypoints[0, jdx, 0]],
#                        [keypoints[0, idx, 1], keypoints[0, jdx, 1]],
#                        'r-'
#                    )
#plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c=np.arange(data_generator.keypoints_shape[0]), s=50, cmap=plt.cm.hsv, zorder=3)
#plt.show()


train_generator = TrainingGenerator(generator=data_generator,
                                downsample_factor=2,
                                augmenter=augmenter,
                                sigma=5,
                                validation_split=0.1,
                                use_graph=True,
                                random_seed=2,
                                graph_scale=1)
train_generator.get_config()


#n_keypoints = data_generator.keypoints_shape[0]
#batch = train_generator(batch_size=1, validation=False)[0]
#inputs = batch[0]
#outputs = batch[1]
#
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
#ax1.set_title('image')
#ax1.imshow(inputs[0,...,0], cmap='gray', vmin=0, vmax=255)
#
#ax2.set_title('posture graph')
#ax2.imshow(outputs[0,...,n_keypoints:-1].max(-1))
#
#ax3.set_title('keypoints confidence')
#ax3.imshow(outputs[0,...,:n_keypoints].max(-1))
#
#ax4.set_title('posture graph and keypoints confidence')
#ax4.imshow(outputs[0,...,-1], vmin=0)
#plt.show()

train_generator.on_epoch_end()



#model = LEAP(train_generator)
model = DeepLabCut(train_generator)




data_size = (10000,) + data_generator.image_shape
x = np.random.randint(0, 255, data_size, dtype="uint8")
y = model.predict(x[:100], batch_size=100) # make sure the model is in GPU memory
t0 = time.time()
y = model.predict(x, batch_size=100, verbose=1)
t1 = time.time()
print((t1 - t0) / x.shape[0])

#t0 = time.time()
#for i in range(100):
#    X, y = train_generator[i]
#t1 = time.time()
#print("Time per batch: ", (t1 - t0) / 100)


logger = Logger(validation_batch_size=10,
                    # filepath saves the logger data to a .h5 file
                    filepath="/media/storage2/open_monkey/log_deep.h5"
                )


reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=2, patience=20)


model_checkpoint = ModelCheckpoint(
        "/media/storage2/open_monkey/checkpoints/best_DLC.h5",
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        verbose=2,
        save_best_only=True,
)

model.distribute_strategy = tf.distribute.MirroredStrategy()

early_stop = EarlyStopping(
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        min_delta=0.001,
        patience=100,
        verbose=2
)
callbacks = [model_checkpoint, logger]

print("starting fit")
model.fit(
        batch_size=16,
        validation_batch_size=10,
        callbacks=callbacks,
        #epochs=1000, # Increase the number of epochs to train the model longer
        epochs=200,
        n_workers=8,
        steps_per_epoch=None,
)
