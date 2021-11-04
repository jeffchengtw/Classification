from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

import  keras_efficientnet_v2

# train param
batch_size = 4
image_size = (480,480)
epochs = 100

# create Generator
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True
)
test_gen = ImageDataGenerator(
    rescale=1./255
)
train_ds = train_gen.flow_from_directory(
    'dataset/train', 
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
val_ds = test_gen.flow_from_directory(
    'dataset/val', 
    target_size = image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# build model
activeOptimizer = optimizers.Adam(learning_rate=0.0001, amsgrad=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, model='auto', epsilon=0.0001)
ckpt = ModelCheckpoint(filepath='bestModelPath',
                                    monitor='val_loss',  # loss, val_loss, val_accuracy
                                    save_best_only=True,
                                    mode='min',  # min max auto
                                    verbose=1)

model = keras_efficientnet_v2.EfficientNetV2M(input_shape=(image_size[0],image_size[1],3), num_classes=200,pretrained='imagenet21k-ft1k', classifier_activation='softmax', dropout=0.2)
   
model.compile(optimizer=activeOptimizer,
                loss="categorical_crossentropy",
                metrics=['accuracy']
                     )

#start training                     
history = model.fit(
    train_ds,
    validation_data = val_ds,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[rlr,ckpt]
)

# training curve
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
