import keras
from keras import layers
from keras import models
from keras.utils import image_dataset_from_directory
import tensorflow as tf
import matplotlib.pyplot as plt


train_ds = image_dataset_from_directory(
    directory='train/',
    labels='inferred',
    label_mode='binary',
    batch_size=20,
    image_size=(150, 150)
)


val_ds = image_dataset_from_directory(
    directory='test/',
    labels='inferred',
    label_mode='binary',
    batch_size=20,
    image_size=(150, 150)
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=5, validation_data=val_ds)

print("Plotting model:")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

model.save('model.keras')