from keras import layers
from keras import models
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


train_ds = image_dataset_from_directory(
    directory='data/',
    validation_split=0.2,
    subset='training',
    seed=123,
    labels='inferred',
    label_mode='binary',
    batch_size=50,
    image_size=(150, 150)
)

val_ds = image_dataset_from_directory(
    directory='data/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    labels='inferred',
    label_mode='binary',
    batch_size=50,
    image_size=(150, 150)
)

model = models.Sequential([
    layers.Resizing(150, 150),
    layers.Rescaling(1./127.5, offset=-1),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Dropout(.1),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Dropout(.1),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Dropout(.1),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Dropout(.1),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(64, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(64, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=5, validation_data=val_ds)


print("Plotting model:")

epochs = range(1, len(history.history['accuracy']) + 1)

plt.plot(epochs, history.history['accuracy'], 'bo', label='accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'b', label = 'val_accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, history.history['loss'], 'bo', label='loss')
plt.plot(epochs, history.history['val_loss'], 'b', label = 'val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('model1.keras')