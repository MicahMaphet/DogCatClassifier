import keras
from keras import models
import numpy as np


images = np.array([
    keras.utils.img_to_array(keras.utils.load_img('test/cat/cat.0.jpg', target_size=(150, 150))),
    keras.utils.img_to_array(keras.utils.load_img('test/cat/cat.1.jpg', target_size=(150, 150))),
    keras.utils.img_to_array(keras.utils.load_img('test/dog/dog.0.jpg', target_size=(150, 150))),
    keras.utils.img_to_array(keras.utils.load_img('test/dog/dog.1.jpg', target_size=(150, 150)))
])

model = models.load_model('model.keras')
predictions = model.predict(images)

print('Cat, Cat, Dog, Dog')
print(predictions)