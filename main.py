from keras import models
from keras.utils import img_to_array
from keras.utils import load_img
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

file_path = ''
model = models.load_model('model.keras')

layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)

print(layer_names)

def open_image():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        input_image.config(image=photo)
        input_image.image = photo

def predict():
    global file_path
    image = np.array([ img_to_array(load_img(file_path, target_size=(150, 150))) ])

    prediction = model.predict(image)[0][0]
    confidence = round(abs(prediction - 0.5) * 2 * 100)
    if (prediction < 0.5): prediction = 'Cat'
    elif (prediction > 0.5): prediction = 'Dog'
    else: prediction = 'No idea'

    prediction_label.config(text=f'{str(prediction)} with {confidence}% confidence')

def show_activation():
    global file_path
    image = np.array([ img_to_array(load_img(file_path, target_size=(150, 150))) ])
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(image)
    layer_activation = activations[int(layer_index.get())]

    imageData = layer_activation[0, :, :, int(channel_index.get())]
    # label the activation image
    layer_name_label.config(text='Activation of ' + layer_names[int(layer_index.get())] + ' on channel ' + str(channel_index.get()))
    # normalize array
    imageData = (imageData - imageData.min()) / (imageData.max() - imageData.min()) * 256
    # save picture
    picture = Image.fromarray(imageData).convert('RGB').resize((200, 200))
    picture.save('activation.jpg')
    # load back the image
    image = Image.open('activation.jpg')
    image = ImageTk.PhotoImage(image)
    # display the image
    activation_image.config(image=image)
    activation_image.image = image

# Create the main window
root = tk.Tk()
root.geometry('600x600')
root.title('Dog and Cat Predictor')

predict_frame = tk.Frame(root)
predict_frame.pack(side=tk.LEFT)

# Create a label to display the image
input_image = tk.Label(predict_frame)
input_image.pack()

# Create a button to open the image
open_button = tk.Button(predict_frame, text='Open Image', command=open_image)
open_button.pack(pady=10)

# Create a button to predict
predict_button = tk.Button(predict_frame, text='Predict', command=predict)
predict_button.pack(pady=10)

# Create a label to display the prediction
prediction_label = tk.Label(predict_frame, text="")
prediction_label.pack()

# Frame for activation analysis
activation_frame = tk.Frame(root)
activation_frame.pack(side=tk.RIGHT)

layer_name_label = tk.Label(activation_frame, text='Activation')
layer_name_label.pack()
activation_image = tk.Label(activation_frame)
activation_image.pack()

layer_activation_button = tk.Button(activation_frame, text='Layer Activation', command=show_activation)
layer_activation_button.pack(pady=10)

layer_index = tk.Scale(activation_frame, length=200, from_=0, to=len(model.layers), orient='horizontal', command=lambda value: show_activation())
layer_index.pack()

channel_index = tk.Scale(activation_frame, length=200, from_=0, to=128, orient='horizontal', command=lambda value: show_activation())
channel_index.pack()

# Run the application
root.mainloop()