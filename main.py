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
    print('File path: ' + file_path)
    image = np.array([ img_to_array(load_img(file_path, target_size=(150, 150))) ])
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(image)
    layer_activation = activations[int(layer_index.get())]
    print(layer_activation)
    print(layer_activation.shape)
    plt.matshow(layer_activation[0, :, :, int(channel_index.get())], cmap='viridis')
    plt.show()

    

# Create the main window
root = tk.Tk()
root.geometry('600x600')
root.title('Dog and Cat Predictor')

# Create a label to display the image
input_image = tk.Label(root)
input_image.pack()

activation_image = tk.Label(root)
activation_image.pack()

# Create a button to open the image
open_button = tk.Button(root, text='Open Image', command=open_image)
open_button.pack(pady=10)

# Create a button to predict
predict_button = tk.Button(root, text='Predict', command=predict)
predict_button.pack(pady=10)

layer_activation_button = tk.Button(root, text='Layer Activation', command=show_activation)
layer_activation_button.pack(pady=10)

tk.Scale(root, from_=0, to=len(model.layers), orient='horizontal').pack()
layer_index = tk.Entry(root, width=20)
layer_index.pack()

tk.Scale(root, from_=0, to=128).pack()
channel_index = tk.Entry(root, width=20)
channel_index.pack()

# Create a label to display the prediction
prediction_label = tk.Label(root, text="")
prediction_label.pack()


# Run the application
root.mainloop()