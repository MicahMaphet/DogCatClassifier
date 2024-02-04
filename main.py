from keras import models
from keras.utils import img_to_array
from keras.utils import load_img
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

file_path = ''
model = models.load_model('model1.keras')
def open_image():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

def predict():
    global file_path
    image = np.array([ img_to_array(load_img(file_path, target_size=(150, 150))) ])

    prediction = model.predict(image)[0][0]
    print(prediction)
    confidence = round(abs(prediction - 0.5) * 2 * 100)
    if (prediction < 0.5): prediction = 'Cat'
    elif (prediction > 0.5): prediction = 'Dog'
    else: prediction = 'No idea'

    prediction_label.config(text=f'{str(prediction)} with {confidence}% confidence')

# Create the main window
root = tk.Tk()
root.title("Dog and Cat Predictor")

# Create a label to display the image
label = tk.Label(root)
label.pack()
# Create a button to open the image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Create a button to predict
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

# Create a label to display the prediction
prediction_label = tk.Label(root, text="")
prediction_label.pack()

# Run the application
root.mainloop()