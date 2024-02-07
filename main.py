import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

model = load_model("200epoch.h5")

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(100, 120, 3))
    img_array = image.img_to_array(img)
    #img_array /= 255
    img_array = preprocess_input(img_array.reshape(1, *img_array.shape))
    return img_array

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        if hasattr(window, "panel"):
            window.panel.pack_forget()

        display_image(file_path)


def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((200, 240), Image.NEAREST)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(window, image=img)
    panel.image = img
    panel.pack(padx=10, pady=10)

    processed_img = load_and_preprocess_image(file_path)

    prediction = model.predict(processed_img)

    result = "İmza Gerçek" if prediction[0, 0] > 0.5 else "İmza Sahte"
    result_label.config(text=f"Modelin Tahmini: {result}")

    window.panel = panel

window = tk.Tk()
window.title("Görüntü Tahmin Arayüzü")
window.geometry("600x400")

open_button = tk.Button(window, text="Dosya Seç", command=open_file_dialog)
open_button.pack(pady=20)

result_label = tk.Label(window, text="Modelin Tahmini: ")
result_label.pack(pady=10)

window.mainloop()