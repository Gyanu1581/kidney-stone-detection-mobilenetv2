import tkinter as tk
from tkinter import filedialog
from predict import predict_image

def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if file_path:
        result = predict_image(file_path)
        result_label.config(text=result)

root = tk.Tk()
root.title("Kidney Stone Detection System")
root.geometry("400x300")

title = tk.Label(root, text="Kidney Stone Detection", font=("Arial", 16))
title.pack(pady=20)

upload_btn = tk.Button(root, text="Upload CT Image", command=upload_image)
upload_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()
