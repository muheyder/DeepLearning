import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import sys

model = load_model("modelo_digitos.h5")

def preparar_imagem(caminho):
    img = Image.open(caminho).convert('L')
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = 1 - img
    return img.reshape(1, 28, 28, 1)

if len(sys.argv) != 2:
    print("Uso: python reconhecer.py exemplos/exemplo_0.png")
    exit()

imagem = preparar_imagem(sys.argv[1])
pred = model.predict(imagem)
classe = np.argmax(pred)

print("Probabilidades por classe:")
for i, p in enumerate(pred[0]):
    print(f"{i}: {p:.4f}")

print(f"\nEste dígito parece ser: {classe}")
