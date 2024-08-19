import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Laad het opgeslagen model
model = tf.keras.models.load_model('mijn_model.h5')

# Functie om afbeeldingen te laden en voor te bereiden
def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (hoogte, breedte, kanalen)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, hoogte, breedte, kanalen)
    img_tensor /= 255.                                      # Model verwacht waarden tussen 0 en 1

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

# Voorbeeldafbeelding laden en voorspelling maken
img_path = 'dataset/test/tulip/4571993204_5b3efe0e78.jpg'
new_image = load_image(img_path, show=True)
pred = model.predict(new_image)
predicted_class = np.argmax(pred)
class_labels = {0: 'Klasse1', 1: 'Klasse2', 2: 'Klasse3', 3: 'Klasse4', 4: 'Klasse5'}  # Update dit met je klassen
predicted_label = class_labels[predicted_class]

# Visualiseer de voorspelling
plt.imshow(new_image[0])
plt.title(f"Voorspelling: {predicted_label} (Waarschijnlijkheid: {pred[0][predicted_class]:.2f})")
plt.axis('off')
plt.show()
