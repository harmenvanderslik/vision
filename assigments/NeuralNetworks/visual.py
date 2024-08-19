import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('mijn_model.h5')

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # h, b, kanalen
    img_tensor = np.expand_dims(img_tensor, axis=0)         # 1, h, b, kanalen
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


img_path = 'dataset/test/tulip/4571993204_5b3efe0e78.jpg'
new_image = load_image(img_path, show=True)
pred = model.predict(new_image)
predicted_class = np.argmax(pred)
class_labels = {0: 'madelief', 1: 'paardebloem', 2: 'roos', 3: 'zonnebloem', 4: 'tulp'}
predicted_label = class_labels[predicted_class]


plt.imshow(new_image[0])
plt.title(f"Voorspelling: {predicted_label} (Waarschijnlijkheid: {pred[0][predicted_class]:.2f})")
plt.axis('off')
plt.show()
