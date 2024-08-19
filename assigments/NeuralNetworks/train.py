import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Stel het pad in naar je dataset
base_dir = 'dataset'

train_dir = f'{base_dir}/train'
test_dir = f'{base_dir}/test'

# Initialiseren van de ImageDataGenerator voor data augmentatie
train_datagen = ImageDataGenerator(
    rescale=1./255,   # Normaliseer de afbeeldingen
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Alleen normalisatie voor testdata

# Laad de trainingsdata
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Afhankelijk van de input size van je model
    batch_size=32,
    class_mode='categorical'  # Gebruik 'categorical' voor meerdere klassen
)

# Laad de testdata
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 klassen voor de bloemen
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=66,  # Aangepast naar het juiste aantal stappen per epoch
    epochs=20,
    validation_data=test_generator,
    validation_steps=19  # Aangepast naar het juiste aantal validatiestappen
)

model.save('mijn_model.h5')
