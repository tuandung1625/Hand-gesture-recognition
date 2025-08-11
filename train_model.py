
# train_model.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained model
model = load_model('Model/keras_model.h5')

# (Optional) Compile again
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'Dataset/train', target_size=(200, 200), class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    'Dataset/test', target_size=(200, 200), class_mode='categorical'
)

# Train
model.fit(train_data, validation_data=test_data, epochs=10)

# (Optional) Save updated model
model.save('Model/keras_model_updated.h5')