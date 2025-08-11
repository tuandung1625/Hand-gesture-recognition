
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model("Model/keras_model_updated.h5")

# Create a data generator for test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'Dataset/test', 
    target_size=(224, 224),   # Must match input size of your model
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Get predicted probabilities
pred_probs = model.predict(test_generator)

# Get predicted class indices
predictions = np.argmax(pred_probs, axis=1)

# True labels
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
print(classification_report(true_labels, predictions, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, xticklabels=class_labels, yticklabels=class_labels, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()