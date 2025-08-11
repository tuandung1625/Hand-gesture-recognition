<<<<<<< HEAD
import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def split_dataset(data_dir='Data', output_dir='Dataset', train_ratio=0.8):
    labels = os.listdir(data_dir)

    for label in labels:
        img_dir = os.path.join(data_dir, label)
        images = os.listdir(img_dir)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for split, img_list in [('train', train_imgs), ('test', test_imgs)]:
            split_folder = os.path.join(output_dir, split, label)
            os.makedirs(split_folder, exist_ok=True)

            for img in img_list:
                shutil.copy(os.path.join(img_dir, img), os.path.join(split_folder, img))

if __name__ == '__main__':
    split_dataset()
=======
import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def split_dataset(data_dir='Data', output_dir='Dataset', train_ratio=0.8):
    labels = os.listdir(data_dir)

    for label in labels:
        img_dir = os.path.join(data_dir, label)
        images = os.listdir(img_dir)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for split, img_list in [('train', train_imgs), ('test', test_imgs)]:
            split_folder = os.path.join(output_dir, split, label)
            os.makedirs(split_folder, exist_ok=True)

            for img in img_list:
                shutil.copy(os.path.join(img_dir, img), os.path.join(split_folder, img))

if __name__ == '__main__':
    split_dataset()
>>>>>>> 4c6101a (Save local files)
