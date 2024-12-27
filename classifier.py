import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd

# Пути к папкам с изображениями
train_dir = "D:\RoadSigns\\trainRoadSign"
test_dir = "D:\RoadSigns\\testRoadSign"

# Размеры изображений
img_width, img_height = 64, 64

# Список для хранения данных и меток
data = []
labels = []

# Функция для загрузки и предобработки изображений
def load_images(directory):
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_width, img_height))
                    img = img / 255.0  # Нормализация
                    data.append(img)
                    labels.append(label)

# Загрузка данных из обучающей и тестовой папок
load_images(train_dir)
load_images(test_dir)

test_dir_1 = "D:\\rtsd-r1\\test"
train_dir_1 = "D:\\rtsd-r1\\train"

labels_path = "D:\\rtsd-r1\\numbers_to_classes.csv"
test_lables_path = "D:\\rtsd-r1\gt_test.csv"
train_lables_path = "D:\\rtsd-r1\gt_train.csv"

df = pd.read_csv(labels_path)
df1 = pd.read_csv(test_lables_path)
df2 = pd.read_csv(train_lables_path)

dff = df.merge(df1,how='left',left_on='class_number',right_on='class_number')
images_names = dff['filename']
labels_names = dff['sign_class']

for i in range(len(images_names)):
    image_path = f"{test_dir_1}/{images_names[i]}"
    img = cv2.imread(image_path)
    if img is None:
        print(f'Could not read image from {image_path}')
        continue

    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0  # Нормализация
    data.append(img)
    labels.append(labels_names[i])

dff = df.merge(df2,how='left',left_on='class_number',right_on='class_number')
images_names = dff['filename']
labels_names = dff['sign_class']
for i in range(len(images_names)):
    image_path = f"{train_dir_1}/{images_names[i]}"
    img = cv2.imread(image_path)
    if img is None:
        print(f'Could not read image from {image_path}')
        continue
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0  # Нормализация
    data.append(img)
    labels.append(labels_names[i])

print('load')
# Преобразование данных в массивы NumPy
data = np.array(data)
labels = np.array(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(labels)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.3, random_state=42)

# Создание модели CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(len(np.unique(y_encoded)), activation='relu'))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Вычисление точности
accuracy = accuracy_score(y_test, y_pred_classes)
print("Точность:", accuracy)

model.save("D:\\trained_model_1.keras")

with open('D:\\lable_1.pkl','wb') as file:
    pickle.dump(le,file)