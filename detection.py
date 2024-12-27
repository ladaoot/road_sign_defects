import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Параметры
images_folder = 'D:\\archive'  # Укажите путь к папке с изображениями
annotations_file = "D:\\archive\\full-gt.csv.csv"  # Укажите путь к CSV файлу с аннотациями


# Загрузка данных из CSV
def load_data(csv_path):
    return pd.read_csv(csv_path)


data = load_data(annotations_file)

# Разделение данных на тренировочный и тестовый наборы
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)


# Функция для загрузки изображений и аннотаций
def load_images_and_annotations(data):
    images = []
    boxes = []
    for index, row in data.iterrows():
        img_path = os.path.join(images_folder, row['filename'])
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))  # Измените размер в соответствии с моделью
        images.append(image)

        # Координаты bounding box
        boxes.append([int(row['x_from']), int(row['y_from']), int(row['x_from'])-int(row['width']), int(row['y_from'])-int(row['height'])])

    return np.array(images), np.array(boxes)


X_train, y_boxes_train = load_images_and_annotations(train_data)
X_test, y_boxes_test = load_images_and_annotations(test_data)

# Создание модели
input_layer = layers.Input(shape=(224, 224, 3))

# Сеть для предсказания координат bounding box
x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
box_output = layers.Dense(4, name='box_output')(x)  # x_min, y_min, x_max, y_max

model = models.Model(inputs=input_layer, outputs=box_output)

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Обучение модели
history = model.fit(X_train,
                    y_boxes_train,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=32)

# Оценка модели на тестовом наборе
test_loss = model.evaluate(X_test, y_boxes_test)
print(f'Test loss: {test_loss}')

# Сохранение обученной модели
model.save('traffic_sign_detection_model.h5')


# Функция для вычисления IoU
def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


# Функция для вычисления среднего IoU
def mean_iou(y_true, y_pred):
    ious = []
    for true_box, pred_box in zip(y_true, y_pred):
        ious.append(calculate_iou(true_box, pred_box))
    return np.mean(ious)


# Функция для вычисления точности и полноты
def calculate_precision_recall(y_true, y_pred, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    for true_box in y_true:
        found_match = False
        for pred_box in y_pred:
            if calculate_iou(true_box, pred_box) >= iou_threshold:
                found_match = True
                TP += 1
                break
        if not found_match:
            FN += 1

    FP = len(y_pred) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


# Получение предсказаний модели на тестовом наборе
predictions = model.predict(X_test)

# Вычисление метрик
miou = mean_iou(y_boxes_test, predictions)
precision, recall = calculate_precision_recall(y_boxes_test, predictions)

# Вывод результатов
print(f'Mean IoU: {miou}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Вычисление потерь для регрессии bounding boxes
mse_loss = mean_squared_error(y_boxes_test, predictions)
print(f'Mean Squared Error Loss: {mse_loss}')
