import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.cluster import DBSCAN

data = []

files_dir = "D:\\rtsd-frames"
reference_images_folder = "D:\\ref_sign"
crop_model_path = 'C:\\Users\ladac\PycharmProjects\PythonProject1\\traffic_sign_detection_model.h5'  # Путь к модели обрезки
classify_model_path = "D:\\trained_model_1.keras"  # Путь к модели классификации

def load_models(crop_model_path, classify_model_path):
    crop_model = load_model(crop_model_path)  # Модель для обрезки изображений
    classify_model = load_model(classify_model_path)  # Модель для классификации знаков
    return crop_model, classify_model

# Загрузка моделей
crop_model, classify_model = load_models(crop_model_path, classify_model_path)

train_dir = "D:\RoadSigns\\trainRoadSign"
test_dir = "D:\RoadSigns\\testRoadSign"

# Размеры изображений
img_width, img_height = 224, 224

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
                    # img = img / 255.0  # Нормализация
                    data.append(img)
                    labels.append(label)

# Загрузка данных из обучающей и тестовой папок
load_images(train_dir)
load_images(test_dir)

# Загрузка эталонных изображений
reference_images = {}
for filename in os.listdir(reference_images_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(reference_images_folder, filename)
        reference_image = cv2.imread(img_path)
        reference_images[filename] = reference_image


# Функция для обрезки изображения с использованием модели
def crop_image(model, images):
    data = []
    # Предобработка изображения (например, изменение размера)
    for image in images:
        img_resized = cv2.resize(image, (224, 224))  # Предполагаем, что модель принимает 224x224 изображения
        # img_array = np.expand_dims(img_resized, axis=0) / 255.0  # Нормализация
        data.append(img_resized)

    # Получение предсказания от модели
    predictions = model.predict(data)

    cropped_images =[]
    for i in range(len(predictions)):
        x1, y1, x2, y2 = predictions[i][0]  # Измените в зависимости от формата выхода вашей модели
        # Обрезка изображения
        cropped_images.append(data[i][int(y1):int(y2), int(x1):int(x2)])

    return cropped_images


# Функция для классификации знака
def classify_sign(model, images):
    data =[]
    for image in images:
        img_resized = cv2.resize(image, (64, 64))  # Измените размер в зависимости от вашей модели
        data.append(np.expand_dims(img_resized, axis=0) / 255.0 ) # Нормализация

    predictions = model.predict(data)
    class_indexes = np.argmax(predictions, axis=1)
    return class_indexes


# Функция для проверки на дефекты с использованием SSIM
def is_defective(current_image, reference_image):
    current_image_resized = cv2.resize(current_image, (reference_image.shape[1], reference_image.shape[0]))
    current_gray = cv2.cvtColor(current_image_resized, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(reference_gray, current_gray, full=True)
    threshold = 0.9
    return score < threshold

def cluster_defects(defect_signs):
    # Кластеризация дефектов
    if not defect_signs:
        return []

    distances = np.array([distance for _, _, distance in defect_signs]).reshape(-1, 1)

    # Кластеризация с использованием DBSCAN
    clustering = DBSCAN(eps=50, min_samples=2).fit(distances)

    grouped_defects = {}
    for idx, (filename, label_name, distance) in enumerate(defect_signs):
        cluster_id = clustering.labels_[idx]
        if cluster_id not in grouped_defects:
            grouped_defects[cluster_id] = []
        grouped_defects[cluster_id].append((filename, label_name, distance))

    return grouped_defects

# Детектим знаки
cropped_image = crop_image(crop_model, data)
cropped_image = [i for i in cropped_image if i is not None and i.size>0]

classes = classify_sign(classify_model,cropped_image)

# Обработка обнаруженных знаков
defective_images = []
for i in range(len(cropped_image)):
    reference_image = reference_images.get(classes[i])
    if reference_image is not None:
        if is_defective(cropped_image[i], reference_image):
            defective_images.append(cropped_image[i])

grouped_defects = cluster_defects(defective_images)

for cluster_id, defects in grouped_defects.items():
    print(f"Группа дефектов {cluster_id}:")

