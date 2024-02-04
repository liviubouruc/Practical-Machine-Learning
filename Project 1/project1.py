import os
import cv2
import PIL
import numpy as np # linear algebra
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix


input_path = '/kaggle/input/cards-image-datasetclassification'
output_path = '/kaggle/working'

# accuracy function from laboratory
def accuracy_score(ground_truth_labels, predicted_labels):
    return np.mean(ground_truth_labels == predicted_labels)

# reading the images with custom function
def load_images_from_folder(folder):
    images = []
    labels = []
    label = 0
    for card_type in os.listdir(folder):
        path = folder + "/" + card_type
        for card in os.listdir(path):
            img = cv2.imread(path + "/" + card)
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(label)
        label += 1 # label each category with a number
    return images, labels

train_images, train_labels = load_images_from_folder(input_path+'/train')
val_images, val_labels = load_images_from_folder(input_path+'/valid')
test_images, test_labels = load_images_from_folder(input_path+'/test')

print(train_images[0])
print(len(train_labels))


# Naive Bayes
gnb = GaussianNB()
gnb.fit(train_images, train_labels)
predicted = gnb.predict(val_images)
acc = accuracy_score(val_labels, predicted)
print(acc)

test_predicted = gnb.predict(test_images)
test_acc = accuracy_score(test_labels, test_predicted)
print(test_acc)

cm = confusion_matrix(test_labels, test_predicted, labels=gnb.classes_)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
disp.plot()
plt.show()


# Random Forest
rf_param_grid = { 
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier()
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid)
rf_grid.fit(train_images, train_labels)
print(rf_grid.best_params_)

rf_classifier = RandomForestClassifier(n_estimators=200, max_features='log2')
rf_classifier.fit(train_images, train_labels)
predicted = rf_classifier.predict(val_images)
acc = accuracy_score(val_labels, predicted)
print(acc)

test_predicted = rf_classifier.predict(test_images)
test_acc = accuracy_score(test_labels, test_predicted)
print(test_acc)

cm = confusion_matrix(test_labels, test_predicted, labels=rf_classifier.classes_)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_classifier.classes_)
disp.plot()
plt.show()


# SVM
svm_param_grid = {'C': [1, 3.5, 10], 
                  'kernel': ['poly', 'rbf']} 
svm_grid = GridSearchCV(svm.SVC(), svm_param_grid)
svm_grid.fit(train_images, train_labels)
print(rf_grid.best_params_)

svm_classifier = svm.SVC(C=10)
svm_classifier.fit(train_images, train_labels)
predicted = svm_classifier.predict(val_images)
acc = accuracy_score(val_labels, predicted)
print(acc)

test_predict = svm_classifier.predict(test_images)
test_acc = accuracy_score(test_labels, test_predicted)
print(acc)

cm = confusion_matrix(test_labels, test_predicted, labels=svm_classifier.classes_)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_classifier.classes_)
disp.plot()
plt.show()


# With HSV features
def load_images_with_hsv_features_from_folder(folder):
    images = []
    labels = []
    label = 0
    for card_type in os.listdir(folder):
        path = folder + "/" + card_type
        for card in os.listdir(path):
            img = PIL.Image.open(path + "/" + card)
            img = img.resize((64, 64))
            img = img.convert('HSV')
            flatten_img = np.mean(np.array(img), axis=2).flatten()
            images.append(flatten_img)
            labels.append(label)
        label += 1
    return images, labels

hsv_train_images, train_labels = load_images_with_hsv_features_from_folder(input_path+'/train') 
hsv_val_images, val_labels = load_images_with_hsv_features_from_folder(input_path+'/valid')
hsv_test_images, test_labels = load_images_with_hsv_features_from_folder(input_path+'/test')

gnb = GaussianNB()
gnb.fit(hsv_train_images, train_labels)
predicted = gnb.predict(hsv_val_images)
acc = accuracy_score(val_labels, predicted)
print(acc)

rf_classifier = RandomForestClassifier(n_estimators=100, max_features='log2')
rf_classifier.fit(hsv_train_images, train_labels)
predicted = rf_classifier.predict(hsv_val_images)
acc = accuracy_score(val_labels, predicted)
print(acc)

svm_classifier = svm.SVC(C=1, kernel='poly')
svm_classifier.fit(hsv_train_images, train_labels)
predicted = svm_classifier.predict(hsv_val_images)
acc = accuracy_score(val_labels, predicted)
print(acc)


# CNN
train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=input_path+'/train')
val_data = tf.keras.preprocessing.image_dataset_from_directory(directory=input_path+'/valid')
test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=input_path+'/test')

cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(53, activation='softmax')
])

opt = keras.optimizers.Adam(learning_rate=1e-3) 
cnn_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_data, epochs=20, batch_size=64, validation_data=val_data)
print(cnn_model.evaluate(test_data))