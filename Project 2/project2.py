import cv2 as cv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from collections import Counter

from sklearn.cluster import MeanShift, estimate_bandwidth


input_path = '/content/archive'


raw_data = pd.read_csv(input_path+'/english.csv')
raw_images = []
labels = []
for _, row_data in raw_data.iterrows():
  img = cv.imread(input_path + "/" + row_data['image'])
  img = cv.resize(img, (224, 224))
  img_array = np.expand_dims(img_to_array(img), axis = 0)
  raw_images.append(img_array)
  labels.append(row_data['label'])


# Extract features with pretrained VGG16
model = VGG16(weights='imagenet', include_top=False)
def extract_vgg_features(raw_images):
    images_features = []
    for img in raw_images:
        features = np.array(model.predict(preprocess_input(img)))
        images_features.append(features.flatten())       
    return images_features

images_features = extract_vgg_features(raw_images)
images_features = np.array(images_features)

# PCA
pca_images_features = PCA(n_components=512).fit_transform(images_features)
print(pca_images_features.shape)

train_pca_data, val_pca_data, train_labels, val_labels = train_test_split(pca_images_features, labels, test_size=0.2)


# Agglomerative
agg_clusters = AgglomerativeClustering(n_clusters = 62).fit(train_pca_data)
print(Counter(agg_clusters.labels_))

metrics = ["l1", "l2", "manhattan"]
linkages = ["complete", "average"]
s_scores = []
for metric in metrics:
    for linkage in linkages:
        agg_clusters = AgglomerativeClustering(n_clusters = 62, metric=metric,  linkage=linkage).fit(train_pca_data)
        print(metric, " ", linkage, Counter(agg_clusters.labels_), silhouette_score(train_pca_data, agg_clusters.labels_))
        s_scores.append(silhouette_score(train_pca_data, agg_clusters.labels_))


agg_clusters = AgglomerativeClustering(n_clusters = 62).fit(images_features)
print(sorted(Counter(agg_clusters.labels_).items()))

agg_clusters = AgglomerativeClustering(n_clusters = 62).fit(pca_images_features)
print(sorted(Counter(agg_clusters.labels_).items()))

from scipy.cluster.hierarchy import dendrogram, linkage
dendrogram(linkage(pca_images_features, method='complete'))


# Mean-Shift
estimated_bandwidth = estimate_bandwidth(train_pca_data)
print(estimated_bandwidth)

bandwidths = [1150, 1200, 1235, 1250, 1300]
s_scores = []
for bandwidth in bandwidths:
  ms_clusters = MeanShift(bandwidth = bandwidth)
  ms_clusters.fit(pca_images_features)
  s_scores.append(silhouette_score(pca_images_features, ms_clusters.labels_))

plt.plot(range(len(s_scores)), s_scores)



# Supervised method
train_data, val_data, train_labels, val_labels = train_test_split(pca_images_features, labels, test_size=0.2)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(train_data, train_labels)
predicted = rf_classifier.predict(val_data)

def accuracy_score(ground_truth_labels, predicted_labels):
    return np.mean(ground_truth_labels == predicted_labels)

acc = accuracy_score(val_labels, predicted)
print(acc)

cm = confusion_matrix(val_labels, predicted, labels=rf_classifier.classes_)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_classifier.classes_)
disp.plot()
plt.show()

# Random chance
import random
rand_values = [random.choice(val_labels) for i in range(len(val_labels))]
acc = accuracy_score(val_labels, predicted)
print(acc)