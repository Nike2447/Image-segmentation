import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.cluster import KMeans

# Reading in the image
image = imread(os.path.join(r"C:\\Users\\NIKE2\\OneDrive\\Desktop","Image segmentation","nature.jpg"))

# The shape is in the format (hright,width,color_channel(3 for rgb))
# Reshaping image into giving a long list of arrays of rgb colors 
X = image.reshape((-1,3))

# Applying kmeans to the label X
# Number of clusters denotes the number of colors that will be used
k1 = KMeans(n_clusters=8).fit(X)
segmented_img1 = k1.cluster_centers_[k1.labels_]
segmented_img1 = segmented_img1.reshape(image.shape)

k2= KMeans(n_clusters=6).fit(X)
segmented_img2 = k2.cluster_centers_[k2.labels_]
segmented_img2 = segmented_img2.reshape(image.shape)


k3 = KMeans(n_clusters=4).fit(X)
segmented_img3 = k3.cluster_centers_[k3.labels_]
segmented_img3= segmented_img3.reshape(image.shape)


k4 = KMeans(n_clusters=2).fit(X)
segmented_img4 = k4.cluster_centers_[k4.labels_]
segmented_img4 = segmented_img4.reshape(image.shape)

k5 = KMeans(n_clusters=10).fit(X)
segmented_img5 = k5.cluster_centers_[k5.labels_]
segmented_img5 = segmented_img5.reshape(image.shape)



plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(2,3,2)
plt.title("10 Clusters")
plt.imshow(segmented_img5.astype(int))
plt.axis('off')


plt.subplot(2,3,3)
plt.title("8 Clusters")
plt.imshow(segmented_img1.astype(int))
plt.axis('off')

plt.subplot(2,3,4)
plt.title("6 Clusters")
plt.imshow(segmented_img2.astype(int))
plt.axis('off')

plt.subplot(2,3,5)
plt.title("4 Clusters")
plt.imshow(segmented_img3.astype(int))
plt.axis('off')

plt.subplot(2,3,6)
plt.title("2 Clusters")
plt.imshow(segmented_img4.astype(int))
plt.axis('off')

plt.show()