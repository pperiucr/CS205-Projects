#Homework 2 code to test the plot results of Dendrogram
# Python imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Selected colors from the list
selectedColors = {
    'Black': [0, 0, 0],
    'DarkSlateGrey': [47, 79, 79],
    'Gold': [255, 215, 0],
    'SandyBrown': [244, 164, 96],
    'Teal': [0, 128, 128]
}

#Create list of selected colors
colorLabels = list(selectedColors.keys())
#Converted to array
rgbValues = np.array(list(selectedColors.values()))
#Link with theier euclidean distance and average
linked = linkage(rgbValues, method='average', metric='euclidean')
#Plot the figure using matplot lib
plt.figure(figsize=(8, 5))
# Call dendrogram plot function with the data
dendrogram(linked, labels=colorLabels, distance_sort='ascending')
plt.title("HW2 - Dendrogram of RGB Colors")
plt.xlabel("HW2 Colors")
plt.ylabel("Euclidean Distance")
plt.show()
