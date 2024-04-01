'''
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

original = io.imread('original_image.jpg')
original = original[:,:,0:3]  # remove alpha channel
n_colors = 4

arr = original.reshape((-1, 3))
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
less_colors = centers[labels].reshape(original.shape).astype('uint8')

plt.imshow(less_colors)
plt.axis('off')  # Turn off axes
plt.show()
'''

import numpy as np
from sklearn.cluster import KMeans
from skimage import io
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import time
from flask import Flask, request, render_template
import cv2
from flask import send_file
import skimage.measure

app = Flask(__name__)

def subsample_data(image):
    # 2:1 subsampling in horizontal and vertical directions
    sampled_image = skimage.measure.block_reduce(image, block_size=(2,2,1), func=np.mean)
    return sampled_image
    
def calculate_weights(image):

    print(image.shape)
    subsampled_image = subsample_data(image)
    print(subsampled_image.shape)

    # Convert image to one-dimensional array
    pixels = subsampled_image.reshape(-1, subsampled_image.shape[-1])

    #print(pixels.size)

    # Calculate histogram of pixel colors
    hist, _ = np.histogram(pixels, bins=pixels.size)

    # Flatten the histogram to a 1D array
    flattened_hist = hist.flatten()

    # Calculate weights proportional to frequency
    weights = flattened_hist / np.sum(flattened_hist)

    print(weights.shape)

    return weights

def assign_weights(image, possible_colors):
    # Convert image to one-dimensional array
    pixels = image.reshape(-1, image.shape[-1])
    
    # Initialize an empty array to store weights
    weights = np.zeros(pixels.shape[0])
    
    # Iterate over each pixel in the image
    for i, pixel in enumerate(pixels):
        # Map pixel's RGB value to index in possible_colors array
        color_index = pixel[0] * (256**2) + pixel[1] * 256 + pixel[2]
        
        # Retrieve the frequency associated with the color index
        if color_index < len(possible_colors):
            weights[i] = possible_colors[color_index]
    
    return weights


def initialize_centers(X, K):
    # Select random subset of X as initial cluster centers
    indices = np.random.choice(X.shape[0], K, replace=False)
    return X[indices]

def update_centers(X, weights, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        cluster_points = X[labels == k]
        weight_sum = np.sum(weights[labels == k])
        if weight_sum != 0:  # Check if sum of weights is not zero
            centers[k] = np.sum(weights[labels == k, None] * cluster_points, axis=0) / weight_sum
        else:
            # If all weights are zero, set center to the centroid of the cluster points
            centers[k] = np.mean(cluster_points, axis=0)
    return centers

def kmeans_weighted(X, weights, K, max_iter=100):
    # Initialize cluster centers
    centers = initialize_centers(X, K)
    labels = np.zeros(X.shape[0])

    # Normalize the weights
    weights /= X.shape[0]  # Dividing by the number of pixels
    
    for _ in range(max_iter):
        # Assign points to nearest cluster
        labels, _ = pairwise_distances_argmin_min(X, centers)
        
        # Update cluster centers
        new_centers = update_centers(X, weights, labels, K)
        
        # Check for convergence
        if np.any(np.isnan(new_centers)):  # Check if any center is NaN
            print("centroid has NAN")
            break

        if np.any(np.isnan(X)):  # Check if any center is NaN
            print("ERROR X")
            break
            
        centers = new_centers
    
    return centers, labels


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            original = io.imread(file)

            #original = original[:,:,0:3]  # remove alpha channel
            # Get number of colors from form input
            n_colors = int(request.form['colors'])
            
            # Apply color quantization
            original = original[:, :, :3]  # remove alpha channel
            arr = original.reshape((-1, 3))

            weights = calculate_weights(original)

            weights = assign_weights(image=original, possible_colors=weights)

            start_time = time.time()

            # Perform weighted k-means clustering
            cluster_centers, labels = kmeans_weighted(arr, weights, n_colors)

            # Convert cluster centers to uint8
            cluster_centers = cluster_centers.astype('uint8')

            # Assign each pixel to its nearest cluster center
            less_colors = cluster_centers[labels].reshape(original.shape).astype('uint8')

            # Stop the timer
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print("Elapsed Time:", elapsed_time, "seconds")

            # Save quantized image
            cv2.imwrite('quantized_image.jpg', cv2.cvtColor(less_colors, cv2.COLOR_RGB2BGR))

            # Render template with the image path
            return send_file('quantized_image.jpg', as_attachment=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

