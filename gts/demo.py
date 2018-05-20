#coding = utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import random       #随机
from nnet import*  
from load import Load 

path = "F:/gts/gtsdate"
load = Load(path)

# Pick 10 random images
sample_indexes = random.sample(range(len(load.images32)), 10) 
sample_images = [load.images32[i] for i in sample_indexes]
sample_labels = [load.labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()