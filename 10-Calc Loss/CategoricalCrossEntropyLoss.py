import math
import numpy as np
# Categorical Cross-Entropy Loss
# Loss function is a function that determine how wrong our model is
# And categorical cross-entropy is used to compare a "ground-truth" probability (y or "targets")
# and some predicted distirbuition (y-hat or "predictions")

# Formula
# Li = -sum(Y,i,j*log(Y-hat,i,j))
# Y,i,j = target value
# Y-hat, i, j = predicted value

# Further we will simplify to -log(correct_class_confidence) the formula is
# L,i = -Log(Y-hat,i,k)

# Code

softmax_output = [0.7, 0.1, 0.2]
# One-hot target vectors
targets = [1, 0, 0]

Li = -(math.log(softmax_output[0])*targets[0] +
       math.log(softmax_output[1])*targets[1] +
       math.log(softmax_output[2])*targets[2])
print("Not siplified")
print(Li)

# If you notice, targets[1] and targets[2] is 0, and a number times a 0 es equals 0. So we can cut this part off
# and a number times 1 is equal the number, só we can simplify to
print("-" * 20)
Li2 = -math.log(softmax_output[0])
print("Simplified!")
print(Li2)
print("-" * 20)
# Loss values examples
# Loss values raise when aproaching 0 and decreases when aproaching 1
print(math.log(1))
print(math.log(.95))
print(math.log(.90))
print(math.log(.8))
print("-" * 20)
print(math.log(.2))
print(math.log(.1))
print(math.log(.05))
print(math.log(.01))

# Until now we use a single softmax output, but we will work with batches
# So we need to update our process

# Target vectors
# [0, 1, 1]
# [2, 0, 1]
# It will be the position of the desired output in each sample of my softmax_output

# Imagine we have three classes, dog[0], cat[1], human[2] and we receive a target vector liked this [0, 1, 1]
# We are saying thata on our firsth sample, the class of index 0 is our target, seccond sample index 1 is the
# second target and the third sample is also the index 1

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]
target_values = [0, 1, 1]
# First sample, target value is 0.7, second is 0.5 and third is 0.9
# So if we want to access the confidence score of the target values we can do like this
# Withou zip method
confidence_score = []
confidence_score2 = []
for sample in range(len(softmax_outputs)):
    confidence_score.append(softmax_outputs[sample][target_values[sample]])

print("Not using zip")
print(confidence_score)

# With zip method
for sample, target in zip(softmax_outputs, target_values):
    confidence_score2.append(sample[target])

print("Using zip")
print(confidence_score2)

# Using Numpy
softmax_outputs2 = np.array([[0.7, 0.1, 0.2],
                             [0.1, 0.5, 0.4],
                             [0.02, 0.9, 0.08]])
target_values2 = np.array([0, 1, 1])
target_values3 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0]])
class_targets = target_values2
correct_confidences = []
print("Using numpy")
print(softmax_outputs2[range(len(softmax_outputs2)), target_values2])

# Now with numpy we can calculate the losses of our batch of softmax_outputs2
print("Loss values")
array_of_categorical_cross_entropy_loss = -np.log(softmax_outputs2[range(len(softmax_outputs2)), target_values2])
# But we not finished our loss calculation, now we need to calculate the average loss per batch
print("Average Loss per batch")
average_loss = np.mean(array_of_categorical_cross_entropy_loss)
print(average_loss)
print()
neg_log = -np.log(np.sum(softmax_outputs2 * target_values3, axis=1))
print(np.mean(neg_log))

# Now we have to adapt our code to accept either the one hot encode target values or the sparse values
# If its a one hot encoded array its a list of list and its shapes is 2, if its sparse then the shape is one
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs2[range(len(softmax_outputs2)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs2 * class_targets, axis= 1)
neg_log2 = -np.log(correct_confidences)
average_loss = np.mean(neg_log2)
print(average_loss)

# But there is a problem with that. Imagine if our model is 100% sure about a prediction, it means that the others
# class will be 0, and log of 0 (ln(0)) is undefined and log of 1 is a negative number. So we have to clip the max value to not be 1 and the min value no to be 0
# It is aways be closer to 1 or closer to 0
softmax_outputs3 = np.array([[0.7, 0.1, 0.2],
                             [0.1, 0.0, 0.9],
                             [0.0, 1.0, 0.0]])
target_values4 = np.array([0, 1, 1])
correct_confidences2 = softmax_outputs3[range(len(softmax_outputs3)), target_values4] 
cliped_correct_confidences2 = np.clip(correct_confidences2, 1e-7, 1-1e-7)
print(cliped_correct_confidences2)
