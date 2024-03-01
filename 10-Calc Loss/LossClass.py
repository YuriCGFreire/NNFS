# Here you will find both, loss class and categorical cross entropy class
import numpy as np;
class Loss:
    def claculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_cliped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_cliped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_cliped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                             [0.1, 0.5, 0.4],
                             [0.02, 0.9, 0.08]])
target_values1 = np.array([0, 1, 1])
target_values2 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0]])
class_targets = target_values2

loss_function = Categorical_Cross_Entropy()
loss = loss_function.claculate(softmax_outputs, class_targets)
print(loss)