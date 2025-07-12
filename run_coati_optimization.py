
import numpy as np
from coati_optimization import coati_optimization

# Placeholder for the actual CNN training function
def build_and_train_cnn(params):
    num_filters = int(params[0])
    learning_rate = 10 ** params[1]
    batch_size = int(params[2])

    # Here you would build and train your CNN with the above hyperparameters
    # and return the validation accuracy. This is just a dummy example:
    accuracy = np.random.uniform(0.7, 0.99)  # Replace with real training & evaluation
    return accuracy

# Define parameter bounds: [num_filters, log10(learning_rate), batch_size]
bounds = [
    (14, 62),       # number of filters
    (-4, -2),       # log10 learning rate (10^-4 to 10^-2)
    (34, 130)       # batch size
]

best_params, best_score = coati_optimization(build_and_train_cnn, bounds)
print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)
