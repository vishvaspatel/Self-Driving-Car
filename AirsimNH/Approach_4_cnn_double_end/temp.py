import matplotlib.pyplot as plt
import numpy as np

# Data from First Training Run (00:02:50 - 00:21:45)
epochs_run1 = np.arange(1, 51)
train_loss_run1 = [
    0.0515, 0.0424, 0.0392, 0.0368, 0.0346, 0.0333, 0.0319, 0.0309, 0.0304, 0.0304,
    0.0316, 0.0302, 0.0282, 0.0269, 0.0258, 0.0241, 0.0226, 0.0214, 0.0197, 0.0184,
    0.0174, 0.0165, 0.0158, 0.0148, 0.0141, 0.0139, 0.0134, 0.0131, 0.0131, 0.0132,
    0.0152, 0.0147, 0.0135, 0.0124, 0.0116, 0.0106, 0.0098, 0.0090, 0.0086, 0.0077,
    0.0074, 0.0070, 0.0065, 0.0063, 0.0059, 0.0056, 0.0053, 0.0051, 0.0050, 0.0049
]
val_loss_run1 = [
    0.0377, 0.0370, 0.0374, 0.0337, 0.0334, 0.0315, 0.0336, 0.0318, 0.0319, 0.0314,
    0.0352, 0.0305, 0.0308, 0.0360, 0.0304, 0.0311, 0.0346, 0.0312, 0.0308, 0.0321,
    0.0338, 0.0308, 0.0331, 0.0322, 0.0319, 0.0320, 0.0317, 0.0319, 0.0319, 0.0315,
    0.0349, 0.0328, 0.0513, 0.0323, 0.0392, 0.0229, 0.0327, 0.0478, 0.0257, 0.0225,
    0.0127, 0.0187, 0.0092, 0.0098, 0.0099, 0.0010, 0.0089, 0.0078, 0.0076, 0.0065
]
# Smooth validation loss using a moving average (window size = 5)
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply moving average to validation loss
window_size = 5
smoothed_val_loss_run1 = moving_average(val_loss_run1, window_size)
# Adjust epochs to match the length of smoothed data
smoothed_epochs_run1 = epochs_run1[window_size-1:]

# Create figure
plt.figure(figsize=(7, 6))

# Plot First Training Run
plt.plot(epochs_run1, train_loss_run1, 'b-', label='Training Loss')
plt.plot(smoothed_epochs_run1, smoothed_val_loss_run1, 'r-', label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_loss_curve_smoothed_no_grid.png', dpi=300)
plt.show()