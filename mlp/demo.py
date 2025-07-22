import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from matplotlib.colors import ListedColormap

train_data = 'data.txt'
test_data = 'experiment_data.txt'
output_history = 'demo_training_history.txt'
output_boundary = 'demo_decision_boundary_data.txt'
# Load the data (save the data as data.txt), convert the tag -1 to 0, and keep 1 as 1
data = np.loadtxt(train_data)
X = data[:, 1:3]  # Extract the Angle (Column 2) and velocity (column 3)
y = data[:, 0]     # Tag (Column 1)
y = np.where(y == 1, 1, 0)

# Divide the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.007)  # The learning rate is set at 0.007

# Compilation model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=128, #X_train_scaled.shape[0],
    validation_split=0.1,
    verbose=1
)

# Evaluation model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Accuracy: {test_acc:.4f}')

def save_training_history(history, filename = output_history):
    """Save the metric data during the training process to a text file"""
    # Extract historical data and convert it into a numpy array
    epochs = np.arange(1, len(history.history['loss']) + 1)
    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])

    # Merge into a two-dimensional array (each row corresponds to one epoch)
    history_data = np.column_stack((epochs, train_loss, val_loss, train_acc, val_acc))

    # Set the file header
    header = "Epoch | Train_Loss | Val_Loss | Train_Accuracy | Val_Accuracy"

    # Save as a formatted text file
    np.savetxt(filename,
               history_data,
               fmt='%4d  %.6f    %.6f     %.6f        %.6f',
               header=header,
               comments='')

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy subgraph
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])  # Fixing the Y-axis range facilitates observation
    plt.legend()

    # Loss subgraph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Visualized decision boundary
def plot_decision_boundary():
    # Generate grid points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Standardize grid data and make predictions
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    binary_Z = np.where(Z.ravel() >= 0.5, 1, 0)
    data_to_save = np.column_stack((xx.ravel(), yy.ravel(), binary_Z))
    np.savetxt(output_boundary,
               data_to_save,
               fmt='%.6f %.6f %d',
               header='x_coordinate y_coordinate prediction_probability',  # 列标题
               comments='')

    # Color mapping: Failure = red, success = green
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.4, colors=['red', 'green'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'green']), edgecolors='k')
    plt.scatter(new_data[:, 0], new_data[:, 1], c='purple', marker='x', s=100, label='预测点')
    plt.xlabel('Angle')
    plt.ylabel('Speed')
    plt.title('Decision Boundary (Red: Failure, Green: Success)')
    plt.show()

new_data = np.loadtxt(test_data, usecols=(1, 2))

# Standardized processing
new_data_scaled = scaler.transform(new_data)

# Batch prediction
predictions = model.predict(new_data_scaled)
pred_labels = np.where(predictions >= 0.5, 1, 0).flatten()

# Output the prediction result
print("\n=== Prediction Results ===")
for i, (features, prob, label) in enumerate(zip(new_data, predictions, pred_labels)):
    status = "Success" if label == 1 else "Failure"
    print(f"Sample {i+1}:")
    print(f"  Angle: {features[0]:.2f}°, Speed: {features[1]:.2f}m/s")
    print(f"  Success Probability: {prob[0]:.2%}")
    print(f"  Prediction: {status}\n")


plot_decision_boundary()
plot_training_history(history)
save_training_history(history)