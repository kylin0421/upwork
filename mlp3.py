# Import necessary libraries
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import json
import matplotlib.pyplot as plt

# Custom callback for saving loss history and printing training data
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, loss_file='loss_history.json', model_reference=None, X_train=None, y_train=None):
        super(LossHistory, self).__init__()
        self.loss_file = loss_file
        self.losses = []
        self.epochs = []
        self.model_reference = model_reference
        self.X_train = X_train
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        # Save loss and epoch information after each epoch
        loss = logs.get('loss')
        accuracy = logs.get('accuracy')
        self.losses.append(loss)
        self.epochs.append(epoch)

        # Save loss history to a JSON file
        with open(self.loss_file, 'w') as f:
            json.dump({'epochs': self.epochs, 'losses': self.losses}, f)

        # Save model weights every 100 epochs
        if epoch % 100 == 0 and self.model_reference:
            self.model_reference.save_weights(f'./train_result/model_epoch_{epoch}.weights.h5')
            print(f"Model saved at epoch {epoch}")

            # Print sample training data and labels
            print("Sample training data:")
            print(self.X_train[:5])
            print("Sample training labels:")
            print(self.y_train[:5])


# Classification model for multi-class classification
class ClassificationModel:
    def __init__(self, csv_file, batch_size=100, model_path='./train_result/model_checkpoint.weights.h5',
                 loss_file='./train_result/loss_history.json'):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.model_path = model_path
        self.loss_file = loss_file

        # Load dataset from CSV file
        self.data = pd.read_csv(self.csv_file).values

        # Split dataset into features and labels (last 4 columns are labels)
        self.X = self.data[:, :-4]
        self.y = self.data[:, -4:]

        # Normalize selected columns (assumed columns 1-6)
        self.X[:, 1:7] = MinMaxScaler().fit_transform(self.X[:, 1:7])

        # Split data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2,
                                                                              random_state=42)

        # Build the neural network model
        self.model = self.build_model()

        # Initialize the loss history callback
        self.loss_history = LossHistory(loss_file=self.loss_file, model_reference=self.model,
                                        X_train=self.X_train, y_train=self.y_train)

    def build_model(self):
        # Build a deep neural network with multiple dense layers
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(self.X.shape[1],)),
            tf.keras.layers.Dense(512, activation='relu'),  # Hidden layer 1
            tf.keras.layers.Dense(256, activation='relu'),  # Hidden layer 2
            tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 3
            tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer 4
            tf.keras.layers.Dense(32, activation='relu'),   # Hidden layer 5
            tf.keras.layers.Dense(4, activation='softmax')  # Output layer for 4 classes
        ])

        # Compile the model with Adam optimizer and categorical cross-entropy loss
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, epochs=100, save_interval=1000):
        # Define model checkpoint callback for periodic saving
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                                                 save_weights_only=True,
                                                                 save_freq=save_interval)

        # Train the model with training and validation data
        self.model.fit(self.X_train, self.y_train,
                       epochs=epochs,
                       batch_size=self.batch_size,
                       validation_data=(self.X_val, self.y_val),
                       callbacks=[checkpoint_callback, self.loss_history])

    def load_model(self):
        # Load pre-trained model weights if available
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            print("Model loaded successfully.")
        else:
            print("Model path does not exist. Starting with a new model.")

    def save_model(self):
        # Save the trained model weights
        self.model.save_weights(self.model_path)
        print("Model saved successfully.")

    def plot_loss(self):
        # Plot training loss curve from loss history file
        if os.path.exists(self.loss_file):
            with open(self.loss_file, 'r') as f:
                history = json.load(f)

            epochs = history['epochs']
            losses = history['losses']

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, label='Training Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print(f"Loss file {self.loss_file} does not exist.")

    def evaluate_accuracy(self, test_csv):
        # Load test dataset
        test_data = pd.read_csv(test_csv).values

        # Retain original test data before normalization
        original_test_data = test_data[:, :-4].copy()

        X_test = test_data[:, :-4]
        y_test = test_data[:, -4:]

        # Normalize selected columns in test data
        X_test[:, 1:7] = MinMaxScaler().fit_transform(X_test[:, 1:7])

        # Evaluate model accuracy on test data
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Predict test data
        y_pred = self.model.predict(X_test)

        # Convert predictions to one-hot format
        y_pred_one_hot = np.zeros_like(y_pred)
        y_pred_one_hot[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

        # Save prediction results along with original features
        result_data = np.hstack((original_test_data, y_test, y_pred_one_hot))
        result_df = pd.DataFrame(result_data,
                                 columns=[*['OriginalFeature' + str(i) for i in range(original_test_data.shape[1])],
                                          'True1', 'True2', 'True3', 'True4', 'Pred1', 'Pred2', 'Pred3', 'Pred4'])
        result_df.to_csv('./train_result/predicted_results_original_data.csv', index=False)
        print("Predicted results saved to './train_result/predicted_results_original_data.csv'.")

    def convert_to_tflite(self):
        # Convert the Keras model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()

        tflite_model_path = os.path.join('./train_result/', 'model.tflite')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        print(f"TensorFlow Lite model saved to: {tflite_model_path}")


# Instantiate the model and run tasks
csv_file = './rawdata/train.csv'
model = ClassificationModel(csv_file=csv_file)
model.load_model()
model.convert_to_tflite()
#model.train(epochs=50, save_interval=10)
#model.save_model()
model.plot_loss()
test_csv_file = './rawdata/testing.csv'
model.evaluate_accuracy(test_csv=test_csv_file)
