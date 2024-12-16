import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error

# Generate simulated data for ML model
np.random.seed(42)
# Simulated features: 1000 samples, representing features like time in zone, motion, etc.
X = np.random.rand(1000, 5)  # Assume 5 features for tracking metrics
# Simulated labels: time spent in checkout zone (in seconds)
y = np.random.rand(1000) * 60  # Random times between 0 and 60 seconds

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and build the ML model
model = Sequential([
    Dense(64, input_dim=5, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")  # Regression output
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error on test data: {test_mae:.2f} seconds")

# Example usage with video processing
def process_video_with_model(video_path, model, scaler):
    cap = cv2.VideoCapture(video_path)
    customer_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features (this would need actual feature extraction logic from the video)
        # Here we simulate random features for demonstration purposes.
        features = np.random.rand(1, 5)
        features_scaled = scaler.transform(features)

        # Predict service time using the trained model
        predicted_time = model.predict(features_scaled)[0][0]
        customer_times.append(predicted_time)

        # Display frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print the average service time
    if customer_times:
        avg_time = sum(customer_times) / len(customer_times)
        print(f"Predicted Average Service Time: {avg_time:.2f} seconds")
    else:
        print("No customers detected.")

if __name__ == "__main__":
    video_path = r'F:\IT\fringe\avg_time\check.mp4'  # Replace with the actual video path
    process_video_with_model(video_path, model, scaler)
