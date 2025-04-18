# Import NumPy, the fundamental package for fast vector and matrix operations
import numpy as np   

# -----------------------------------------------------------------------------
# 1) BUILD THE COMPLETE DATASET OF 4‑PIXEL IMAGES
# -----------------------------------------------------------------------------

# Create a list of all integers 0–15, format each as 4‑bit binary (e.g. 5 → "0101"),
# convert each character ('0' or '1') to an int, and wrap the result in a NumPy array.
# Each row now represents one image of four pixels.
X_pixels = np.array(
    [[int(bit) for bit in f"{i:04b}"] for i in range(16)],
    dtype=int
)

# Count the number of white pixels (1 = bright) in each image.
# Label is 1 (bright) if that count is ≥ 2, else 0 (dark).
y_pixels = (X_pixels.sum(axis=1) >= 2).astype(int)

# Print each image with its label so we can visually verify the rule
print("All images and their labels (1 = bright, 0 = dark):")
for img, label in zip(X_pixels, y_pixels):
    print(img, "→", label)
print()   # blank line for readability

# -----------------------------------------------------------------------------
# 2) ADD A BIAS INPUT TO EVERY IMAGE
# -----------------------------------------------------------------------------

# Create a column vector of ones; this is the “always‑on” bias input.
bias_column = np.ones((X_pixels.shape[0], 1), dtype=int)

# Concatenate the bias column to the left of the pixel data so each sample now
# has five inputs: [bias, pixel1, pixel2, pixel3, pixel4].
X = np.hstack([bias_column, X_pixels])

# -----------------------------------------------------------------------------
# 3) INITIALISE THE WEIGHT VECTOR RANDOMLY
# -----------------------------------------------------------------------------

# Use NumPy’s random number generator with a fixed seed for reproducibility.
rng = np.random.default_rng(seed=0)

# Draw one weight per input (5 total) from the range (−0.5, 0.5).
w = rng.uniform(-0.5, 0.5, X.shape[1])

# Show the starting weights before learning begins
print("Initial weights (bias first):", np.round(w, 3), "\n")

# -----------------------------------------------------------------------------
# 4) TRAIN THE PERCEPTRON WITH THE CLASSIC UPDATE RULE
# -----------------------------------------------------------------------------

# Set the learning rate (η) – how much each mistake changes the weights
eta = 0.2

# Number of passes (epochs) over the entire training set
epochs = 25

# Loop over epochs
for epoch in range(1, epochs + 1):

    # Compute current predictions before any updates this epoch
    current_preds = (X @ w >= 0).astype(int)

    # Calculate accuracy at the start of the epoch
    start_acc = (current_preds == y_pixels).mean()

    # Print epoch number and starting accuracy
    print(f"Epoch {epoch:02d} – starting accuracy {start_acc:.2f}")

    # Loop over each training sample (input xi and its target label)
    for i, (xi, target) in enumerate(zip(X, y_pixels)):

        # Calculate the neuron's raw output (z) – dot product of inputs and weights
        z = np.dot(xi, w)

        # Apply the step activation: output 1 if z ≥ 0 else 0
        pred = 1 if z >= 0 else 0

        # Compute the weight change: η * (target − prediction) * inputs
        update = eta * (target - pred) * xi

        # For the first two samples of the first three epochs, print the update so
        # you can see how the weights get nudged toward the correct answer
        if i < 2 and epoch <= 3:
            print("  sample", xi[1:], "target", target,
                  "prediction", pred, "update", update)

        # Apply the weight update (in‑place modification of w)
        w += update

# -----------------------------------------------------------------------------
# 5) FINAL EVALUATION ON ALL 16 IMAGES
# -----------------------------------------------------------------------------

# Generate final predictions using the trained weights
predictions = (X @ w >= 0).astype(int)

# Compute overall accuracy (fraction of correct predictions)
accuracy = (predictions == y_pixels).mean()

# Print learned weights and final accuracy
print("\nFinal learned weights (bias first):", np.round(w, 3))
print("Final accuracy on the full dataset:", accuracy)
