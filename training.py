import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize components
lemmatizer = WordNetLemmatizer()

# Load dataset
intents = json.loads(open("data.json").read())

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Extract patterns and classes, including sub-tags
for intent in intents["intents"]:
    if "subtags" in intent:
        # Process sub-tags
        for subtag in intent["subtags"]:
            for pattern in subtag["patterns"]:
                word_list = word_tokenize(pattern)
                words.extend(word_list)
                documents.append((word_list, subtag["tag"]))
                if subtag["tag"] not in classes:
                    classes.append(subtag["tag"])
    else:
        # Process main tag if no sub-tags
        for pattern in intent["patterns"]:
            word_list = word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes for consistent ordering
classes = sorted(set(classes))

# Save words and classes to disk for later use
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag-of-words
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    # Create one-hot encoded output
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Ensure consistency in bag-of-words and output length
for item in training:
    assert len(item[0]) == len(words), "Inconsistent bag-of-words length"
    assert len(item[1]) == len(classes), "Inconsistent output class length"

random.shuffle(training)

# Convert to numpy arrays
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile the model with SGD optimizer
# Adjust training epochs, learning rate, and other hyperparameters to prevent underfitting
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(
    train_x,
    train_y,
    epochs=250,  # Increase epochs for better training
    batch_size=10,  # Adjust batch size for stable training
    verbose=1
)


# Save the trained model
model.save("chatbot_model.model.keras", hist)
print("Training complete. Model saved.")

hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
print("Final training loss:", hist.history['loss'][-1])
print("Final training accuracy:", hist.history['accuracy'][-1])

# Visualize loss and accuracy trends
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'], label='Loss')
plt.plot(hist.history['accuracy'], label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Model Training')
plt.legend()
plt.show()
