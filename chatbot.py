# Import necessary modules
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Load pre-trained components
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data.json').read())
words = pickle.load(open("words.pkl", 'rb'))
classes = pickle.load(open("classes.pkl", 'rb'))
model = load_model("chatbot_model.model.keras")

# Utility functions to clean input and create bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': r[1]} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand. Can you clarify?"

# Feedback storage
feedback_storage = []

# Chat loop with feedback mechanism
print("Go! The bot is running.")

while True:
    message = input("You: ")
    prediction = predict_class(message)
    response = get_response(prediction, intents)

    print(f"Chatbot: {response}")

    # Ask for feedback
    feedback = input("Was this helpful? (yes/no): ")

    # If positive feedback, store the question and response
    if feedback.lower() == "yes":
        feedback_storage.append({
            "question": message,
            "response": response,
            "intent": prediction[0]["intent"]
        })

# Update the dataset with new data from feedback
def update_dataset(feedback_storage, dataset_path="data.json"):
    # Load existing dataset
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    # Add new questions and responses to the corresponding intents
    for feedback in feedback_storage:
        # Find the intent in the dataset
        for intent in data["intents"]:
            if intent["tag"] == feedback["intent"]:
                # Add the new question to patterns
                intent["patterns"].append(feedback["question"])
                # Optionally, add the response to responses
                intent["responses"].append(feedback["response"])

    # Save the updated dataset
    with open(dataset_path, 'w') as file:
        json.dump(data, file, indent=4)

    # Clear feedback storage after updating
    feedback_storage.clear()

# Retrain the model with the updated dataset
def retrain_model():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import SGD
    
    # Re-import necessary components
    nltk.download("punkt")
    nltk.download("wordnet")
    
    words = []
    classes = []
    documents = []
    ignore_words = ["?", "!"]

    # Load dataset
    with open("data.json", 'r') as file:
        data = json.load(file)
    
    # Build vocabulary and document set
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])
    
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    # Save the vocabulary and classes
    with open("words.pkl", 'wb') as f:
        pickle.dump(words, f)
    
    with open("classes.pkl", 'wb') as f:
        pickle.dump(classes, f)

    # Create training data
    training = []
    for doc in documents:
        bag = [0] * len(words)
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        for word in pattern_words:
            if word in words:
                bag[words.index(word)] = 1
        training.append([bag, classes.index(doc[1])])

    training = np.array(training)

    # Build and train a simple neural network model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(words),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation="softmax"))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # Train the model
    model.fit(training[:, 0], tf.keras.utils.to_categorical(training[:, 1], num_classes=len(classes)), epochs=200, batch_size=5, verbose=1)

    # Save the new model
    model.save("chatbot_model.model.keras")
