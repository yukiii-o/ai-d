import streamlit as st

# Sample labeled data
data = [
    ("She enjoys visiting new cafes around the city.", 0),
    ("We went bowling with some friends last weekend.", 0),
    ("He spent the day at the beach, reading a book.", 0),
    ("I made a cup of herbal tea to relax.", 0),
    ("She took a picture of the sunset with her camera.", 0),
    ("We enjoyed a delicious lunch at a local diner.", 0),
    ("He loves collecting old coins from different countries.", 0),
    ("The stars were so bright on the clear night.", 0),
    ("I spent the evening working on a puzzle.", 0),
    ("She wore a cozy sweater to keep warm.", 0),
    # (More samples can be added as needed...)
]

# Separate text and labels
texts, labels = zip(*data)

# Preprocessing

def preprocess(text):
    text = text.lower()
    text = "".join(char for char in text if char.isalpha() or char.isspace())
    return text

def tokenize(text):
    return text.split()

# Build vocabulary
vocab = set()
for text in texts:
    tokens = tokenize(preprocess(text))
    vocab.update(tokens)
vocab = list(vocab)

def text_to_vector(text):
    vector = [0] * len(vocab)
    tokens = tokenize(preprocess(text))
    for token in tokens:
        if token in vocab:
            vector[vocab.index(token)] += 1
    return vector

# Convert data
X = [text_to_vector(text) for text in texts]
y = list(labels)

# Split manually (can also use sklearn)
def train_test_split(X, y, test_size=0.25):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        classes = set(y)

        for c in classes:
            self.class_probs[c] = sum(1 for label in y if label == c) / n_samples

        for c in classes:
            self.feature_probs[c] = [0] * n_features
            class_samples = [X[i] for i in range(n_samples) if y[i] == c]
            for i in range(n_features):
                count = sum(sample[i] for sample in class_samples)
                self.feature_probs[c][i] = (count + 1) / (len(class_samples) + n_features)

    def predict(self, X):
        predictions = []
        for sample in X:
            max_prob = -1
            best_class = -1
            for c in self.class_probs:
                prob = self.class_probs[c]
                for i in range(len(sample)):
                    if sample[i] > 0:
                        prob *= self.feature_probs[c][i]
                if prob > max_prob:
                    max_prob = prob
                    best_class = c
            predictions.append(best_class)
        return predictions

    def predict_proba(self, X):
        probabilities = []
        for sample in X:
            class_probs = {}
            total = 0
            for c in self.class_probs:
                prob = self.class_probs[c]
                for i in range(len(sample)):
                    if sample[i] > 0:
                        prob *= self.feature_probs[c][i]
                class_probs[c] = prob
                total += prob
            for c in class_probs:
                class_probs[c] = class_probs[c] / total
            probabilities.append(class_probs)
        return probabilities

# Train classifier
classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)

# Detection function
def detect_ai_text(text):
    vector = text_to_vector(text)
    prediction = classifier.predict([vector])[0]
    proba = classifier.predict_proba([vector])[0]
    confidence = proba[prediction] * 100
    return ("AI-generated" if prediction == 1 else "Human-written", confidence)

# Streamlit UI
st.title("AI Text Detector")
st.write("Enter a sentence to detect whether it's written by an AI or a human.")

user_input = st.text_input("Your sentence:")

if user_input:
    label, confidence = detect_ai_text(user_input)
    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")
