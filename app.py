import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="AI Text Detector", page_icon="🤖", layout="centered")

st.title("AIDENTIFY - AI Text Detector")
st.markdown("Welcome! Paste a sentence or paragraph below and find out if it was **AI-generated** or **Human-written**.")

# Large input area
user_input = st.text_area("Input your text here:", height=200, placeholder="Type or paste your text...")

# Detection logic triggers on button click
if st.button("SCAN") and user_input.strip():
    # ====== Begin original code block ======

    import math

    data = [
        ("She enjoys visiting new cafes around the city.", 0),
        ("We went bowling with some friends last weekend.", 0),
        ("He spent the day at the beach, reading a book.", 0),
        # (continue with the rest of your dataset)
    ]

    texts, labels = zip(*data)

    def preprocess(text):
        text = text.lower()
        text = "".join(char for char in text if char.isalpha() or char.isspace())
        return text

    def tokenize(text):
        return text.split()

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

    X = [text_to_vector(text) for text in texts]
    y = list(labels)

    def train_test_split(X, y, test_size=0.25):
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

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
                self.feature_probs[c] = [0] * n_features
                class_samples = [X[i] for i in range(n_samples) if y[i] == c]
                for feature_index in range(n_features):
                    feature_count = sum(sample[feature_index] for sample in class_samples)
                    self.feature_probs[c][feature_index] = (feature_count + 1) / (len(class_samples) + n_features)

        def predict(self, X):
            predictions = []
            for sample in X:
                max_prob = -math.inf
                best_class = -1
                for c in self.class_probs:
                    prob = math.log(self.class_probs[c])
                    for feature_index in range(len(sample)):
                        if sample[feature_index] > 0:
                            prob += math.log(self.feature_probs[c][feature_index])
                    if prob > max_prob:
                        max_prob = prob
                        best_class = c
                predictions.append(best_class)
            return predictions

        def predict_proba(self, X):
            probabilities = []
            for sample in X:
                class_probs = {}
                total_prob = 0
                for c in self.class_probs:
                    prob = self.class_probs[c]
                    for feature_index in range(len(sample)):
                        if sample[feature_index] > 0:
                            prob *= self.feature_probs[c][feature_index]
                    class_probs[c] = prob
                    total_prob += prob

                for c in class_probs:
                    class_probs[c] = class_probs[c] / total_prob
                probabilities.append(class_probs)
            return probabilities

    classifier = NaiveBayesClassifier()
    classifier.fit(X_train, y_train)

    def detect_ai_text(text):
        vector = text_to_vector(text)
        prediction = classifier.predict([vector])[0]
        proba = classifier.predict_proba([vector])[0]
        confidence = proba[prediction] * 100
        return ("AI-generated" if prediction == 1 else "Human-written", confidence)

    result, confidence = detect_ai_text(user_input)

    # Output section
    st.markdown("### 📊 Result:")
    st.success(f"**Prediction:** {result}")
    st.info(f"**Confidence:** {confidence:.2f}%")
  print("Thank you for using our program :) ")
else:
    st.markdown("Enter a sentence and click the button to get a prediction.")
