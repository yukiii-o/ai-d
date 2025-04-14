import streamlit as st

# Dataset: 0 = human-written, 1 = AI-generated
data = [
    ("This text was generated by an AI model.", 1),
    ("The AI-generated content is becoming more advanced.", 1),
    ("Machine learning models can generate realistic text.", 1),
    ("AI-generated text is often indistinguishable from human writing.", 1),

    ("This is a human-written sentence.", 0),
    ("I enjoy reading books and writing stories.", 0),
    ("Natural language processing is a fascinating field.", 0),
    ("I love spending time with my family and friends.", 0),
]

# --- Preprocessing ---
def preprocess(text):
    text = text.lower()
    text = "".join(char for char in text if char.isalpha() or char.isspace())
    return text

def tokenize(text):
    return text.split()

vocab = set()
for text, _ in data:
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

# --- Split Data ---
def train_test_split(X, y, test_size=0.25):
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

texts, labels = zip(*data)
X = [text_to_vector(text) for text in texts]
y = list(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# --- Naive Bayes Classifier ---
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
            total_prob = 0
            for c in self.class_probs:
                prob = self.class_probs[c]
                for i in range(len(sample)):
                    if sample[i] > 0:
                        prob *= self.feature_probs[c][i]
                class_probs[c] = prob
                total_prob += prob
            for c in class_probs:
                class_probs[c] = class_probs[c] / total_prob
            probabilities.append(class_probs)
        return probabilities

# --- Train Model ---
classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)

# --- Detection Function ---
def detect_ai_text(text):
    vector = text_to_vector(text)
    prediction = classifier.predict([vector])[0]
    proba = classifier.predict_proba([vector])[0]
    confidence = proba[prediction] * 100
    label = "AI-generated" if prediction == 1 else "Human-written"
    return label, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="AI Text Detector", page_icon="🤖")
st.title("🤖 AI vs Human Text Detector")

user_input = st.text_area("Enter a sentence or paragraph to check:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result, confidence = detect_ai_text(user_input)
        st.subheader("🧠 Result")
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")

st.markdown("---")
st.caption("This tool uses a simple Naive Bayes model on a small dataset. For better accuracy, expand the dataset or use larger ML models.")
