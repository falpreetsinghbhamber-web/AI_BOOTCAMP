!pip install scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Toy Training Dataset
training_sentences = [
    "Claim your free gift card worth $1000 now",
    "URGENT! You have won a lottery, click here to claim",
    "Win cash prizes and free bonuses today",
    "Hey, are we still meeting for group study at 5 PM?",
    "Please find attached the project submission document",
    "Can you share the lecture notes for tomorrow's class?",
]
labels = ["Spam", "Spam", "Spam", "Ham", "Ham", "Ham"]  # 'Ham' means legitimate

# 2. Convert text sentences into numerical feature vectors (Bag of Words)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# 3. Train Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X_train, labels)

print("Spam Detector Model Trained Successfully!\n")

# 4. CHANGED: accept messages from the user and predict
print("Type a message to check it. Type 'quit' to stop.\n")

while True:
  message = input("Enter message: ")            
  if message.strip().lower() == "quit":
    print("Goodbye!")
    break
  if message.strip() == "":                    
    continue

  X_test = vectorizer.transform([message])
  prediction = classifier.predict(X_test)[0]
  probs = classifier.predict_proba(X_test)[0]  
  confidence = max(probs) * 100

  print(f"Prediction: [{prediction}]  (confidence: {confidence:.1f}%)\n")
