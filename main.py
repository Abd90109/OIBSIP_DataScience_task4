# Spam Message Detection using Machine Learning

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
data = pd.read_csv("spam (1).csv", encoding="latin-1")

# Keep only the useful columns from the dataset
data = data[['v1', 'v2']]

# Rename the columns to make them easier to understand
data.columns = ['label', 'message']

print("Preview of dataset:")
print(data.head())


# Convert text labels into numbers
# ham = 0 (normal message)
# spam = 1 (spam message)
data['label'] = data['label'].map({'ham': 0, 'spam': 1}  )


# Separate input and output
X = data['message']
y = data['label']


# Split the data into training and testing sets
# 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Convert text messages into numerical features using TF-IDF
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train the model using Naive Bayes algorithm
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# Predict the results on test data
predictions = model.predict(X_test_vec)


# Check model performance
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Test the model with a custom message
test_message = ["Congratulations! You have won a free lottery ticket"]

test_vector = vectorizer.transform(test_message)

result = model.predict(test_vector)

if result[0] == 1:
    print("\nThis message is SPAM")
else:
    
    print("\nThis message is NOT SPAM")