import pandas as pd
import re
import sys

from ucimlrepo import fetch_ucirepo

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Fetch dataset
youtube_spam_collection = fetch_ucirepo(id=380)

# Ensure Windows stdout can print Unicode
sys.stdout.reconfigure(encoding='utf-8')

# Features and labels
X = youtube_spam_collection.data.features
y = youtube_spam_collection.data.targets

# Combine into dataframe
df = pd.concat([X, y], axis=1)

df.columns = list(X.columns) + ['CLASS']

# Clean text function
def clean_text(text):

    text = str(text).lower()

    # remove links
    text = re.sub(r"http\S+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text

# Apply cleaning
df['CONTENT'] = df['CONTENT'].apply(clean_text)

# Convert text into vectors
vectorizer = TfidfVectorizer(ngram_range=(1,2))

X = vectorizer.fit_transform(df['CONTENT'])

y = df['CLASS']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Test custom comment
user_comment = input("Enter Comment: ")

comment = [user_comment]

# Transform comment
comment_vector = vectorizer.transform(comment)

# Predict
prediction = model.predict(comment_vector)

# Result
print("\nTest Comment:", comment[0])

if prediction[0] == 1:
    print("Prediction: Spam Comment")
else:
    print("Prediction: Genuine Comment")