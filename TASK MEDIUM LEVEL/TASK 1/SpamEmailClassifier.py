# ==========================================
#       SPAM EMAIL CLASSIFIER
# ==========================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
messages = [
    "Congratulations! You won a free lottery ticket",
    "Claim your prize now by clicking this link",
    "You have won 100000 dollars, claim your reward",
    "URGENT! You have been selected for a cash prize",
    "Get free money now, limited time offer",
    "Win a brand new car by entering this contest",
    "Congratulations, you are the lucky winner",
    "Click here to claim your free gift",
    "You have won a free vacation",
    "Exclusive offer! Get 90 percent discount now",
    "Free entry in our weekly prize draw",
    "You have been selected for a special reward",

    "Hello, how are you doing today?",
    "Can we meet tomorrow for the project discussion?",
    "Please send me the assignment report",
    "Your meeting is scheduled for tomorrow",
    "The project deadline has been extended",
    "Can you share the presentation with me?",
    "Thank you for your help with the project",
    "Please find the attached document",
    "Let's discuss the project during the meeting",
    "Your order has been shipped successfully",
    "Your appointment is confirmed for Monday",
    "The class will start at 10 AM tomorrow"
]

labels = [
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,

    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
]
X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 45)
print("       SPAM EMAIL CLASSIFIER")
print("=" * 45)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Spam", "Spam"],
    zero_division=0
))
while True:

    email = input("\nEnter an email/message to classify: ")

    if email.lower() == "exit":
        print("\nThank you for using the Spam Email Classifier.")
        break

    email_vectorized = vectorizer.transform([email])

    prediction = model.predict(email_vectorized)[0]

    probability = model.predict_proba(email_vectorized)[0]

    if prediction == 1:
        confidence = probability[1] * 100
        print(f"\nResult: SPAM")
        print(f"Confidence: {confidence:.2f}%")

    else:
        confidence = probability[0] * 100
        print(f"\nResult: NOT SPAM")
        print(f"Confidence: {confidence:.2f}%")