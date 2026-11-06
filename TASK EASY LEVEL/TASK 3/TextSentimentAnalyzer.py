# TASK 3 EASY LEVEL
positive_words = [
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "happy",
    "love",
    "wonderful",
    "fantastic",
    "best",
    "nice",
    "enjoy",
    "beautiful",
    "perfect"
]
negative_words = [
    "bad",
    "terrible",
    "horrible",
    "worst",
    "sad",
    "hate",
    "angry",
    "poor",
    "awful",
    "disappointed",
    "boring",
    "difficult",
    "useless",
    "problem"
]
text = input("Enter a sentence: ")
text = text.lower()
words = text.split()
positive_count = 0
negative_count = 0
for word in words:
    word = word.strip(".,!?;:")
    if word in positive_words:
        positive_count += 1
    if word in negative_words:
        negative_count += 1
print("\n SENTIMENT ANALYSIS")
print("Positive words:", positive_count)
print("Negative words:", negative_count)
if positive_count > negative_count:
    print("Sentiment: POSITIVE ")
elif negative_count > positive_count:
    print("Sentiment: NEGATIVE ")
else:
    print("Sentiment: NEUTRAL ")