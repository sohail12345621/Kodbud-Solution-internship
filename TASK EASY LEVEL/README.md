# KODBUD SOLUTIONS — EASY LEVEL TASKS

This folder contains the **Easy Level tasks** completed as part of my internship at **Kodbud Solutions**.

The tasks focus on implementing basic Python programs and understanding fundamental programming concepts such as user input, conditional statements, loops, lists, dictionaries, and basic text processing.

---

## Tasks Overview

| Task | Project | Description | Technology |
|---|---|---|---|
| Task 1 | Basic Python Program | A beginner-level Python application demonstrating fundamental programming concepts. | Python |
| Task 2 | Movie Recommendation System | Recommends movies based on the genre selected by the user. | Python |
| Task 3 | Text Sentiment Analyzer | Determines whether user-provided text is positive, negative, or neutral. | Python |

---

# Task 1 — Basic Python Program

### Objective

The objective of Task 1 was to create a simple Python program while understanding the fundamentals of programming and user interaction.

### Implementation

The program demonstrates basic Python concepts including:

- Taking input from the user
- Processing user input
- Using variables
- Conditional statements
- Displaying results using `print()`

### Concepts Used

- Python variables
- `input()`
- `print()`
- Conditional statements
- Basic data processing

### Outcome

A functional Python program was created that accepts user input, processes it according to the defined logic, and displays the appropriate result.

---

# Task 2 — Movie Recommendation System

### Objective

The objective of Task 2 was to understand the basic concept behind **recommendation systems** by creating a simple movie recommendation program.

### Implementation

A collection of movies was stored using a Python dictionary. Movies were categorized according to different genres such as:

- Action
- Comedy
- Drama
- Horror
- Sci-Fi

The program asks the user to enter their preferred genre and then displays movies belonging to that category.

### How It Works

1. The program stores movie information in a dictionary.
2. The user enters their preferred genre.
3. The input is converted to lowercase for easier matching.
4. The program checks whether the selected genre exists.
5. Matching movies are displayed to the user.
6. If the genre is unavailable, an appropriate message is displayed.

### Concepts Used

- Python dictionaries
- Python lists
- User input
- String manipulation
- `if-else` conditions
- `for` loops

### Example

```text
Enter your preferred genre: action

Recommended Movies:
- Avengers: Endgame
- John Wick
- Mission Impossible
- Top Gun: Maverick
```

### Outcome

A simple **content-based movie recommendation system** was developed that recommends movies according to the user's selected genre.

---

# Task 3 — Text Sentiment Analyzer

### Objective

The objective of Task 3 was to understand how text can be analyzed to determine basic emotional sentiment.

### Implementation

Two lists of words were created:

- **Positive words** — such as `good`, `great`, `amazing`, `excellent`, and `happy`
- **Negative words** — such as `bad`, `terrible`, `horrible`, `boring`, and `sad`

The program accepts a sentence from the user and checks how many positive and negative words occur in the sentence.

### How It Works

1. The user enters a sentence.
2. The sentence is converted to lowercase.
3. The sentence is divided into individual words.
4. Each word is compared against the positive and negative word lists.
5. Positive and negative word counts are calculated.
6. The counts are compared to determine the overall sentiment.

### Sentiment Rules

```text
Positive words > Negative words → POSITIVE

Negative words > Positive words → NEGATIVE

Positive words = Negative words → NEUTRAL
```

### Example

```text
Enter a sentence: I love this amazing movie

Positive words: 2
Negative words: 0
Sentiment: POSITIVE
```

### Concepts Used

- Python lists
- String manipulation
- `split()`
- Loops
- Conditional statements
- Word matching
- Basic text analysis

### Outcome

A basic **rule-based sentiment analyzer** was developed that classifies user-provided text as positive, negative, or neutral based on predefined vocabulary.



#  Technologies Used

The Easy Level tasks were implemented using:

- **Python**
- Lists
- Dictionaries
- Loops
- Conditional Statements
- String Processing
- User Input



#  Learning Outcomes

Through these tasks, the following concepts were practiced:

- Writing and executing Python programs
- Working with user input
- Using lists and dictionaries to store data
- Applying conditional logic
- Using loops for data processing
- Performing basic string manipulation
- Building simple rule-based systems
- Understanding the basic idea behind recommendation systems
- Understanding fundamental sentiment analysis


##  Author

**Sohail Shaikh**

Computer Engineering Student  
Thakur College of Engineering and Technology

##  Internship

**Kodbud Solutions Internship**

This repository contains the practical tasks completed during the internship program.