# Movie Recommendation System: TASK 2 EASY
movies = {
    "action": [
        "Top Gun: Maverick"
    ],

    "comedy": [
        "The Hangover",
        "3 Idiots",
        "Home Alone",
        "Jumanji"
    ],

    "drama": [
        "The Shawshank Redemption",
         "Avengers: Endgame",
        "John Wick",
        "Mission Impossible",
       "Forrest Gump",
        "The Pursuit of Happiness",
        "12th Fail"
    ],

    "horror": [
        "The Conjuring",
        "Insidious",
        "It",
        "A Quiet Place"
    ],

    "sci-fi": [
        "Interstellar",
        "Inception",
        "The Matrix",
        "Avatar"
    ]
}
print("      MOVIE RECOMMENDATION SYSTEM")
print("\nAvailable Genres:")
print("1. Action")
print("2. Comedy")
print("3. Drama")
print("4. Horror")
print("5. Sci-Fi")
genre = input("\nEnter your preferred genre: ").lower().strip()
if genre in movies:
    print("\nRecommended Movies for", genre.title(), ":")
    print("------------------------------------")

    for movie in movies[genre]:
        print("•", movie)

else:
    print("\nSorry! No movies found for this genre.")
    print("Please choose from: Action, Comedy, Drama, Horror, Sci-Fi")