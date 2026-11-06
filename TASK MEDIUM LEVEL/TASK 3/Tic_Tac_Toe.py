
#   AI-BASED TIC-TAC-TOE GAME

import math
board = [" " for _ in range(9)]
def display_board():
    print("\n")
    print("     |     |")
    print(f"  {board[0]}  |  {board[1]}  |  {board[2]}")
    print("-----+-----+-----")
    print(f"  {board[3]}  |  {board[4]}  |  {board[5]}")
    print("-----+-----+-----")
    print(f"  {board[6]}  |  {board[7]}  |  {board[8]}")
    print("     |     |")
    print()
def display_positions():
    print("Board positions:")
    print("""
     1 | 2 | 3
    ---+---+---
     4 | 5 | 6
    ---+---+---
     7 | 8 | 9
    """)
def check_winner(player):
    winning_combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]
    for combination in winning_combinations:
        if all(board[position] == player for position in combination):
            return True
    return False
def is_board_full():
    return " " not in board
def available_moves():
    return [i for i in range(9) if board[i] == " "]
def minimax(is_maximizing):
    if check_winner("O"):
        return 1
    if check_winner("X"):
        return -1
    if is_board_full():
        return 0
    if is_maximizing:
        best_score = -math.inf
        for move in available_moves():
            board[move] = "O"
            score = minimax(False)
            board[move] = " "
            best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for move in available_moves():
            board[move] = "X"
            score = minimax(True)
            board[move] = " "
            best_score = min(best_score, score)
        return best_score
def ai_move():

    best_score = -math.inf
    best_move = None

    for move in available_moves():

        board[move] = "O"

        score = minimax(False)

        board[move] = " "

        if score > best_score:
            best_score = score
            best_move = move
    board[best_move] = "O"
def play_game():

    print("=" * 40)
    print("       AI TIC-TAC-TOE")
    print("=" * 40)

    print("\nYou are X")
    print("Computer is O")

    display_positions()

    while True:

        display_board()

        while True:

            try:
                move = int(input("Enter your move (1-9): "))

                if move < 1 or move > 9:
                    print("Please enter a number between 1 and 9.")
                    continue

                position = move - 1

                if board[position] != " ":
                    print("That position is already occupied.")
                    continue

                board[position] = "X"
                break

            except ValueError:
                print("Please enter a valid number.")
        if check_winner("X"):
            display_board()
            print(" Congratulations! You won!")
            break
        if is_board_full():
            display_board()
            print("It's a draw!")
            break
        print("Computer is thinking...")
        ai_move()
        if check_winner("O"):
            display_board()
            print(" Computer wins!")
            break
        if is_board_full():
            display_board()
            print("It's a draw!")
            break
play_game()