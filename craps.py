"""
Craps Pass Line Simulator
- User chooses number of games and a fixed per-game bet.
- Come-out: 7/11 = win; 2/3/12 = loss; else set 'point' and roll until point (win) or 7 (loss).
- Payout model: win => +2*bet (keep bet + equal winnings); loss => -bet.
- Prints per-game outcomes and a final summary.
"""

import random

def roll_two():
    """Return sum of two fair six-sided dice."""
    return random.randint(1, 6) + random.randint(1, 6)

def ask_int_at_least(prompt, minimum):
    """
    Keep asking until user enters an integer >= minimum.
    Explains errors instead of crashing.
    """
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
            if val >= minimum:
                return val
            print(f"Please enter a whole number >= {minimum}.")
        except ValueError:
            print("Please enter a whole number (e.g., 3).")

def play_one_game(bet):
    """
    Play one Pass Line game.
    Returns tuple (delta_cash, message).
    delta_cash is +2*bet for a win, -bet for a loss.
    """
    s = roll_two()  # come-out sum

    # Immediate decisions on the come-out roll
    if s in (7, 11):
        return (2 * bet, f"WIN on come-out ({s})")
    if s in (2, 3, 12):
        return (-bet, f"LOSS on come-out ({s})")

    # Point phase: keep rolling until point or seven-out
    point = s
    while True:
        t = roll_two()
        if t == point:
            return (2 * bet, f"WIN by making the point {point}")
        if t == 7:
            return (-bet, "LOSS (seven-out)")
        # otherwise roll again

def main():
    """Collect inputs, play multiple games, and print a readable summary."""
    print("=== Pass Line Simulator ===")

    # Inputs with validation
    games = ask_int_at_least("How many games? ", 1)
    bet   = ask_int_at_least("Bet per game (whole dollars, ≥ 0): ", 0)

    # Session bookkeeping
    total = 0
    wins = 0
    losses = 0

    # Play requested number of games
    for g in range(1, games + 1):
        delta, msg = play_one_game(bet)
        total += delta
        if delta > 0:
            wins += 1
        else:
            losses += 1
        print(f"Game {g:>3}: {msg:28s} | Change: {delta:+d} | Bankroll: {total:+d}")

    # Summary
    print("\n--- Summary ---")
    print(f"Games: {games}  Wins: {wins}  Losses: {losses}")
    win_rate = (wins / games) * 100
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Net winnings: ${total:+d}")

if __name__ == "__main__":
    main()
