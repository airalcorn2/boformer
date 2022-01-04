import os

from animator import Match
from settings import DATA_DIR, GAMES_DIR

match_id = "8pcj21kccdevojs2f5l8lotwq"
try:
    match = Match(DATA_DIR, GAMES_DIR, match_id)
except FileNotFoundError:
    home_dir = os.path.expanduser("~")
    DATA_DIR = f"{home_dir}/scratch"
    GAMES_DIR = f"{home_dir}/scratch"
    match = Match(DATA_DIR, GAMES_DIR, match_id)

start_period = 1
start_time = "15:00"
stop_period = 1
stop_time = "15:15"
match.show_seq(start_period, start_time, stop_period, stop_time)
