from iphyre.simulator import IPHYRE
from iphyre.games import PARAS
import iphyre

print(iphyre.games.GAMES)

# games = iphyre.games.GAMES
game_name = "hole"
id = 1


# for id in range(20):


import json

# Load the game configuration from the specified file
with open(
    "../iphyre/test_toy20250110/20250427/output_rotate/20250428_175035_game_15/config.json",
    "r",
) as f:
    game_config = json.load(f)

game_name = "test"
PARAS[game_name] = game_config["config"]


def write_game(gap=60):
    left = 300 - gap / 2
    right = 300 + gap / 2

    border = 10

    game_basic_config = {
        "block": [
            [[100.0, 400.0], [left, 400.0]],
            [[right, 400.0], [500.0, 400.0]],
            [[500.0, 300.0], [500.0, 380.0]],
            [[100.0, 150.0], [380.0, 200.0]],
            [[160.0, 100.0], [160.0, 130.0]],
            [[left, 360.0], [left, 380.0]],
            [[right, 360.0], [right, 380.0]],
            [[100.0, 300.0], [100.0, 380.0]],
        ],
        "ball": [[120.0, 120.0, 20.0]],
        "eli": [0, 0, 0, 1, 1, 0, 0, 0, 0],
        "dynamic": [0, 0, 0, 0, 0, 0, 0, 0, 1],
    }

    game_name = f"hole_{id}"

    PARAS[game_name] = game_basic_config


env = IPHYRE(game=game_name)
env.play()
