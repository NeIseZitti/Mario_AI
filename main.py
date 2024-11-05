print("miyav")

import gym
import retro

available_games = retro.data.list_games()

print("Mevcut Oyunlar:")

for game in available_games:
    print(game)


