import retro

import utils

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

for i in range(1000000000000):
    env.render()  # Oyun ekranını gösterir.




print(0x6E)

print("miyav")
enemies = []
enemies = [None for _ in range(5)]
for i in range(5):
    print(enemies[i])

print(0x100)


print(utils.SMB.get_mario_location_in_level(ram))