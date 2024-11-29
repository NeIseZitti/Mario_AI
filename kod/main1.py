import utils1

import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

distance_traveled = ram[0x006D] # bu şekilde ramden value alınıyor

# import numpy as np
# from collections import namedtuple

for i in range(1):
    env.render()  # Oyun ekranını gösterir

    tiles = list(utils1.SMB.get_tiles(ram))
    keys = utils1.SMB.get_tiles(ram).keys()
    values = utils1.SMB.get_tiles(ram).values()

    keys = list(keys)
    values = [tile.value for tile in values]  

    for i in range(15):
        for j in range(16):
            print(values[i*16 + j], end=" ")
        print()




print(utils1.EnemyType.Goomba.value)


print("miyav")