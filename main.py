import utils

import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

distance_traveled = ram[0x006D] # bu şekilde ramden value alınıyor

import numpy as np
from collections import namedtuple

for i in range(10):
    env.render()  # Oyun ekranını gösterir

    tiles = list(utils.SMB.get_tiles(ram))
    keys = utils.SMB.get_tiles(ram).keys()
    values = utils.SMB.get_tiles(ram).values()

    keys = list(keys)
    values = [tile.value for tile in values]  
    # values listesi içindeki her tile'in value değerini alır

    for i in range(15):
        for j in range(16):
            # Anahtar değerini ve karşılık gelen value değerini yan yana yazdır
            print(values[i*16 + j], end=" ")
        print()


print("miyav")