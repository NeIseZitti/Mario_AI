print("miyav")

import retro
import utils

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

distance_traveled = ram[0x006D]

print(distance_traveled)




import numpy as np
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
# Set bins. Blocks are 16x16 so we create bins offset by 16
ybins = list(range(16, 240, 16))
xbins = list(range(16, 256, 16))

def get_enemy_locations(RAM):
    enemy_locations = []

    for enemy_num in range(5):
        enemy = RAM[0xF + enemy_num]
        # RAM locations 0x0F through 0x13 are 0 if no enemy
        # drawn or 1 if drawn to screen
        if enemy:
            # Grab the enemy location
            x_pos_level  = RAM[0x6E + enemy_num]
            x_pos_screen = RAM[0x87 + enemy_num]
            # The width in pixels is 256. 0x100 == 256.
            # Multiplying by x_pos_level gets you the
            # screen that is actually displayed, and then
            # add the offset from x_pos_screen
            enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen
            enemy_loc_y = RAM[0xCF + enemy_num]
            # Get row/col
            row = np.digitize(enemy_loc_y, ybins)
            col = np.digitize(enemy_loc_x, xbins)
            # Add location.
            # col moves in x-direction
            # row moves in y-direction
            location = Point(col, row)
            enemy_locations.append(location)

    return enemy_locations

print(get_enemy_locations(ram))

for i in range(1):
    action = env.action_space.sample()  # Rastgele bir hareket al
    env.step(action)  # Oyunda hareket ettir
    env.render()  # Oyun ekranını göster
    
    tiles = list(utils.SMB.get_tiles(ram))
    keys = utils.SMB.get_tiles(ram).keys()
    values = utils.SMB.get_tiles(ram).values()

    print(values)

    keys = list(keys)
    values = list(values)

    for i in range(15):
        for j in range(16):
            print(keys[i*16 + j], end=" ")
        print()

    #print(values)