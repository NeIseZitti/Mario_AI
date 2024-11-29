from enum import Enum, unique # enum kullanmamızın sebebi sayıları değil isimleri kullanmak. Bu şekilde daha anlamlı olur.

# sayıların yanında bulunan 0x bunların 16lık tabanda yazıldığını ifade ediyor yani 1A gibi değerleri yazabiliyoruz.
# bu girdiğimiz değerleri rastgele vermedik. https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
# sitesindeki değerleri kullandık.

from collections import namedtuple # shape ve point adlı iki tane oluşturmak için bunlar konumu ve şekli tutacak.

import numpy as np

@unique
class EnemyType(Enum):
    Green_Koopa1 = 0x00
    Red_Koopa1   = 0x01
    Buzzy_Beetle = 0x02
    Red_Koopa2 = 0x03
    Green_Koopa2 = 0x04
    Hammer_Brother = 0x05
    Goomba = 0x06
    Blooper = 0x07
    Bullet_Bill = 0x08
    Green_Koopa_Paratroopa = 0x09
    Grey_Cheep_Cheep = 0x0A
    Red_Cheep_Cheep = 0x0B
    Pobodoo = 0x0C
    Piranha_Plant = 0x0D
    Green_Paratroopa_Jump = 0x0E
    Bowser_Flame1 = 0x10
    Lakitu = 0x11
    Spiny_Egg = 0x12
    Fly_Cheep_Cheep = 0x14
    Bowser_Flame2 = 0x15

    Generic_Enemy = 0xFF

@unique
class StaticTileType(Enum):
    Empty = 0x00
    Fake = 0x01
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Flagpole_Top =  0x24
    Flagpole = 0x25
    Coin_Block1 = 0xC0
    Coin_Block2 = 0xC1 
    Coin = 0xC2
    Breakable_Block = 0x51

    Generic_Static_Tile = 0xFF

@unique
class DynamicTileType(Enum):
    Mario = 0xAA

    Static_Lift1 = 0x24
    Static_Lift2 = 0x25
    Vertical_Lift1 = 0x26
    Vertical_Lift2 = 0x27
    Horizontal_Lift = 0x28
    Falling_Static_Lift = 0x29
    Horizontal_Moving_Lift=  0x2A
    Lift1 = 0x2B
    Lift2 = 0x2C
    Vine = 0x2F
    Flagpole = 0x30
    Start_Flag = 0x31
    Jump_Spring = 0x32
    Warpzone = 0x34
    Spring1 = 0x67
    Spring2 = 0x68

    Generic_Dynamic_Tile = 0xFF

Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])

class Tile(object):
    __slots__ = ['type']  # bu slots yazıldığında ekstra alan almıyomuş, type adında bir attribute'u olacak sadece
    def __init__(self, type: Enum): # type adındaki değişkeni alacak ve bu Enum türünde olacakmış
        self.type = type # girilen type değişkeni

class Enemy(object):
    def __init__(self, enemy_id: int, location: Point, tile_location: Point):
        enemy_type = EnemyType(enemy_id)
        self.type = EnemyType(enemy_id)
        self.location = location
        self.tile_location = tile_location




class SMB(object):
    # ekranda en fazla 5 enemy bulunabilir, bu yüzden sadece 5 enemy kontrol edilir.
    MAX_NUM_ENEMIES = 5
    PAGE_SIZE = 256
    NUM_BLOCKS = 8
    RESOLUTION = Shape(256, 240)
    NUM_TILES = 416  # 0x69f - 0x500 + 1 , ilk tile ve son tile arası
    NUM_SCREEN_PAGES = 2
    TOTAL_RAM = NUM_BLOCKS * PAGE_SIZE

    sprite = Shape(width=16, height=16)
    resolution = Shape(256, 240)
    status_bar = Shape(width=resolution.width, height=2*sprite.height)

    xbins = list(range(16, resolution.width, 16))
    ybins = list(range(16, resolution.height, 16))


    @unique
    class RAMLocations(Enum):
        # Since the max number of enemies on the screen is 5, the addresses for enemies are
        # the starting address and span a total of 5 bytes. This means Enemy_Drawn + 0 is the
        # whether or not enemy 0 is drawn, Enemy_Drawn + 1 is enemy 1, etc. etc.
        Enemy_Drawn = 0x0F
        Enemy_Type = 0x16
        Enemy_X_Position_In_Level = 0x6E
        Enemy_X_Position_On_Screen = 0x87
        Enemy_Y_Position_On_Screen = 0xCF

        Player_X_Postion_In_Level       = 0x06D
        Player_X_Position_On_Screen     = 0x086

        Player_X_Position_Screen_Offset = 0x3AD
        Player_Y_Position_Screen_Offset = 0x3B8
        Enemy_X_Position_Screen_Offset = 0x3AE

        Player_Y_Pos_On_Screen = 0xCE
        Player_Vertical_Screen_Position = 0xB5


    @classmethod
    def get_enemy_locations(cls, ram: np.ndarray):
        # Sadece çizilmiş enemyleri umursuyoruz. Ötekiler bellekte olabilir ama görmediğin şey seni incitmez -_-
        enemies = []

        for enemy_num in range(cls.MAX_NUM_ENEMIES): # 0dan 5e kadar gidiyü
            enemy = ram[cls.RAMLocations.Enemy_Drawn.value + enemy_num] # bellekteki enemydrawn değişkenlerini tek tek alıyor.
            # düşman var mi? 1/0
            if enemy:
                # Get the enemy X location.
                x_pos_level = ram[cls.RAMLocations.Enemy_X_Position_In_Level.value + enemy_num]
                x_pos_screen = ram[cls.RAMLocations.Enemy_X_Position_On_Screen.value + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen #- ram[0x71c]
                # print(ram[0x71c])
                # enemy_loc_x = ram[cls.RAMLocations.Enemy_X_Position_Screen_Offset.value + enemy_num]
                # Get the enemy Y location.
                enemy_loc_y = ram[cls.RAMLocations.Enemy_Y_Position_On_Screen.value + enemy_num]
                # Set location
                location = Point(enemy_loc_x, enemy_loc_y)
                ybin = np.digitize(enemy_loc_y, cls.ybins)
                xbin = np.digitize(enemy_loc_x, cls.xbins)
                tile_location = Point(xbin, ybin)

                # Grab the id
                enemy_id = ram[cls.RAMLocations.Enemy_Type.value + enemy_num]
                # Create enemy-
                e = Enemy(0x6, location, tile_location)

                enemies.append(e)

        return enemies
    
    @classmethod
    def get_mario_location_in_level(cls, ram: np.ndarray) -> Point:
        mario_x = ram[cls.RAMLocations.Player_X_Postion_In_Level.value] * 256 + ram[cls.RAMLocations.Player_X_Position_On_Screen.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value]
        return Point(mario_x, mario_y)
