import retro

import utils

import torch

import numpy as np

import os  # Konsolu temizlemek için


env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

# Sağ tuşuna basmayı temsil eden aksiyon dizisi
right_action = [0, 0, 0, 0, 0, 0, 0, 1, 0]
# Sağ ve zıplama tuşlarına basmayı temsil eden aksiyon dizisi
right_and_jump_action = [0, 0, 0, 0, 0, 0, 0, 1, 1]

# 7x10 luk input kullanıcaz.
# bi de mario hep kendi sütununu görsün istiyom bu inputu ona göre almak lazım
# input = []

# Değerleri normalize etmek için bir min-max scaling fonksiyonu tanımlayalım.
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

prev_lives = 2
for i in range(10000):
    
    env.render()  # Oyun ekranını gösterir.
    ram = env.get_ram()

    # 'get_tiles' fonksiyonu ile RAM'den tile verilerini alıyoruz
    values = utils.SMB.get_tiles(ram).values()
    # Her bir tile'ın value özelliğini alarak yeni bir liste oluşturuyoruz
    values = [tile.value for tile in values]

    # 7x10 luk input şu an ama 8x10'a geçmek istiyom. onu sonra burda değiştiririz.
    # bi de mario hep kendi sütununu görsün istiyom bu inputu ona göre almak lazım
    input = []

    if(utils.SMB.get_mario_row_col(ram)[1] < 10):
        for j in range(10):
            for k in range(7):
                input.append(values[(4+j)*16 + k + utils.SMB.get_mario_row_col(ram)[1]])
    else:
        for j in range(10):
            for k in range(7):
                input.append(values[(4+j)*16 + k + 8])

    # 0, 1, 170 ve 255 değerlerini normalize edelim.
    normalized_input = [normalize(v, 0, 255) for v in input]
    

    if(i%20 == 0):
        # print("\033[H\033[J", end="")  # Konsolu temizler ve imleci başa alır

        os.system('cls' if os.name == 'nt' else 'clear')  # Konsolu temizle

        # 15 satır ve 16 sütunluk bir matris gibi yazdırıyoruz
        for i in range(15):  # 15 satır
            for j in range(16):  # 16 sütun
                print(values[i*16 + j], end=" ")  # Her satırdaki 16 elemanı yazdır
            print()  # Yeni satıra geç
    

        print("input 7x10")
        for i in range(10):
            for j in range(7):
                print(input[7*i+j], end=" ")
            print()


        print("normalized input 7x10")
        for i in range(10):
            for j in range(7):
                print(normalized_input[7*i+j], end=" ")
            print()

        print(utils.SMB.get_mario_row_col(ram)[1])

    # Her 25 framede bir elini zıplamadan çek
    if i % 25 > 1:
        action = right_and_jump_action
    else:
        action = right_action



    # Sağ hareketini uygula
    obs, reward, done, info = env.step(action)
    
    # Oyun içinde her frame'de can sayısını kontrol edelim
    if info['lives'] < prev_lives:
        print("Mario'nun canı azaldı! Oyun bitiriliyor...")
        break  # Oyunu bitiririz
    
    # Oyunun başında Mario'nun canını kaydedelim
    prev_lives = info['lives']
    

env.close()

# print(utils.SMB.get_mario_row_col(ram)) # bu marionun hangi indexte olduğunu dönüyor.
# print(values)



# lazım olacak girdiler.
# bloklar
# belki marionun lokasyonu






# fitness fonksiyonumuz bu olsun
def fitness(frames, distance, did_win):
    return max(distance ** 1.9 - \
            frames ** 1.5 +   \
            min(max(distance-50, 0), 1) * 2000 + \
            did_win * 1e6, 0.00001)
# bu herhangi bi bireyin son noktadaki puanı olcak
# buna göre aralarından seçim yapıp yeni jenerasyon üreticez



# print(0x6E)

# print("miyav")
# enemies = []
# enemies = [None for _ in range(5)]
# for i in range(5):
#     print(enemies[i])

# print(0x100)


# print(utils.SMB.get_mario_location_in_level(ram))