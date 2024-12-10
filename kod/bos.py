import json
import numpy as np

# JSON dosyasını açma
with open('chromosomes.json', 'r') as f:
    data = json.load(f)

# Başlangıçta 64 Mario'yu tutacak array
num_marios = 64  # 64 Mario
input_size = 70  # Girdiler
hidden_layer_1 = 64  # 1. katman
hidden_layer_2 = 32  # 2. katman
output_size = 6  # Çıktı

# 64 Mario'yu tutacak array oluşturuyoruz (ilk başta boş)
marios = np.empty((num_marios, 2), dtype=object)

# İlk 16 Mario'yu dolduruyoruz
for mario_id in range(16):
    mario_data = data[f'mario_{mario_id+1}']
    
    weights = mario_data['weights']
    biases = mario_data['biases']
    
    # Ağırlıkları 3 boyutlu array'e ekliyoruz
    mario_weights = [
        weights['input_to_hidden1'],
        weights['hidden1_to_hidden2'],
        weights['hidden2_to_output']
    ]
    
    # Biasları 3 boyutlu array'e ekliyoruz
    mario_biases = [
        biases['hidden1_bias'],
        biases['hidden2_bias'],
        biases['output_bias']
    ]
    
    # Ağırlıkları ve biasları marios array'ine yerleştiriyoruz
    marios[mario_id][0] = mario_weights  # 0: weights
    marios[mario_id][1] = mario_biases  # 1: biases


print("------------------------")


# marionun bir weight değerine erişirkene olur gibi
print(marios[0][0][0][0][0])
print(marios[1][0][0][0][0])

# marionun bir bias değerine erişirkene olur gibi
print(marios[0][1][0][0])

print("--------------------------")
print("--------------------------")
print("mario sayisi:", len(marios))
print("weight ve bias için iki tane array:", len(marios[0]))
print("------------------------")
print("------weight kısmı------")
print("layerlar arası için 3 tane array:", len(marios[0][0]))
print("ilk layerdan ikinciye 70'er array:", len(marios[0][0][0]))      # ikinci layerlarda kullanılacak weight sayıları
print("70 arrayin her birinde 64 tane weight:",len(marios[0][0][0][0]))
print("ikinci layerdan üçüncüye 64'er array:", len(marios[0][0][1]))
print("64 arrayin her birinde 32 tane weight:",len(marios[0][0][1][0]))
print("üçüncü layerdan dördüncüye 32şer array:", len(marios[0][0][2]))
print("32 arrayin her birinde 6 tane weight:",len(marios[0][0][2][0]))
print("------------------------")
print("-------bias kısmı-------")
print("başlangıç hariç her layer için biasleri tutan array sayisi:", len(marios[0][1]))
print("ikinci layerdaki bias sayısı:", len(marios[0][1][0]))
print("üçüncü layerdaki bias sayısı:", len(marios[0][1][1]))
print("dördüncü layerdaki bias sayısı:", len(marios[0][1][2]))
print("--------------------------")
print("--------------------------")