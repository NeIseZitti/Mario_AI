import json
import numpy as np
import random

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



def crossover(parent1, parent2):
    """İki ebeveynin ağırlık ve biaslarını birleştirerek yeni bir çocuk oluşturur."""
    child_weights = []
    child_biases = []
    
    # Ağırlıklar için crossover
    for w1, w2 in zip(parent1[0], parent2[0]):  # Her bir layer için
        w1 = np.array(w1)  # Listeyi numpy array'e çevir
        w2 = np.array(w2)  # Listeyi numpy array'e çevir
        mask = np.random.rand(*w1.shape) < 0.5
        child_weights.append(np.where(mask, w1, w2).tolist())  # Geri listeye çevir
    
    # Biaslar için crossover
    for b1, b2 in zip(parent1[1], parent2[1]):  # Her bir layer için
        b1 = np.array(b1)  # Listeyi numpy array'e çevir
        b2 = np.array(b2)  # Listeyi numpy array'e çevir
        mask = np.random.rand(*b1.shape) < 0.5
        child_biases.append(np.where(mask, b1, b2).tolist())  # Geri listeye çevir
    
    return [child_weights, child_biases]

# İlk 16 Mario’dan 64 Mario’ya çıkış
for i in range(16, 64):
    # Rastgele iki ebeveyn seç
    parent1 = random.choice(marios[:16])
    parent2 = random.choice(marios[:16])
    
    # Çocuk oluştur ve mevcut marios array'ine ekle
    marios[i] = crossover(parent1, parent2)

# Sonuçları kontrol edebilirsiniz:
# print("Yeni Mario sayısı:", len(new_marios))
# print("İlk çocuğun ağırlıkları:", new_marios[16][0])  # İlk çocuğun ağırlıkları
# print("İlk çocuğun biasları:", new_marios[16][1])  # İlk çocuğun biasları


print(marios[63][0][2][0][0])
print(marios[63][0][2][0][1])


def mutate(mario, mutation_rate=0.05):
    """
    Mario'nun ağırlık ve biaslarına ±%5 mutasyon uygular.
    
    Args:
        mario (list): Mario'nun ağırlık (weights) ve bias (biases) verilerini içeren liste.
        mutation_rate (float): Mutasyon miktarı (ör. 0.05 ile ±%5 arasında değişim).

    Returns:
        list: Mutasyon uygulanmış Mario verileri.
    """
    mutated_weights = []
    mutated_biases = []
    
    # Ağırlıklara mutasyon uygula
    for weight_layer in mario[0]:  # Her bir layer için
        weight_layer = np.array(weight_layer)  # Listeyi NumPy array'e çevir
        mutation = (np.random.rand(*weight_layer.shape) * 2 - 1) * mutation_rate
        mutated_layer = weight_layer + mutation
        mutated_weights.append(mutated_layer.tolist())  # Geri listeye çevir
    
    # Biaslara mutasyon uygula
    for bias_layer in mario[1]:  # Her bir layer için
        bias_layer = np.array(bias_layer)  # Listeyi NumPy array'e çevir
        mutation = (np.random.rand(*bias_layer.shape) * 2 - 1) * mutation_rate
        mutated_layer = bias_layer + mutation
        mutated_biases.append(mutated_layer.tolist())  # Geri listeye çevir
    
    return [mutated_weights, mutated_biases]

# Yeni Mario'lara mutasyon uygulama
for i in range(16, 64):  # İlk 16 Mario'yu koruyoruz, geri kalan 48'e mutasyon uyguluyoruz
    marios[i] = mutate(marios[i])


print(marios[63][0][2][0][0])
print(marios[63][0][2][0][1])
