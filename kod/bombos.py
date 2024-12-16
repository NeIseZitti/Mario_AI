import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os

import json


# Feedforward Neural Network
class MarioNN(nn.Module):
    def __init__(self, input_size, hidden_layer_1, hidden_layer_2, output_size):
        super(MarioNN, self).__init__()
        
        # İlk katman: 70 giriş, 64 nöronlu gizli katman
        self.fc1 = nn.Linear(input_size, hidden_layer_1)
        
        # İkinci katman: 64 nöron, 32 nöronlu gizli katman
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        
        # Çıkış katmanı: 32 nöron, 6 çıkış
        self.fc3 = nn.Linear(hidden_layer_2, output_size)

    def forward(self, x):
        # Aktivasyon fonksiyonu ile işlem
        x = F.relu(self.fc1(x))  # İlk katmandan geçiş
        x = F.relu(self.fc2(x))  # İkinci katmandan geçiş
        x = self.fc3(x)          # Çıkış katmanından geçiş (sonuç)
        return x

# Modeli oluşturma
input_size = 70  # Girdi boyutu
hidden_layer_1 = 64  # İlk gizli katman
hidden_layer_2 = 32  # İkinci gizli katman
output_size = 6  # Çıkış boyutu

model = MarioNN(input_size, hidden_layer_1, hidden_layer_2, output_size)

# Modeli test etme (örnek input ile)
sample_input = torch.randn(1, input_size)  # Rastgele örnek input
output = model(sample_input)
print(output)




# fitness fonksiyonumuz bu olsun
def fitness1(frames, distance, did_win):
    return max(distance ** 2 - \
            frames ** 1.2 +   \
            min(max(distance-50, 0), 1) * 100 + \
            did_win * 1e5, 0.00001)


frames = np.random.randint(1000, 5000, size=64)  # Her Mario için frame sayısı
distance = np.random.randint(0, 300, size=64)   # Mario'nun kat ettiği mesafe
did_win = np.random.randint(0, 2, size=64)      # 0: kaybetti, 1: kazandı

# Fitness değerlerini saklamak için liste veya array
fitness_values = []

for mario_id in range(64):
    f = fitness1(frames[mario_id], distance[mario_id], did_win[mario_id])
    fitness_values.append(f)

print(fitness_values)

average_fitness = sum(fitness_values) / len(fitness_values)
print(f"Average Fitness: {average_fitness}")







# JSON dosyasının adı
history_json = "history.json"

# Jenerasyonu yükleyen veya başlatan fonksiyon
def load_generation():
    if os.path.exists(history_json):
        with open(history_json, 'r') as file:
            data = json.load(file)
            return data.get("generation", 0)  # Eğer "generation" yoksa 0 döner
    return 0

# Yeni jenerasyonu kaydeden fonksiyon
def save_generation(generation, stats=None):
    # JSON dosyası varsa yükle
    if os.path.exists(history_json):
        with open(history_json, 'r') as file:
            data = json.load(file)
    else:
        data = {}

    # Jenerasyonu güncelle
    data["generation"] = generation

    # Eğer stats verilmişse, ekle
    if stats:
        if "stats" not in data:
            data["stats"] = {}
        data["stats"][f"generation_{generation}"] = stats

    # Dosyayı yaz
    with open(history_json, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Generation {generation} saved successfully!")

# Program başlatıldığında jenerasyonu yükle
current_generation = load_generation()

# Yeni jenerasyon için hesaplamalar
current_generation += 1
average_fitness = 2400.7
average_distance = 110.5
average_frames = 430.2
finish_rate = 0.15

# Yeni jenerasyon istatistiklerini kaydet
stats = {
    "average_fitness": average_fitness,
    "average_distance": average_distance,
    "average_frames": average_frames,
    "average_finish_rate": finish_rate
}
save_generation(current_generation, stats)




def elitist_selection(fitness_values, marios, num_elites=16):
    # Fitness değerlerini sıralayıp en iyi indeksleri bul
    elite_indices = np.argsort(fitness_values)[-num_elites:][::-1]  # En yüksek fitness'lar
    # Elit Mario'ları seç
    elites = marios[elite_indices]
    # Mario dizisinin ilk num_elites kısmına elitleri koy
    marios[:num_elites] = elites
    return marios

# Örnek kullanım:
fitness_values = np.random.rand(64)  # Fitness değerleri
marios = np.random.rand(64, 5, 5, 5, 5)  # 5 boyutlu array (örnek)

# Elitist seçim
marios = elitist_selection(fitness_values, marios)

# Kontrol
print("Elit Mario'ların fitness değerleri:")
print(fitness_values[np.argsort(fitness_values)[-16:][::-1]])

