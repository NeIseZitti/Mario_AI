import retro
import utils
import numpy as np
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import atexit

# Başlangıçta 64 Mario'yu tutacak arrayin özellikleri için
num_marios = 64  # 64 Mario
input_size = 70  # Girdiler
hidden_layer_1 = 64  # 1. katman
hidden_layer_2 = 32  # 2. katman
output_size = 6  # Çıktı

# JSON dosyasını açma
with open('chromosomes.json', 'r') as f:
    data = json.load(f)

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


# 16 marioyu 64 marioya çıkarmak için kullanacağımız crossover fonksiyonu
def crossover(parent1, parent2):
    """İki ebeveynin ağırlık ve biaslarını rastgele birleştirerek yeni bir çocuk oluşturur."""
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

# weight ve bias değerlerini belirli miktar değiştirecek mutasyon fonksiyonu
def mutate(mario, mutation_rate=0.05):
    """
    Mario'nun ağırlık ve biaslarına ±%5 mutasyon uygular.
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

# Değerleri normalize etmek için bir min-max scaling fonksiyonu tanımlayalım.
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class MarioNN(nn.Module):
    def __init__(self):
        super(MarioNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, output_size)

    def forward(self, x, mario_id=0):
        """Herhangi bir Mario'ya ait ağırlıkları ve biasları uygula."""
        
        # Ağırlıkları ve biasları NumPy array'e dönüştürüp transpozisyon yapıyoruz
        weights1 = np.array(marios[mario_id][0][0])  # NumPy array'e dönüştür
        self.fc1.weight.data = torch.tensor(weights1.T, dtype=torch.float32)  # Torch tensor'a çevir
        self.fc1.bias.data = torch.tensor(marios[mario_id][1][0], dtype=torch.float32)
        
        weights2 = np.array(marios[mario_id][0][1])  # NumPy array'e dönüştür
        self.fc2.weight.data = torch.tensor(weights2.T, dtype=torch.float32)  # Torch tensor'a çevir
        self.fc2.bias.data = torch.tensor(marios[mario_id][1][1], dtype=torch.float32)
        
        weights3 = np.array(marios[mario_id][0][2])  # NumPy array'e dönüştür
        self.fc3.weight.data = torch.tensor(weights3.T, dtype=torch.float32)  # Torch tensor'a çevir
        self.fc3.bias.data = torch.tensor(marios[mario_id][1][2], dtype=torch.float32)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def output_to_action(output, threshold=0.5):
    action = [1 if x > threshold else 0 for x in output]
    return action

def action_to_buttons(action):
    buttons = [0] * 9  # Başlangıçta 9 uzunluğunda 0'lar ile bir liste oluştur
    button_indices = [0, 4, 5, 6, 7, 8]  # B, UP, DOWN, LEFT, RIGHT, A'nın sırası
    for i in range(len(action)):
        if action[i] == 1:
            buttons[button_indices[i]] = 1
    return buttons


def fitness1(frames, distance, did_win):
    return max(distance ** 2 - frames ** 1.2 + min(max(distance-50, 0), 1) * 100 + did_win * 1e5, 0.00001)

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


def elitist_selection(fitness_values, marios, num_elites=16):
    fitness_values = np.array(fitness_values)
    # Fitness değerlerini sıralayıp en iyi indeksleri bul
    elite_indices = np.argsort(fitness_values)[-num_elites:][::-1]  # En yüksek fitness'lar
    # Elit Mario'ları seç
    elites = marios[elite_indices]
    # Mario dizisinin ilk num_elites kısmına elitleri koy
    marios[:num_elites] = elites
    return marios



# Jenerasyonun sonunda çalışacak ve `marios` verilerini kaydedecek fonksiyon
def save_marios_to_json():
    best_marios = marios[:16]
    data = {}
    for i, mario in enumerate(best_marios):
        mario_data = {
            'weights': {
                'input_to_hidden1': np.array(mario[0][0]).tolist(),
                'hidden1_to_hidden2': np.array(mario[0][1]).tolist(),
                'hidden2_to_output': np.array(mario[0][2]).tolist()
            },
            'biases': {
                'hidden1_bias': np.array(mario[1][0]).tolist(),
                'hidden2_bias': np.array(mario[1][1]).tolist(),
                'output_bias': np.array(mario[1][2]).tolist()
            }
        }
        data[f'mario_{i+1}'] = mario_data
    
    with open('chromosomes.json', 'w') as f:
        json.dump(data, f, indent=4)

atexit.register(save_marios_to_json)

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

# Modeli başlat
model = MarioNN()

# JSON dosyasının adı
history_json = "history.json"

# Program başlatıldığında jenerasyon sayısını yükle
current_generation = load_generation()



# Programın ana döngüsü
while True:
    print(f"Jenerasyon {current_generation} başlatılıyor...")

    # İlk 16 Mario’dan 64 Mario’ya çıkış
    for i in range(16, 64):
        parent1 = random.choice(marios[:16])
        parent2 = random.choice(marios[:16])
        marios[i] = crossover(parent1, parent2)

    for i in range(16, 64):
        marios[i] = mutate(marios[i])

    fitness_values = [0] * 64
    distance_values = [0] * 64
    frame_values = [0] * 64
    finish_rate_values = [0] * 64

    # Jenerasyon hesaplaması ve seçim işlemi
    current_generation += 1
    for mario_id in range(64):
        prev_lives = 2
        frames = 0
        did_win = False
        env.reset()

        while True:
            ram = env.get_ram()
            values = utils.SMB.get_tiles(ram).values()
            values = [tile.value for tile in values]
            input = []

            if utils.SMB.get_mario_row_col(ram)[1] < 10:
                for j in range(10):
                    for k in range(7):
                        input.append(values[(4 + j) * 16 + k + utils.SMB.get_mario_row_col(ram)[1]])
            else:
                for j in range(10):
                    for k in range(7):
                        input.append(values[(4 + j) * 16 + k + 8])

            normalized_input = [normalize(v, 0, 255) for v in input]
            normalized_input_tensor = torch.tensor(normalized_input, dtype=torch.float32)
            output = model(normalized_input_tensor, mario_id=mario_id)
            action = output_to_action(output)
            buttons = action_to_buttons(action)
            obs, reward, done, info = env.step(buttons)
            frames += 1

            did_win = ram[0x001d] == 0x03
            if info['lives'] < prev_lives or did_win:
                distance = utils.SMB.get_mario_location_in_level(ram).x
                finish_rate_values[mario_id] = 0
                distance_values[mario_id] = distance
                frame_values[mario_id] = frames

                if did_win:
                    finish_rate_values[mario_id] = 1
                    frame_values[mario_id] = 9821

                fitness_values[mario_id] = fitness1(frames, distance, did_win)
                break

            prev_lives = info['lives']

    # İstatistikler
    average_fitness = sum(fitness_values) / 64
    average_distance = sum(distance_values) / 64
    average_frames = sum(frame_values) / 64
    finish_rate = sum(finish_rate_values) / 64

    stats = {
        "average_fitness": average_fitness,
        "average_distance": average_distance,
        "average_frames": average_frames,
        "average_finish_rate": finish_rate
    }

    try:
        save_generation(current_generation, stats)
    except Exception as e:
        print(f"JSON kaydedilirken hata oluştu: {e}")

    marios = elitist_selection(fitness_values, marios)

    print(f"Jenerasyon {current_generation} sonlandı.")
