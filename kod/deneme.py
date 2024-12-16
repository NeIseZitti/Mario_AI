# allah rızası için programı ctrl+c ile kapatın (visual studio code kullanıyorsanız) (terminali kapayınca çalışmıyo) (bu artık önemli değil amk)
import retro

import utils

import numpy as np

import os  # Konsolu temizlemek için

import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# import atexit

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


# Değerleri normalize etmek için bir min-max scaling fonksiyonu tanımlayalım.
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class MarioNN(nn.Module):
    def __init__(self):
        super(MarioNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_1)  # input_size = 70, hidden_layer_1 = 64
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)  # hidden_layer_1 = 64, hidden_layer_2 = 32
        self.fc3 = nn.Linear(hidden_layer_2, output_size)  # hidden_layer_2 = 32, output_size = 6

    def forward(self, x, mario_id=0):
        """Herhangi bir Mario'ya ait ağırlıkları ve biasları uygula."""
        self.fc1.weight.data = torch.tensor(np.array(marios[mario_id][0][0]).T, dtype=torch.float32)  # Ağırlıkları numpy array'e çevirip transpozisyon
        self.fc1.bias.data = torch.tensor(marios[mario_id][1][0], dtype=torch.float32)  # Mario'nun hidden1 bias'ları
        self.fc2.weight.data = torch.tensor(np.array(marios[mario_id][0][1]).T, dtype=torch.float32)  # Transpozisyon
        self.fc2.bias.data = torch.tensor(marios[mario_id][1][1], dtype=torch.float32)  # Mario'nun hidden2 bias'ları
        self.fc3.weight.data = torch.tensor(np.array(marios[mario_id][0][2]).T, dtype=torch.float32)  # Transpozisyon
        self.fc3.bias.data = torch.tensor(marios[mario_id][1][2], dtype=torch.float32)  # Mario'nun output bias'ları

        # bunlar test amaçlı bizim json dosyasındaki weightler ve biasler gerçekten modele işlemiş mi diye bakıyoz!!!!
        # print(self.fc1.weight)
        # print(self.fc1.bias)
        # print(self.fc2.weight)
        # print(self.fc2.bias)
        # print(self.fc3.weight)
        # print(self.fc3.bias)

        x = F.relu(self.fc1(x))  # İlk katmandan geçiş
        x = F.relu(self.fc2(x))  # İkinci katmandan geçiş
        x = torch.sigmoid(self.fc3(x))  # Çıktıya ulaşma
        return x



def output_to_action(output, threshold=0.5):
    # Sinir ağından gelen outputu, eşik değerine göre hangi tuşlara basılacağını belirleyen 0 ve 1'lere dönüştür
    action = [1 if x > threshold else 0 for x in output]
    return action

def action_to_buttons(action):
    # 6 uzunluğundaki action listesini 9 uzunluğunda bir listeye dönüştür
    buttons = [0] * 9  # Başlangıçta 9 uzunluğunda 0'lar ile bir liste oluştur
    
    # Action'daki her 1'i, doğru indekse yerleştir
    button_indices = [0, 4, 5, 6, 7, 8]  # B, UP, DOWN, LEFT, RIGHT, A'nın sırası
    
    for i in range(len(action)):
        if action[i] == 1:
            buttons[button_indices[i]] = 1  # action'daki 1'i doğru tuşa yerleştir
    
    return buttons


# fitness fonksiyonumuz bu olsun
def fitness1(frames, distance, did_win):
    return max(distance ** 2 - \
            frames ** 1.2 +   \
            min(max(distance-50, 0), 1) * 100 + \
            did_win * 1e5, 0.00001)
# bu herhangi bi bireyin son noktadaki puanı olcak
# buna göre aralarından seçim yapıp yeni jenerasyon üreticez

# bu da alternatif ama çok hoşuma gitmedi
def fitness2(frames, distance, did_win):
    return max(np.log1p(distance) * 1000 - \
               frames ** 1.2 + \
               max(distance - 50, 0) * 100 + \
               did_win * 1e5, 0.00001)


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


# # Program kapandığında çalışacak fonksiyon, bu fonksiyon mariolarımızın kromozomlarını kaydediyo
# def save_marios_to_json():
#     # İlk 16 Mario'yu alıyoruz
#     best_marios = marios[:16]
    
#     # 16 Mario'nun ağırlıkları ve bias'larını JSON formatında hazırlıyoruz
#     data = {}
#     for i, mario in enumerate(best_marios):
#         mario_data = {
#             'weights': {
#                 'input_to_hidden1': mario[0][0].tolist(),
#                 'hidden1_to_hidden2': mario[0][1].tolist(),
#                 'hidden2_to_output': mario[0][2].tolist()
#             },
#             'biases': {
#                 'hidden1_bias': mario[1][0].tolist(),
#                 'hidden2_bias': mario[1][1].tolist(),
#                 'output_bias': mario[1][2].tolist()
#             }
#         }
#         data[f'mario_{i+1}'] = mario_data
    
#     # Verileri JSON dosyasına kaydediyoruz
#     with open('chromosomes.json', 'w') as f:
#         json.dump(data, f, indent=4)
    
#     print("\n program sonlandı ve kromozomlar güncellendi.")


# save_marios_to_json fonksiyonunu yazalım
def save_marios_to_json():
    best_marios = marios[:16]  # İlk 16 Mario
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
    
    # Verileri JSON dosyasına kaydediyoruz
    with open('chromosomes.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print("JSON dosyasına kaydedildi.")


# tanımların sonu

# tek sefer çalışacak kodlar
# atexit.register(save_marios_to_json) # çalışmadı amk

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

# Modeli başlat
model = MarioNN()

# JSON dosyasının adı
history_json = "history.json"

# Program başlatıldığında jenerasyon sayısını yükle
current_generation = load_generation()

# tüm programın döngüsü // veya jenerasyonların döngüsü
while True:
    print(f"Jenerasyon {current_generation+1} başlatılıyor...")
    print()

    # İlk 16 Mario’dan 64 Mario’ya çıkış
    for i in range(16, 64):
        # Rastgele iki ebeveyn seç
        parent1 = random.choice(marios[:16])
        parent2 = random.choice(marios[:16])
        
        # Çocuk oluştur ve mevcut marios array'ine ekle
        marios[i] = crossover(parent1, parent2)


    # Yeni Mario'lara mutasyon uygulama
    for i in range(16, 64):  # İlk 16 Mario'yu koruyoruz, geri kalan 48'e mutasyon uyguluyoruz
        marios[i] = mutate(marios[i])
    

    fitness_values = [0] * 64
    distance_values = [0] * 64
    frame_values = [0] * 64
    finish_rate_values = [0] * 64

    average_fitness = 0
    average_distance = 0
    average_frames = 0
    finish_rate = 0

    current_generation += 1

    # 1 popülasyonun (jenerasyonun döngüsü)
    for mario_id in range(64):
        prev_lives = 2
        frames = 0
        did_win = False


        env.reset()  # Her Mario için oyunu sıfırlıyoruz
        
        print(f"Mario {mario_id} başlatılıyor...")

        # 1 tane marionun döngüsü
        while True:
            # env.render()  # Oyun ekranını gösterir.
            ram = env.get_ram()

            # 'get_tiles' fonksiyonu ile RAM'den tile verilerini alıyoruz
            values = utils.SMB.get_tiles(ram).values()
            # Her bir tile'ın value özelliğini alarak yeni bir liste oluşturuyoruz
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

            # 0, 1, 170 ve 255 değerlerini normalize edelim.
            normalized_input = [normalize(v, 0, 255) for v in input]


            # normalized_input'ı PyTorch Tensor'una çevir
            normalized_input_tensor = torch.tensor(normalized_input, dtype=torch.float32)

            # Modeli çağırırken Tensor verisi gönderiyoruz
            output = model(normalized_input_tensor, mario_id=mario_id)
            action = output_to_action(output)
            buttons = action_to_buttons(action)

            # Aksiyonla oyunda bir adım atıyoruz
            obs, reward, done, info = env.step(buttons)

            frames+=1

            # Oyun içinde her frame'de can sayısını önceki frame'dekiyle kontrol edelim
            # ram[0x001d] == 0x03 bu marionun flagpole animasyonununa girip girmediğini söylüyo
            did_win = ram[0x001d] == 0x03
            if info['lives'] < prev_lives or did_win:
                # print(f"Mario {mario_id} oyunu bitti.")
                
                distance = utils.SMB.get_mario_location_in_level(ram).x
                # print("distance:", distance)
                # print("frames:", frames)
                
                finish_rate_values[mario_id] = 0
                distance_values[mario_id] = distance
                frame_values[mario_id] = frames

                if did_win:
                    finish_rate_values[mario_id] = 1
                    frame_values[mario_id] = 9821 # bu değer erkenden birilerinin ölüp grafiği kötü göstermesin diye

                # print("did win:", did_win)

                # print("fitness:", fitness1(frames, distance, did_win))
                
                fitness_values[mario_id] = fitness1(frames, distance, did_win)

                
                break  # Oyunu bitiririz



            # Önceki framedeki lives değişkeni artık şu anki lives değişkeni olacak
            prev_lives = info['lives']

        # print(f"Mario {mario_id} oyunu tamamladı.")



    # istatistikleri öncelikle bi tutalım.
    average_fitness = sum(fitness_values) / 64
    print(f"Average Fitness: {average_fitness}")
    average_distance = sum(distance_values) / 64
    print(f"Average Distance: {average_distance}")
    average_frames = sum(frame_values) / 64
    print(f"Average Frames: {average_frames}")
    finish_rate = sum(finish_rate_values) / 64
    print(f"Finish Rate: {finish_rate:.2%}")

    # Yeni jenerasyon istatistiklerini kaydediyoz. hata oluşmasın diye de try except kullandık.
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


    # döngünün sonraki adımına geçmeden önce bi selection algorithm lazım
    marios = elitist_selection(fitness_values, marios)

    # Marios array'ini elitist seçme işlemi sonrası kaydet
    save_marios_to_json()  # Her jenerasyonun sonunda manuel olarak kaydediyoruz

    print()
    print(f"Jenerasyon {current_generation} sonlandı.")

# program döngüsünün sonu



# Traceback (most recent call last):
#   File "c:\Dev\Mario\kod\deneme.py", line 422, in <module>

#   File "c:\Dev\Mario\kod\deneme.py", line 239, in save_marios_to_json       
#     'hidden1_to_hidden2': mario[0][1].tolist(),
# AttributeError: 'list' object has no attribute 'tolist'

# böyle bi hata veriyo amk

env.close()






# bu da test amaçlıydı bunun amacıysa modelin gerçekten bize tuş takımlarını verip vermediğine bakmak ve weight ile biaslerin düzgün atanıp atanmadığına bakmak
# tabi bi de farklı weightler için farklı çıktı veriyo mu diye de test edilebilir.

# data = [
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00392156862745098,
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
#     0.00392156862745098, 0.00392156862745098, 0.00392156862745098, 0.00392156862745098, 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
# ]

# sample_input = torch.tensor(data, dtype=torch.float32)

# # Modeli çağırma
# output = model(sample_input, mario_id=1)  # mario_id 0 ile çağrıldı
# print(output)

# print(marios[1][0][0][0])

# action = output_to_action(output)
# print(action)
# # butonların 9 uzunluğundaki yerleri
# # ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
# buttons = action_to_buttons(action)
# print(buttons)





# # 1 tane marionun döngüsü
# prev_lives = 2
# env.reset()
# while True:
    
#     env.render()  # Oyun ekranını gösterir.
#     ram = env.get_ram()

#     # 'get_tiles' fonksiyonu ile RAM'den tile verilerini alıyoruz
#     values = utils.SMB.get_tiles(ram).values()
#     # Her bir tile'ın value özelliğini alarak yeni bir liste oluşturuyoruz
#     values = [tile.value for tile in values]

#     input = []

#     if(utils.SMB.get_mario_row_col(ram)[1] < 10):
#         for j in range(10):
#             for k in range(7):
#                 input.append(values[(4+j)*16 + k + utils.SMB.get_mario_row_col(ram)[1]])
#     else:
#         for j in range(10):
#             for k in range(7):
#                 input.append(values[(4+j)*16 + k + 8])

#     # 0, 1, 170 ve 255 değerlerini normalize edelim.
#     normalized_input = [normalize(v, 0, 255) for v in input]

#     output = model(sample_input, mario_id=1)
#     print(output)
#     action = output_to_action(output)
#     print(action)
#     buttons = action_to_buttons(action)
#     print(buttons)

#     obs, reward, done, info = env.step(buttons)

#     # Oyun içinde her frame'de can sayısını önceki frame'dekiyle kontrol edelim
#     if info['lives'] < prev_lives or done:
#         print("Mario'nun canı azaldı! Oyun bitiriliyor...")
#         print(info)
#         break  # Oyunu bitiririz
        
    
#     # Önceki framedeki lives değişkeni artık şu anki lives değişkeni olacak
#     prev_lives = info['lives']


# env.close()



# # Sağ tuşuna basmayı temsil eden aksiyon dizisi
# right_action = [0, 0, 0, 0, 0, 0, 0, 1, 0]

# # Sağ ve zıplama tuşlarına basmayı temsil eden aksiyon dizisi
# right_and_jump_action = [0, 0, 0, 0, 0, 0, 0, 1, 1]

# prev_lives = 2 # bu değişken marionun ölüp ölmediğini anlamak için gerekli iki frame arasında
#                # marionun canının azalıp azalmadığını kontrol ediyo.
# for i in range(10000):

#     env.render()  # Oyun ekranını gösterir.
#     ram = env.get_ram()

#     # 'get_tiles' fonksiyonu ile RAM'den tile verilerini alıyoruz
#     values = utils.SMB.get_tiles(ram).values()
#     # Her bir tile'ın value özelliğini alarak yeni bir liste oluşturuyoruz
#     values = [tile.value for tile in values]

#     # 7x10 luk input şu an ama 8x10'a geçmek istiyom. onu sonra burda değiştiririz.
#     # bi de mario hep kendi sütununu görsün istiyom bu inputu ona göre almak lazım
#     input = []

#     if(utils.SMB.get_mario_row_col(ram)[1] < 10):
#         for j in range(10):
#             for k in range(7):
#                 input.append(values[(4+j)*16 + k + utils.SMB.get_mario_row_col(ram)[1]])
#     else:
#         for j in range(10):
#             for k in range(7):
#                 input.append(values[(4+j)*16 + k + 8])

#     # 0, 1, 170 ve 255 değerlerini normalize edelim.
#     normalized_input = [normalize(v, 0, 255) for v in input]

#     if(i%20 == 0):
#         # print("\033[H\033[J", end="")  # Konsolu temizler ve imleci başa alır

#         os.system('cls' if os.name == 'nt' else 'clear')  # Konsolu temizle

#         # 15 satır ve 16 sütunluk bir matris gibi yazdırıyoruz
#         for i in range(15):  # 15 satır
#             for j in range(16):  # 16 sütun
#                 print(values[i*16 + j], end=" ")  # Her satırdaki 16 elemanı yazdır
#             print()  # Yeni satıra geç

#         print("input 7x10")
#         for i in range(10):
#             for j in range(7):
#                 print(input[7*i+j], end=" ")
#             print()

#         print("normalized input 7x10")
#         for i in range(10):
#             for j in range(7):
#                 print(normalized_input[7*i+j], end=" ")
#             print()


#     # Her 25 framede bir elini zıplamadan çek
#     if i % 25 > 1:
#         action = right_and_jump_action
#     else:
#         action = right_action


#     # Sağ hareketini uygula
#     obs, reward, done, info = env.step(action)

#     # Oyun içinde her frame'de can sayısını önceki frame'dekiyle kontrol edelim
#     if info['lives'] < prev_lives:
#         print("Mario'nun canı azaldı! Oyun bitiriliyor...")
#         break  # Oyunu bitiririz
    
#     # Önceki framedeki lives değişkeni artık şu anki lives değişkeni olacak
#     prev_lives = info['lives']


# env.close()


# prev_lives = 2
# for i in range(10000):
    
#     env.render()  # Oyun ekranını gösterir.
#     ram = env.get_ram()

#     # 'get_tiles' fonksiyonu ile RAM'den tile verilerini alıyoruz
#     values = utils.SMB.get_tiles(ram).values()
#     # Her bir tile'ın value özelliğini alarak yeni bir liste oluşturuyoruz
#     values = [tile.value for tile in values]

#     # 7x10 luk input şu an ama 8x10'a geçmek istiyom. onu sonra burda değiştiririz.
#     # bi de mario hep kendi sütununu görsün istiyom bu inputu ona göre almak lazım
#     input = []

#     if(utils.SMB.get_mario_row_col(ram)[1] < 10):
#         for j in range(10):
#             for k in range(7):
#                 input.append(values[(4+j)*16 + k + utils.SMB.get_mario_row_col(ram)[1]])
#     else:
#         for j in range(10):
#             for k in range(7):
#                 input.append(values[(4+j)*16 + k + 8])

#     # 0, 1, 170 ve 255 değerlerini normalize edelim.
#     normalized_input = [normalize(v, 0, 255) for v in input]
    

#     if(i%20 == 0):
#         # print("\033[H\033[J", end="")  # Konsolu temizler ve imleci başa alır

#         os.system('cls' if os.name == 'nt' else 'clear')  # Konsolu temizle

#         # 15 satır ve 16 sütunluk bir matris gibi yazdırıyoruz
#         for i in range(15):  # 15 satır
#             for j in range(16):  # 16 sütun
#                 print(values[i*16 + j], end=" ")  # Her satırdaki 16 elemanı yazdır
#             print()  # Yeni satıra geç
    

#         print("input 7x10")
#         for i in range(10):
#             for j in range(7):
#                 print(input[7*i+j], end=" ")
#             print()


#         print("normalized input 7x10")
#         for i in range(10):
#             for j in range(7):
#                 print(normalized_input[7*i+j], end=" ")
#             print()

#         print(utils.SMB.get_mario_row_col(ram)[1])



  
#     # Sağ hareketini uygula
#     obs, reward, done, info = env.step(action)
    
#     # Oyun içinde her frame'de can sayısını önceki frame'dekiyle kontrol edelim
#     if info['lives'] < prev_lives:
#         print("Mario'nun canı azaldı! Oyun bitiriliyor...")
#         break  # Oyunu bitiririz
    
#     # Önceki framedeki lives değişkeni artık şu anki lives değişkenş olacak
#     prev_lives = info['lives']
    

# env.close()

# print(utils.SMB.get_mario_row_col(ram)) # bu marionun hangi indexte olduğunu dönüyor.
# print(values)



# lazım olacak girdiler.
# bloklar
# belki marionun lokasyonu


# print(0x6E)

# print("miyav")
# enemies = []
# enemies = [None for _ in range(5)]
# for i in range(5):
#     print(enemies[i])

# print(0x100)


# print(utils.SMB.get_mario_location_in_level(ram))