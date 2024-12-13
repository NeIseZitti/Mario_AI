
# bu dosya sadece sadece başlangıçta 1 defaya mahsus çalıştırılır.
# eğer istersek başka bağımsız popülasyonlar oluşturmak için de kullanılır.
# 70 64 32 6

import numpy as np
import json

# Neural Network için input, hidden layer ve output boyutları
input_size = 70  # Girdiler
hidden_layer_1 = 64  # 1. katman
hidden_layer_2 = 32  # 2. katman
output_size = 6  # Çıktı (aksiyonlar)

# 16 Mario için ağırlıklar ve biaslar
num_marios = 16

# Her Mario'nun ağırlıkları ve biasları
mario_params = {}

# Rastgele ağırlıklar ve biaslar oluşturma
for mario_id in range(num_marios):
    weights = {
        'input_to_hidden1': np.random.uniform(-1, 1, (input_size, hidden_layer_1)).tolist(),
        'hidden1_to_hidden2': np.random.uniform(-1, 1, (hidden_layer_1, hidden_layer_2)).tolist(),
        'hidden2_to_output': np.random.uniform(-1, 1, (hidden_layer_2, output_size)).tolist(),
    }
    
    biases = {
        'hidden1_bias': np.random.uniform(-1, 1, hidden_layer_1).tolist(),
        'hidden2_bias': np.random.uniform(-1, 1, hidden_layer_2).tolist(),
        'output_bias': np.random.uniform(-1, 1, output_size).tolist(),
    }
    
    # Mario'nun ağırlık ve biaslarını kaydet
    mario_params[f'mario_{mario_id+1}'] = {'weights': weights, 'biases': biases}

# JSON dosyasına kaydetme
with open('chromosomes_2.json', 'w') as json_file:
    json.dump(mario_params, json_file, indent=4)

print("Kromozomlar başarıyla oluşturuldu.")