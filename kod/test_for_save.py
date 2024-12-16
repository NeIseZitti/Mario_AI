import json
import numpy as np

# JSON dosyasını açma
with open('chromosomes_2.json', 'r') as f:
    data = json.load(f)

# Başlangıçta 64 Mario'yu tutacak arrayin özellikleri için
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


print(marios)

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
    with open('chromosomes_2.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print("JSON dosyasına kaydedildi.")

# Kaydetme işlemini test edelim
save_marios_to_json()

# Kaydedilen veriyi doğrulama
def test_save_marios_to_json():
    try:
        # JSON dosyasını okuyalım
        with open('chromosomes_2.json', 'r') as f:
            saved_data = json.load(f)
        
        # İlk 16 Mario'nun verilerini kontrol edelim
        for i in range(16):
            mario_key = f'mario_{i+1}'
            if mario_key not in saved_data:
                print(f"Hata: {mario_key} bulunamadı.")
                return
            
            mario_data = saved_data[mario_key]
            # Ağırlıkların ve biasların doğru formatta kaydedildiğini kontrol edelim
            if 'weights' not in mario_data or 'biases' not in mario_data:
                print(f"Hata: {mario_key} verisi eksik.")
                return
            weights = mario_data['weights']
            biases = mario_data['biases']
            
            # Ağırlıkların boyutları doğru mu?
            if len(weights['input_to_hidden1']) != 70 or len(weights['hidden1_to_hidden2']) != 64 or len(weights['hidden2_to_output']) != 32:
                print(f"Hata: {mario_key} ağırlık boyutları hatalı.")
                return
            if len(biases['hidden1_bias']) != 64 or len(biases['hidden2_bias']) != 32 or len(biases['output_bias']) != 6:
                print(f"Hata: {mario_key} bias boyutları hatalı.")
                return

        print("Veriler başarıyla kaydedildi ve doğrulandı!")
    except Exception as e:
        print(f"Test sırasında hata oluştu: {e}")

# Testi çalıştıralım
test_save_marios_to_json()
