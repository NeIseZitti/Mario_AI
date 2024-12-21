import json
import matplotlib.pyplot as plt

# history.json dosyasını yükle
with open("history.json", "r") as file:
    data = json.load(file)

# Nesilleri ve değerleri saklamak için listeler oluştur
generations = []
average_fitness = []
average_distance = []

# JSON verilerini işleyerek gerekli bilgileri al
for gen, stats in data["stats"].items():
    generations.append(int(gen.split("_")[1]))  # 'generation_1' -> 1
    average_fitness.append(stats["average_fitness"])
    average_distance.append(stats["average_distance"])

# Ortalama fitness grafiği
plt.figure(figsize=(10, 5))
plt.plot(generations, average_fitness, color="blue", marker="o")
plt.title("Average Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.grid()
plt.tight_layout()
plt.show()

# Ortalama mesafe grafiği
plt.figure(figsize=(10, 5))
plt.plot(generations, average_distance, color="green", marker="s")
plt.title("Average Distance Over Generations")
plt.xlabel("Generation")
plt.ylabel("Average Distance")
plt.grid()
plt.tight_layout()
plt.show()