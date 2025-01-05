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

# Matplotlib siyah arka plan stili
plt.style.use("dark_background")

# Ortalama fitness grafiği
plt.figure(figsize=(10, 5))
plt.plot(generations, average_fitness, color="cyan", marker="o")
plt.title("Average Fitness Over Generations", color="white")
plt.xlabel("Generation", color="white")
plt.ylabel("Average Fitness", color="white")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# Ortalama mesafe grafiği
plt.figure(figsize=(10, 5))
plt.plot(generations, average_distance, color="lime", marker="o")
plt.title("Average Distance Over Generations", color="white")
plt.xlabel("Generation", color="white")
plt.ylabel("Average Distance", color="white")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()