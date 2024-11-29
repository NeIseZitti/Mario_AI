import retro

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
env.reset()
ram = env.get_ram()

for i in range(1):
    env.render()  # Oyun ekranını gösterir.



print(0x1A)