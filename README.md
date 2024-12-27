# Mario_AI
meow

Setup
-----
We used Python 3.7. (The reason is a trade secret -_- )
pip 24.0
gym 0.17.3
gym-retro 0.8.0
pillow 9.5.0
pyqt5 5.15.10
numpy 1.21.6

First, since these libraries are needed, run the following command in the terminal.

"pip install -r requirements.txt"
Next, download the .nes file of the game below and place it in a folder.

"https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html"
Then, use a command like the one below to add the game to retro gym's library.

python -m retro.import "C:\Dev\Mario"
You should get a result like this.

Importing SuperMarioBros-Nes
Imported 1 games
That's it.

Usage of the Code
-----------------
To generate the initial population, run chromosome generator. (If two JSON files with the same name are created, you'll need to delete and fix them manually.)

After generating the initial population, run main or main_2, and that's all. (You'll need to manually specify which chromosome and history file to use.)

If you want to generate statistics later, run the statistics source code. (You'll need to manually specify which history file to plot.)

-----------------------------------------------
There are many things to improve and fix. I won't list them all, but I'll mention a few while they're on my mind.

For example, we already calculate the fitness of the first 16 Marios beforehand. Calculating only 48 per generation should be enough.

We're using the CPU, so it's slow. Also, since we're not using threads, that makes it slow too.

There are lots of unnecessary conversions, if statements, etc. So, there's a lot of room for optimization.

Using Python is also one of the mistakes.

------------------------------------------------

# Mario_AI
miyav

Hazırlanış
------------------------------------------------
Python 3.7 kullandık. (Sebebi ticari sırdır -_- )
pip 24.0
gym 0.17.3
gym-retro 0.8.0
pillow 9.5.0
pyqt5 5.15.10
numpy 1.21.6

Öncelikle şu kütpaneler lazım olduğundan şu komut terminale yazılır.
- "pip install -r requirements.txt"

Sonra aşağıdaki oyunun .nes dosyası alınır. Bir dosyanın içine konur.
- "https://wowroms.com/en/roms/nintendo-entertainment-system/super-mario-bros./23755.html"

Alttaki gibi bir komut ile oyunumuz retro gym kütüphanesindeki oyunlara eklenir.
- python -m retro.import "C:\Dev\Mario"

Şunun gibi bir sonuç alınır.
- Importing SuperMarioBros-Nes
- Imported 1 games

Bu kadar.

Kodun Kullanılışı
------------------------------------------------
Başlangıç popülasyonu oluşturmak için chromosome generator çalıştır. (aynı isimle iki tane json oluşturusansilinir elinle düzeltmen lazım.)

Başlangıç popülasyonunu oluşturduktan sonra main veya main_2 çalıştırılır ve bu kadar. (hangi kromozomu ve hangi history dosyasını kullanacağını elinle değiştirmen lazım yine.)

Sonrasında istatistik çıkarmak istersen statistics kaynak kodunu çalıştırırsın. (hangi history dosyasını çizeceğini elinle değiştirmen gerek.)

------------------------------------------------
geliştirilecek düzeltilecek çok şey var. hepsini yazmıcam ama aklıma gelmişken yazıyım.

mesela ilk 16 marionun fitness'ını zaten öncesinden hesaplıyoruz. biz her jenerasyonda 48 tane hesaplasak yetmeli aslında.

cpu'da yapıyoz yavaş oluyo mesela. ve thread kullanmıyoz yavaş oluyo o yüzden de.

bir sürü dönüşüm gereksiz if ler vesaire vesaire detay birsürü optimizasyon.

python kullanmak da hatalardan birisi.

