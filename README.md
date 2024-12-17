# Mario_AI
miyav

bu proje var ya efsane olcak


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

------------------------------------------------

geliştirilecek düzeltilecek çok şey var. hepsini yazmıcam ama aklıma gelmişken yazıyım.

mesela ilk 16 marionun fitness'ını zaten öncesinden hesaplıyoruz. biz her jenerasyonda 48 tane hesaplasak yetmeli aslında.

cpu'da yapıyoz yavaş oluyo mesela. ve thread kullanmıyoz yavaş oluyo o yüzden de.

bir sürü dönüşüm gereksiz if ler vesaire vesaire detay birsürü optimizasyon.

python kullanmak da hatalardan birisi.