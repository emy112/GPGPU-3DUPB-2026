# Task 1

## 1. Analiza Kernel-urilor CUDA

* **Cate kernel-uri sunt lansate?**  
  Se lanseaza 4 kernel-uri: `renderInit`, `allocateWorld`, `render` si `freeWorld`.

* **Cate blocuri si thread-uri foloseste fiecare kernel?**  
  * `renderInit` si `render`: Folosesc o grila calculata in functie de dimensiunea imaginii si un numar fix de thread-uri per bloc (de exemplu 8x8 = 64 thread-uri per bloc).
  * `allocateWorld` si `freeWorld`: Sunt lansate pe 1 singur bloc cu 1 singur thread (`<<<1, 1>>>`).

* **De ce rulam anumite kernel-uri cu un singur thread?**  
  Alocarea si eliberarea obiectelor din scena (sfere, materiale, camera) sunt operatii secventiale. Daca le-am rula pe mai multe thread-uri simultan, am crea obiecte duplicate si erori de memorie (race conditions).

* **Care este cel mai costisitor kernel si de ce?**  
  Kernel-ul `render` este cel mai costisitor , deoarece executa calculul matematic intensiv de Ray Tracing pentru fiecare pixel in parte, calculand raze, intersectii si reflexii pentru toate sferele din scena.

---

## 2. Comparatie Performanta: CPU vs. GPU

* **Timp CPU (Serial):** 142248 ms (~142.2 secunde)
* **Timp GPU (CUDA):** 2289.1 ms (~2.29 secunde)
* **Accelerare (Speedup):** 142248 / 2289.1 = **62.14x** (versiunea GPU este de ~62 de ori mai rapida).
* **Comparatie cu nucleele GPU:** Accelerarea este de ~62x (si nu de mii de ori) din cauza timpului pierdut cu transferul de memorie, a ramificatiilor din cod (cand razele ricosaza in directii diferite) si a sectiunilor secventiale de initializare.

---

## 3. Sarcina Practica 1: Scalabilitatea Parametrilor

### A. Dublarea sample-urilor per pixel (ns de la 10 la 20)
* **Timp 10 sps (512x512):** 2266.8 ms
* **Timp 20 sps (512x512):** 4442.5 ms
* **Factor de crestere:** 4442.5 / 2266.8 = **1.96x**
* **Explicatie:** Cresterea este liniara (aproape 2x). Fiecare thread executa o bucla de ns ori, deci dublarea sample-urilor dubleaza volumul de munca. Costurile fixe de pornire ale GPU-ului fac ca timpul sa nu fie exact 2.0x.

### B. Dublarea rezolutiei imaginii (de la 512x512 la 1024x1024)
* **Timp 512x512 (10 sps):** 2266.8 ms (262.144 pixeli)
* **Timp 1024x1024 (10 sps):** 8568.7 ms (1.048.576 pixeli)
* **Crestere pixeli:** 4x
* **Factor de crestere timp:** 8568.7 / 2266.8 = **3.78x**
* **Explicatie:** Timpul creste liniar cu numarul de pixeli. Este putin sub 4x datorita optimizarilor hardware ale GPU-ului:
  * **Ocuparea mai buna a GPU-ului:** La rezolutie mare, toate nucleele GPU sunt folosite la capacitate maxima.
  * **Mascarea latentelor (Latency Hiding):** Cand un thread asteapta date din memorie, GPU-ul comuta instant pe altul fara timp mort.
  * **Amortizarea overhead-ului:** Timpul fix de lansare a programului se imparte la un numar de 4 ori mai mare de pixeli.


 # Task2

## 1. Setup si Comanda de Profiling

* **Parametri scena:** Rezolutie 900x600, 10 samples per pixel (sps)
* **Comanda utilizata pentru investigare:**  
  `compute-sanitizer --tool memcheck --leak-check full ./RayTracerAdvancedCUDA.exe`

---

## 2. Comparatie Performanta si Memorie (Stack Size)

* **Implementare Recursiva:**
  * Dimensiune stiva necesara: 4096 bytes (4 KB)
  * Timp de randare: ~9651.7 ms (~9.65 secunde)

* **Implementare Iterativa:**
  * Dimensiune stiva necesara: 4096 bytes (4 KB)
  * Timp de randare: 1724.13 ms (~1.72 secunde)

* **Accelerare (Speedup):**
  Varianta iterativa este de **~5.6 ori mai rapida** decat varianta recursiva (9651.7 ms / 1724.13 ms).

## 3. Analiza si Interpretarea Rezultatelor

* **De ce crapa varianta recursiva pe stiva implicita?**  
  Fiecare apel recursiv genereaza un cadru nou pe stiva locala a fiecarui thread de pe GPU. La un arbore BVH adanc, stiva implicita oferita de CUDA (1 KB) este depasita rapid, cauzand o eroare de tip device-side stack overflow. Marirea stivei la 4 KB (`cudaDeviceSetLimit`) rezolva aceasta problema, dar iroseste memorie VRAM la mii de thread-uri concurente.


 **Rezolvarea problemelor de memorie:**
  * S-au adaugat apelurile `delete[] list;` si `delete[] nodes;` la finalul constructorului `bvh_node` pentru a elibera tablourile temporare din GPU heap.
  * S-a initializat explicit `_last_level = false;` pe nodul radacina pentru a preveni citirea datelor neinitializate.
  * Verificare cu `compute-sanitizer`:


```cmd
C:\Users\emi\Desktop\GPGPU-Nou\Local\build\Debug>compute-sanitizer --tool memcheck --leak-check full RayTracerAdvancedCUDA.exe
========= COMPUTE-SANITIZER
Allocated stack size: 4096 bytes
Successfully rendered 900x600 image with 10 samples per pixel.
Time: 164999ms
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
