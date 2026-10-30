Лабораторна робота №4 - приховування даних у частотній області (ДКП)

Файли
  lab4.py      - програма (завдання 1-5)
  1.bmp        - контейнер (BMP24)
  1.txt        - інформаційне повідомлення
  results/     - результати останнього запуску: графіки (.png) та стего-зображення (Stego_*.bmp)
  run_log.txt  - текстовий вивід програми (таблиці Po та RAZ)

Запуск
  pip install numpy pillow matplotlib
  python lab4.py --image 1.bmp --msg 1.txt --quality 75 --pr 5
