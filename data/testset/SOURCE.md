# Тестовый набор

30 фраз (15 мужских голосов `m01`-`m15`, 15 женских `f01`-`f15`) из тестовой части
русского раздела корпуса **FLEURS** (Google).

- Источник: https://huggingface.co/datasets/google/fleurs (`data/ru_ru`, test)
- Лицензия: Creative Commons Attribution 4.0 (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Авторы: A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera,
  A. Bapna. "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", 2022.

## Изменения относительно оригинала

- отобраны фразы длиной 4-10 с, по одной на предложение;
- частота дискретизации понижена с 16 до 8 кГц, моно, 16 бит;
- громкость: каждая запись приведена к RMS -20 dBFS, но не выше пика -1 dBFS.

`manifest.tsv` - соответствие файлов исходным записям FLEURS (столбец `source`) и тексты фраз.
