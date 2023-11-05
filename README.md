# Summarization-text

Summarization-Model с помощью библиотек transformers, tensorflow, numpy и делаем с помощью подготовленной модели t5-small.

Сервер не был загружен, поскольку загрузка модели объемом 362 МБ заняла слишком много времени.

Сначала я хотел сделать это через RNN-lstm, после поиска в Google я понял, что RNN И CNN - не очень хорошая идея для решения моделей seq2seq. Они обучаются последовательной обработке: предложения должны обрабатываться слово за словом. Это неэффективно и требует много времени для обдумывания более крупных последовательностей и обучаются рекурсивно.

Во время тренинга было допущено много ошибок из-за непонимания причин. Первая ошибка связана с (нехваткой памяти) из-за токенизации в моделях, первая причина заключается в том, что токенизация ставит max_len= 10 ^ 36 и появляется ошибка OOM. Вторая причина связана с отсутствием гиперпараметра padding="max_length" (не забывайте об этом). Вторая ошибка, медленная токенизация, решение: всегда загружайте transformers[sentencepiece].

![Uploading msg1031002797-51582.jpg…]()
