#-=-=-=-=-=- Применение регуляризации для предотвращения переобучения
# Интуитивно представляется, что хорошая модель машинного обучения должна давать малую ошибку на обучающих данных.
# Математически это равносильно минимизации построенной моделью функции потерь на обучающих данных.

# Однако этого может оказаться недостаточно. Модель может стать избыточно сложной, стремясь уловить все связи, присущие обучающим данным.
# У такого увеличения сложности есть два нежелательных последствия. Во­первых, для выполнения сложной модели нужно много времени.
# Во­вторых, сложная модель может показывать великолепное качество на обучающих данных – поскольку она запомнила все присутствующие в них связи,
# но гораздо худшее на контрольных – поскольку модель не обобщается на новые данные.
# Таким образом, обучение свелось не к способности к обобщению, а к запоминанию.
# На следующем графике показана типичная функция потерь, которая убывает как на обучающем, так и на контрольном наборе.
# Однако в какой­то момент потеря на контрольных данных начинает расти из­за переобучения.

# Эвристическое правило состоит в том, что если в процессе обучения мы наблюдаем возрастание потери на контрольном наборе после
# первоначального убывания, значит, модель слишком сложна и слишком близко подогнана к обучающим данным.
# В машинном обучении этот феномен называется переобучением.
# Для решения проблемы переобучения необходимо как­то выразить сложность модели и управлять ею.
# Модель по существу – всего лишь вектор весов. Поэтому ее сложность можно представить в виде количества ненулевых весов.
# Иными словами, если две модели M1 и M2 дают примерно одинаковое качество в терминах функции потерь, то следует предпочесть ту,
# в которой количество ненулевых весов меньше.

# В машинном обучении применяются три способа регуляризации:
# 1. Регуляризация по норме L1 (известная также под названием lasso): сложность модели выражается в виде суммы модулей весов.
# 2. Регуляризация по норме L2 (гребневая): сложность модели выражается в виде суммы квадратов весов.
# 3. Эластичная сеть: для выражения сложности модели применяется комбинация двух предыдущих способов.

# Отметим, что идею регуляризации можно применить к весам, к модели и к активации.
# Keras поддерживает все три формы регуляризации. Добавить регуляризацию просто, ниже показано задание L2­регуляризатора ядра
# (вектора весов W).
from keras import regularizers model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.01)))

#-=-=-=-=-=- Настройка гиперпараметров
# Описанные выше эксперименты помогли составить представление о том, какие имеются способы настройки нейросети.
# Однако то, что годится для данного примера, может не подойти в других случаях.
# Для каждой сети имеется много допускающих оптимизацию параметров (количество скрытых нейронов, BATCH_SIZE, количество периодов и ряд других,
# зависящих от сложности сети).
# Настройкой гиперпараметров называется процесс поиска оптимального сочетания этих параметров, при котором достигается
# минимум функции стоимости. Если имеется n параметров, то можно считать, что они определяют n­-мерное пространство, а наша цель – найти
# в этом пространстве точку, в которой функция стоимости принимает минимальное значение.
# Для достижения этой цели можно, например, создать координатную сетку в про- странстве и для каждого ее узла вычислить значение функции стоимости.
# Иными словами, выполнить полный перебор всех комбинаций параметров.

# -=-=-=-=-=- Предсказание выхода
# Обученную сеть естественно использовать для предсказания. В Keras это очень просто:
# вычислить предсказание
predictions = model.predict(X)

# Для заданного входного вектора можно вычислить несколько значений:
model.evaluate() #: вычисляет потерю;
model.predict_classes() #: вычисляет категориальные выходы;
model.predict_proba() #: вычисляет вероятности классов.
