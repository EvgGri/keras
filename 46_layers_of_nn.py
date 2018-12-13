# -=-=-=-=-=- Обзор готовых слоев нейронных сетей -=-=-=-=-=-=-=

# Keras предоставляет несколько готовых слоев. Мы рассмотрим наиболее употребительные.

# -=-=-=-=-=- Обычный плотный слой
# Плотная модель – это полносвязный слой нейронной сети. Ниже приведен прототип модели со всеми параметрами:
keras.layers.core.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None, kernel_constraint=None,
                        bias_constraint=None)

# -=-=-=-=-=- Рекуррентные нейронные сети – простая, LSTM и GRU
# Рекуррентные нейронные сети – это класс нейронных сетей, в которых используется последовательная природа входных данных.
# Входными данными может быть текст, речь, временные ряды и вообще любой объект, в котором появление элемента последовательности зависит
# от предшествующих элементов. Далее мы будем обсуждать рекуррентные сети трех видов: простые, LSTM и GRU.
# Ниже приведены прототипы моделей со всеми параметрами:

keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False,stateful=False,unroll=False,implementation=0)

keras.layers.recurrent.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                                 recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
                                 recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                 recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)

keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
                           bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                           bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)

keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
                            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                            recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)

# -=-=-=-=-=- Сверточные и пулинговые слои
# Сверточные сети – класс нейронных сетей, в которых сверточные и пулинговые операции используются для постепенного обучения довольно сложных
# моделей с повышающимся уровнем абстракции. Такой способ обучения напоминает модель человеческого зрения, сложившуюся в результате миллионов
# лет эволюции. Ниже приведены прототипы моделей со всеми параметрами:

keras.layers.convolutional.Conv1D(lters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True,
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

keras.layers.convolutional.Conv2D(lters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1),
                                  activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)

keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')

keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

# =-=-=-=-=-=-=-=-=-=-=- Регуляризация

# Цель регуляризации – предотвратить переобучение. В слоях различных типов имеются параметры регуляризации.
# Ниже приведен список параметров регуляризации, часто используемых в плотных и сверточных модулях.
# 1. kernel_regularizer: функция регуляризации, применяемая к матрице весов;
# 2. bias_regularizer: функция регуляризации, применяемая к вектору смещений;
# 3. activity_regularizer: функция регуляризации, применяе- мая к выходу слоя (его функции активации).

# Кроме того, для регуляризации можно использовать прорежи- вание и зачастую это дает весомый эффект:
# rate – вещественное число от 0 до 1, показывающее, сколько входных блоков отбрасывать;
# noise_shape – одномерный целочисленный тензор, задаю- щий форму двоичной маски прореживания, которая умно- жается на входной сигнал;
# seed – целое число, служащее для инициализации генерато- ра случайных чисел.
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)


# =-=-=-=-=-=-=-=-=-=-=- Пакетная нормировка
# Пакетная нормировка (см. https://www.colwiz.com/cite-in- google-docs/cid=f20f9683aaf69ce) позволяет ускорить обучение и в общем случае
# получить большую верность. Примеры будут рассмотрены далее при обсуждении порождающих состязательных сетей.
# Ниже приведен прототип с параметрами:
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                                              moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                              beta_constraint=None, gamma_constraint=None)


# =-=-=-=-=-=-=-=-=-=-=- Обзор готовых функций активации
# К числу готовых функций активации относятся, в частности, сигмоида, линейная функция, гиперболический тангенс и блок
# линейной ректификации (ReLU).



# =-=-=-=-=-=-=-=-=-=-=- Обзор функций потерь
# Функции потерь (или целевые функции) (см. https://keras.io/ losses/) можно отнести к четырем категориям:
# 1. Верность, используемая в задачах классификации. Таких функций четыре:
#     1.1 binary_accuracy (средняя верность по всем предсказаниям в задачах бинарной классификации)
#     1.2 categorical_ accuracy (средняя верность по всем предсказаниям в задачах многоклассовой классификации)
#     1.3 sparse_categorical_accuracy (используется, когда метки разреженные)
#     1.4 top_k_categorical_ accuracy (успехом считается случай, когда истинный целевой класс находится среди первых top_k предсказаний)
#
# 2. Ошибка, измеряющая различие между предсказанными и фактическими значениями. Варианты таковы:
#     2.1 mse (средне­квадратическая ошибка)
#     2.2 rmse(квадратный корень из средне-квадратической ошибки)
#     2.3 mae (средняя абсолютная ошибка)
#     2.4 mape (средняя ошибка в процентах)
#     2.5 msle (средняя квадратично­логарифмическая ошибка).
#
# 3. Кусочно­линейная функция потерь, которая обычно применяется для обучения классификаторов. Существует два варианта:
#     3.1 кусочно-линейная, определяемая как max(1 – ytrue * ypred, 0) и квадратичная кусочно-линейная, равная квадрату кусоч- но­линейной.
#
# 4. Классовая потеря используется для вычисления перекрестной энтропии в задачах классификации. Существует несколько вариантов:
#     включая бинарную перекрестную энтропию (см. https://en.wikipedia.org/wiki/Cross_entropy) и категориальную перекрестную энтропию.



# =-=-=-=-=-=-=-=-=-=-=- Обзор показателей качества
# Функции показателей качества (см. https://keras.io/metrics/) аналогичны целевым функциям.
# Единственное различие между ними состоит в том, что результаты вычисления показателей не используются на этапе обучения модели.
# Примеры мы обсуждали ранее, а дополнительные будут приведены ниже.



# =-=-=-=-=-=-=-=-=-=-=- Обзор оптимизаторов
# К числу оптимизаторов относятся СГС, RMSprop и Adam. Несколько примеров мы видели ранее, а дополнительные (Adagrad и Adadelta,
# см. https://keras.io/optimizers/) будут приведены в следующих главах.
