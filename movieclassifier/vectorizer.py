# Нам не нужно консервировать хэширующий векторизатор HashingVectorizer, поскольку он не требует выполнения подгонки. Вместо этого можно
# создать новый сценарный файл Python, из которого можно импортировать векторизатор в наш текущий сеанс Python.
from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(os.path.realpath('__file__'))
# Костыль, нужно понять, как правильно задавать путь
cur_dir +='/movieclassifier/'
print(cur_dir)

stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))
