# Настроим простую базу SQLite для аккумулирования дополнительной информации с отзывами о выполненных прогнозах от пользователей веб-приложений.
# Мы можем использовать эту обратную связь для обновления нашей модели классификации. СУБД SQLite представляет собой общедоступное ядро
# базы данных SQL, не требующее работы отдельного сервера, что делает ее идеальным вариантом для малых проектов и простых веб-приложений.
# В стандартной библиотеке Python уже имеется программный интерфейс API sqlite3 для работы с БД SQLite.

# Выполнив следующий код, мы создадим в каталоге movieclassifier новую базу данных SQLite и сохраним два киноотзыва в качестве примера:
import sqlite3
import os

# Создаем соединение conn с файлом базы данных SQLite
conn = sqlite3.connect('reviews.sqlite')
# При помощи метода cursor мы создали курсор, позволяющий перемещаться по записям внутри БД
c = conn.cursor()
# Создаем таблицу внутри БД, в таблице задаем следующую структуру: отзывы, мнениеб дата.
c.execute('CREATE TABLE review_db' '(review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'I love this movie'
c.execute("INSERT INTO review_db" "(review, sentiment, date) VALUES" "(?,?,DATETIME('now'))", (example1,1))

example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db" "(review, sentiment, date) VALUES" "(?,?,DATETIME('now'))", (example2,0))

# Записывает изменения в базу данных
conn.commit()
# Закрывает подключение к базе данных
conn.close()

# Проверим, что элементы верно сохранены в таблице базы данных
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date BETWEEN '2015-01-01 00:00:00' AND DATETIME('now')")

results = c.fetchall()
conn.close()
print(results)
