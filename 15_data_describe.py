# Загружаем данные
from sklearn.datasets import load_wine
data = load_wine()
data.target[[10, 80, 140]]
list(data.target_names)

import pandas as pd
df_wine = pd.DataFrame(data.data)

df_wine.columns=['Алкоголь','Яблочная кислота','Зола',
                 'Щелочность золы','Магний','Всего фенола','Флаваноиды',
                 'Фенолы нефлаваноидные','Проантоцианины','Интенсивность цвета',
                 'Оттенок','OD280 / OD315 разбавленных вин','Пролин']

# Опреляем функцию, которая подсчитывает пропущенные значения
def missing_values(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        uniq_vals = df.nunique()
        min_vals = df.min(skipna=True)
        max_vals = df.max(skipna=True)
        median_vals = df.median(skipna=True)
        mean_vals = df.mean(skipna=True)
        mis_val_table = pd.concat([mis_val, mis_val_percent, uniq_vals, min_vals, max_vals, median_vals, mean_vals], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2 : 'Unique Values',3 : 'Min', 4 : 'Max', 5:'Median', 6: 'Mean'})
        mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values('% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

missing_values(df_wine)
