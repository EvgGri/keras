import pandas as pd
# dataset=pd.read_csv('UDataset_final.csv', sep='\t', header='infer')


import re
import pandas
import warnings

myfile = 'UDataset_final.csv'
target_type = str  # The desired output type

with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")

    dataset = pandas.read_csv(myfile, sep="\t", header='infer')
    print("Warnings raised:", ws)
    # We have an error on specific columns, try and load them as string
    for w in ws:
        s = str(w.message)
        print("Warning message:", s)
        match = re.search(r"Columns \(([0-9,]+)\) have mixed types\.", s)
        if match:
            columns = match.group(1).split(',') # Get columns as a list
            columns = [int(c) for c in columns]
            print("Applying %s dtype to columns:" % target_type, columns)
            dataset.iloc[:,columns] = dataset.iloc[:,columns].astype(target_type)

dataset.head()
