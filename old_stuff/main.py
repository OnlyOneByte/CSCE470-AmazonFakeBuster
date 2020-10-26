import pandas as pd
from enum import IntEnum

#read in the data
class Labels(IntEnum):
	label1 = 0
	label2 = 1

class Dataset:
    def __init__(self):
        self.df = None
        self.read_dataset()

    def read_dataset(self):
        self.df=pd.read_csv("reviews.txt",delimiter="\t")
        # print(df.columns.values.tolist())
        # print(df.head())
        return self.df

    def fetch(self):
        return self.df









