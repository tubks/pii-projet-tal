import os
import pandas as pd

data_path = os.path.join('..','data', 'raw', 'train.json')
data = pd.read_json(data_path)
