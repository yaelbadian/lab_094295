import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
sns.set()

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 1000

file_path = r"C:\Users\YaelBadian\Documents\technion\lab\train.tsv"
data = pd.read_csv(file_path, sep="\t")


def json_columns_handler(data, genders=False, jobs=False):
    features = {}
    if isinstance(data, str):
        data = eval(data)
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        data = []
    names = [ele.get('name') for ele in data if 'name' in ele]
    n = len(names)
    features.update({'names': names, 'len': n})
    # if genders and n > 0:
    #     genders_cnt = Counter([ele.get('gender') for ele in data if 'gender' in ele])
    #     features.update({'g:{}'.format(str(k).replace(' ', '_')): v / n for k, v in genders_cnt.items()})
    if jobs:
        jobs_cnt = Counter([ele.get('job') for ele in data if 'job' in ele])
        features.update({'j:{}'.format(str(k).replace(' ', '_')): v for k, v in jobs_cnt.items()})
    return pd.Series(features)


def json2counter(df, col, top=None):
    values = df[col].apply(lambda x: json_columns_handler(x, False, False))['names'].sum()
    return Counter(values).most_common(top)


# col = 'genres'
# data = data.join(data[col].apply(lambda x: json_columns_handler(x,False, False)).add_prefix(f'{col}_'))
col = 'belongs_to_collection'
data = data.join(data[col].apply(lambda x: json_columns_handler(x,False, False)).add_prefix(f'{col}_'))
# col = 'production_companies'
# data = data.join(data[col].apply(lambda x: json_columns_handler(x,False, False)).add_prefix(f'{col}_'))
# col = 'production_countries'
# data = data.join(data[col].apply(lambda x: json_columns_handler(x,False, False)).add_prefix(f'{col}_'))
# col = 'spoken_languages'
# data = data.join(data[col].apply(lambda x: json_columns_handler(x,False, False)).add_prefix(f'{col}_'))
# col = 'Keywords'
# data = data.join(data[col].apply(lambda x: json_columns_handler(x,False, False)).add_prefix(f'{col}_'))
# col = 'cast'
# data = data.join(data[col].apply(lambda x: json_columns_handler(x,True, False)).add_prefix(f'{col}_'))
# col = 'crew'
# data = data.join(data[col].apply(lambda x: json_columns_handler(x,True, True)).add_prefix(f'{col}_'))


# original_language
languages = ['en', 'fr', 'hi', 'ja', 'es', 'ru', 'ko', 'it', 'zh', 'cn', 'de']
data.loc[data['original_language'].isin(languages) == False, 'original_language'] = 'other'
data = pd.get_dummies(data, columns=['original_language'])

# release_date
data['release_date'] = pd.to_datetime(data['release_date'])
data['year'] = data['release_date'].dt.year
data['month'] = data['release_date'].dt.month


data['revenue'] = np.log(data['revenue'] + 1)
data['no_budget'] = data['budget'] == 0
data['budget'] = np.log(data['budget'] + 1)
data['is_collection'] = data['belongs_to_collection_len'] > 0
data['homepage'] = data['homepage'].notna()
data['poster_path'] = data['poster_path'].notna()
data.loc[((data['runtime'] == 0) | (data['runtime'].isna())), 'runtime'] = data['runtime'].median()


data = data.drop(columns=['backdrop_path', 'original_title', 'overview', 'status', 'id', 'tagline', 'title', 'video',
                          'release_date'])
print(data)