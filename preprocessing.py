import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from metric_learn import MLKR
from sklearn.preprocessing import StandardScaler


class Preproccess:
    def __init__(self, train):
        data = Preproccess.extraced_json_columns(train.copy())
        self.languages = ['en', 'fr', 'hi', 'ja', 'es', 'ru', 'ko', 'it', 'zh', 'cn', 'de']
        self.top_genres = [x[0] for x in Counter(data['genres_names'].sum()).most_common()]
        self.top_countries = [x[0] for x in Counter(data['production_companies_names'].sum()).most_common() if x[1] > 50]

        self.cnts = {'production_companies_names': Counter(data['production_companies_names'].sum()),
                     'production_countries_names': Counter(data['production_countries_names'].sum()),
                     'cast_names': Counter(data['cast_names'].sum()),
                     'crew_names': Counter(data['crew_names'].sum())}
        self.ids = {}
        self.mlkrs = {}
        Preproccess.create_embedding(self, data, 'production_companies_names', threshold=0, n_components=5)
        Preproccess.create_embedding(self, data, 'cast_names', threshold=10, n_components=10)
        Preproccess.create_embedding(self, data, 'crew_names', threshold=10, n_components=10)
        Preproccess.create_embedding(self, data, 'crew_jobs', threshold=10, n_components=10)
        Preproccess.create_embedding(self, data, 'Keywords_names', threshold=10, n_components=10)

        # self.keywords = Counter(data[''].sum()).most_common()

    def preprocessing(self, data):
        data = Preproccess.extraced_json_columns(data)
        data = Preproccess.list2binary(data, 'genres_names', self.top_genres)
        data = self.count_instances(data, 'production_companies_names', 1, 3)
        data = self.count_instances(data, 'production_companies_names', 4, 10)
        data = Preproccess.embedding(self, data, 'production_companies_names')
        data = self.count_instances(data, 'production_countries_names', 0, 10)
        data = self.count_instances(data, 'production_countries_names', 11, 50)
        data = Preproccess.list2binary(data, 'production_countries_names', self.top_countries)
        data = self.count_instances(data, 'cast_names', 4, 10000)
        data = Preproccess.embedding(self, data, 'cast_names')
        data = Preproccess.embedding(self, data, 'crew_names')
        data = Preproccess.embedding(self, data, 'Keywords_names')
        data = Preproccess.embedding(self, data, 'crew_jobs')

        # original_language
        data.loc[data['original_language'].isin(self.languages) == False, 'original_language'] = 'other'
        data = pd.get_dummies(data, columns=['original_language'])

        # release_date
        data['release_date'] = pd.to_datetime(data['release_date'])
        data['year'] = data['release_date'].dt.year
        data['month'] = data['release_date'].dt.month

        data['revenue'] = np.log(data['revenue'] + 1)
        data['no_budget'] = data['budget'] == 0
        data['budget'] = np.log(data['budget'] + 1)
        data['collection'] = data['belongs_to_collection_len'] > 0
        data['homepage'] = data['homepage'].notna()
        data['poster_path'] = data['poster_path'].notna()
        data.loc[((data['runtime'] == 0) | (data['runtime'].isna())), 'runtime'] = data['runtime'].median()

        data = data.drop(
            columns=['backdrop_path', 'original_title', 'overview', 'status', 'id', 'tagline', 'title', 'video',
                     'release_date', 'belongs_to_collection', 'genres', 'imdb_id',
                     'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast',
                     'crew', 'genres_names', 'belongs_to_collection_names', 'belongs_to_collection_len',
                     'production_companies_names', 'production_countries_names', 'Keywords_names', 'cast_names',
                     'crew_names', 'crew_jobs'])

        data.to_csv('data.csv', sep='\t', index=False)

    def count_instances(self, data, col, l, u):
        counter = self.cnts[col]
        new_col = col + '_' + str(l) + '-' + str(u)
        data[new_col] = data[col].apply(lambda x: len([ele for ele in x if l <= counter[ele] <= u]))
        return data

    @staticmethod
    def map_to_ids(df, col, threshold=0):
        # here
        cnt = [x[0] for x in Counter(df[col].apply(lambda x: list(x)).sum()).most_common() if x[1] > threshold]
        return {x: i for i, x in enumerate(cnt)}


    def create_embedding(self, data, col, threshold=0, n_components=5):
        self.ids[col] = Preproccess.map_to_ids(data, col, threshold)
        mat = Preproccess.list2vector(data, col, self.ids[col])
        self.mlkrs[col] = MLKR(n_components, verbose=True, max_iter=10)
        self.mlkrs[col].fit(mat, data['revenue'])


    def embedding(self, data, col):
        mat = Preproccess.list2vector(data, col, self.ids[col])
        return data.join(pd.DataFrame(self.mlkrs[col].transform(mat)).add_prefix(col + '_'))


    @staticmethod
    def json_columns_handler(data, jobs=False):
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
        if jobs:
            jobs = dict(Counter([ele.get('job') for ele in data if 'job' in ele]))
            features.update({'jobs': jobs, 'jobs_len': len(jobs)})
        return pd.Series(features)

    @staticmethod
    def extraced_json_columns(data):
        col = 'genres'
        data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'belongs_to_collection'
        data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'production_companies'
        data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'production_countries'
        data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x)).add_prefix(f'{col}_'))
        # col = 'spoken_languages'
        # data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x,False, False)).add_prefix(f'{col}_'))
        col = 'Keywords'
        data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'cast'
        data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'crew'
        data = data.join(data[col].apply(lambda x: Preproccess.json_columns_handler(x, True)).add_prefix(f'{col}_'))
        return data

    @staticmethod
    def json2counter(df, col, top=None):
        values = df[col].apply(lambda x: Preproccess.json_columns_handler(x, False))['names'].sum()
        return Counter(values).most_common(top)

    @staticmethod
    def list2binary(df, col, cnt):
        def list_handler(ls):
            return pd.Series({'{}_{}'.format(col, str(k).replace(' ', '_')): 1 for k in ls if k in cnt})

        return df.join(df[col].apply(lambda x: list_handler(x)).fillna(0).astype(int))

    @staticmethod
    def list2vector(df, col, ids):
        def list_handler(ls):
            # here
            if isinstance(ls, dict):
                return pd.Series({ids[k]: v for k, v in ls.items() if k in ids})
            else:
                return pd.Series({ids[k]: 1 for k in ls if k in ids})
        vectors = df[col].apply(lambda x: list_handler(x)).fillna(0).astype(int)
        for i in ids.values():
            if i not in vectors.columns:
                vectors[i] = 0
        return vectors[list(range(len(ids)))]


if __name__ == '__main__':
    file_path = 'train.tsv'
    data = pd.read_csv(file_path, sep="\t")
    preproccess = Preproccess(data)
    preproccess.preprocessing(data)
