import pandas as pd
import numpy as np
from collections import Counter
from metric_learn import MLKR
from sklearn.preprocessing import StandardScaler
import pickle


class Preprocess:
    def __init__(self):
        self.tops = {'original_language': ['en', 'fr', 'hi', 'ja', 'es', 'ru', 'ko', 'it', 'zh', 'cn', 'de']}
        self.cnts = {}
        self.scaler = None
        self.scaler_columns = []
        self.ids = {}
        self.mlkrs = {}


    def fit_transform(self, train):
        data = Preprocess.extraced_json_columns(train.copy())
        data['revenue'] = np.log(data['revenue'] + 1)
        self.tops['genres_names'] = [x[0] for x in Counter(data['genres_names'].sum()).most_common()]
        self.tops['production_countries_names'] = [x[0] for x in Counter(data['production_countries_names'].sum()).most_common() if x[1] > 50]
        self.tops['production_companies_names'] = [x[0] for x in Counter(data['production_companies_names'].sum()).most_common() if x[1] > 35]
        self.tops['cast_names'] = [x[0] for x in Counter(data['cast_names'].sum()).most_common() if x[1] > 35]
        self.tops['crew_names'] = [x[0] for x in Counter(data['crew_names'].sum()).most_common() if x[1] > 35]
        self.tops['Keywords_names'] = [x[0] for x in Counter(data['Keywords_names'].sum()).most_common() if x[1] > 35]
        self.tops['crew_jobs'] = [x[0] for x in Counter(data['crew_jobs'].apply(lambda x: list(x)).sum()).most_common() if x[1] > 35]


        self.cnts = {'production_companies_names': Counter(data['production_companies_names'].sum()),
                     'production_countries_names': Counter(data['production_countries_names'].sum()),
                     'cast_names': Counter(data['cast_names'].sum()),
                     'crew_names': Counter(data['crew_names'].sum())}

        # Preprocess.create_embedding(self, data, 'production_companies_names', 'revenue', threshold=0, n_components=5)
        # Preprocess.create_embedding(self, data, 'cast_names', 'revenue', threshold=10, n_components=10)
        # Preprocess.create_embedding(self, data, 'crew_names', 'revenue', threshold=10, n_components=10)
        # Preprocess.create_embedding(self, data, 'crew_jobs', 'revenue', threshold=10, n_components=10)
        # Preprocess.create_embedding(self, data, 'Keywords_names', 'revenue', threshold=10, n_components=10)
        return self.transform(data, init=True)

    def transform(self, data, init=False):
        if not init:
            data = Preprocess.extraced_json_columns(data)

        # original_language
        data.loc[data['original_language'].isin(self.tops['original_language']) == False, 'original_language'] = 'other'
        data = pd.get_dummies(data, columns=['original_language'])

        # release_date
        data['release_date'] = pd.to_datetime(data['release_date'])
        data['year'] = data['release_date'].dt.year
        data['month'] = data['release_date'].dt.month
        data['no_budget'] = data['budget'] == 0
        data['budget'] = np.log(data['budget'] + 1)
        data['collection'] = data['belongs_to_collection_len'] > 0
        data['homepage'] = data['homepage'].notna()
        data['poster_path'] = data['poster_path'].notna()
        data.loc[((data['runtime'] == 0) | (data['runtime'].isna())), 'runtime'] = data['runtime'].median()

        data = self.list2binary(data, 'genres_names')
        data = self.list2binary(data, 'production_countries_names')
        data = self.list2binary(data, 'production_companies_names')
        data = self.list2binary(data, 'cast_names')
        data = self.list2binary(data, 'crew_names')
        data = self.list2binary(data, 'Keywords_names')
        data = self.list2binary(data, 'crew_jobs')

        # data = self.embedding(data, 'production_companies_names')
        # data = self.embedding(data, 'cast_names')
        # data = self.embedding(data, 'crew_names')
        # data = self.embedding(data, 'Keywords_names')
        # data = self.embedding(data, 'crew_jobs')

        data = self.count_instances(data, 'production_companies_names', 1, 3)
        data = self.count_instances(data, 'production_companies_names', 4, 10)
        data = self.count_instances(data, 'production_countries_names', 0, 10)
        data = self.count_instances(data, 'production_countries_names', 11, 50)
        data = self.count_instances(data, 'cast_names', 4, 10000)

        if init:
            self.scaler = StandardScaler()
            self.scaler_columns = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'year',
                                   'month', 'genres_len', 'production_companies_len', 'production_countries_len',
                                   'Keywords_len', 'cast_len', 'crew_len', 'crew_jobs_len',
                                   'production_companies_names_1-3', 'production_companies_names_4-10',
                                   'production_countries_names_0-10', 'production_countries_names_11-50',
                                   'cast_names_4-10000']
            self.scaler.fit(data[self.scaler_columns])

        data[self.scaler_columns] = self.scaler.transform(data[self.scaler_columns])
        data = data.drop(
            columns=['backdrop_path', 'original_title', 'overview', 'status', 'id', 'tagline', 'title', 'video',
                     'release_date', 'belongs_to_collection', 'genres', 'imdb_id',
                     'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast',
                     'crew', 'genres_names', 'belongs_to_collection_names', 'belongs_to_collection_len',
                     'production_companies_names', 'production_countries_names', 'Keywords_names', 'cast_names',
                     'crew_names', 'crew_jobs'])
        return data.reindex(sorted(data.columns), axis=1)

    def count_instances(self, data, col, l, u):
        counter = self.cnts[col]
        new_col = col + '_' + str(l) + '-' + str(u)
        data[new_col] = data[col].apply(lambda x: len([ele for ele in x if l <= counter[ele] <= u]))
        return data

    @staticmethod
    def map_to_ids(data, col, threshold=0):
        cnt = [x[0] for x in Counter(data[col].apply(lambda x: list(x)).sum()).most_common() if x[1] > threshold]
        return {x: i for i, x in enumerate(cnt)}

    def create_embedding(self, data, col, target_col, threshold=0, n_components=5):
        self.ids[col] = Preprocess.map_to_ids(data, col, threshold)
        mat = Preprocess.list2vector(data, col, self.ids[col])
        self.mlkrs[col] = MLKR(n_components, verbose=True, max_iter=1000)
        self.mlkrs[col].fit(mat, data[target_col])

    def embedding(self, data, col):
        mat = Preprocess.list2vector(data, col, self.ids[col])
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
        data = data.join(data[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'belongs_to_collection'
        data = data.join(data[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'production_companies'
        data = data.join(data[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'production_countries'
        data = data.join(data[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'Keywords'
        data = data.join(data[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'cast'
        data = data.join(data[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
        col = 'crew'
        data = data.join(data[col].apply(lambda x: Preprocess.json_columns_handler(x, True)).add_prefix(f'{col}_'))
        return data

    @staticmethod
    def json2counter(data, col, top=None):
        values = data[col].apply(lambda x: Preprocess.json_columns_handler(x, False))['names'].sum()
        return Counter(values).most_common(top)

    def list2binary(self, data, col):
        def list_handler(ls):
            return pd.Series({'{}_{}'.format(col, str(k).replace(' ', '_')): 1 for k in ls if k in self.tops[col]})
        return data.join(data[col].apply(lambda x: list_handler(x)).fillna(0).astype(int))

    @staticmethod
    def list2vector(data, col, ids):
        def list_handler(ls):
            # here
            if isinstance(ls, dict):
                return pd.Series({ids[k]: v for k, v in ls.items() if k in ids})
            else:
                return pd.Series({ids[k]: 1 for k in ls if k in ids})

        vectors = data[col].apply(lambda x: list_handler(x)).fillna(0).astype(int)
        for i in ids.values():
            if i not in vectors.columns:
                vectors[i] = 0
        return vectors[list(range(len(ids)))]

    def save(self, path):
        with open(path, 'wb') as file:
            return pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

