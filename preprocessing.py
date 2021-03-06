import pandas as pd
import numpy as np
from collections import Counter
from metric_learn import MLKR
from sklearn.preprocessing import StandardScaler
import pickle


class Preprocess:
    def __init__(self, scale):
        self.tops = {'original_language': ['en', 'fr', 'hi', 'ja', 'es', 'ru', 'ko', 'it', 'zh', 'cn', 'de']}
        self.cnts = {}
        self.yearly_df = None
        self.yearly_bins = None
        self.new_budgets = pd.read_csv('budgets.csv')
        self.scale = scale
        self.scaler = None
        self.scaler_columns = []
        self.ids = {}
        self.mlkrs = {}
        self.features = []


    def fit_transform(self, train):
        data = Preprocess.extraced_json_columns(train)
        data['revenue'] = np.log(data['revenue'] + 1)
        self.tops['genres_names'] = [x[0] for x in Counter(data['genres_names'].sum()).most_common()]
        self.tops['production_countries_names'] = [x[0] for x in Counter(data['production_countries_names'].sum()).most_common(15)] + ['other']
        self.tops['production_companies_names'] = [x[0] for x in Counter(data['production_companies_names'].sum()).most_common(50)] + ['other']
        self.tops['cast_names'] = [x[0] for x in Counter(data['cast_names'].sum()).most_common(50)] + ['other']
        self.tops['crew_names'] = [x[0] for x in Counter(data['crew_names'].sum()).most_common(50)] + ['other']
        self.tops['Keywords_names'] = [x[0] for x in Counter(data['Keywords_names'].sum()).most_common(50)] + ['other']
        self.tops['crew_jobs'] = [x[0] for x in Counter(data['crew_jobs'].apply(lambda x: list(x)).sum()).most_common(30)] + ['other']

        self.cnts = {'production_companies_names': Counter(data['production_companies_names'].sum()),
                     'production_countries_names': Counter(data['production_countries_names'].sum()),
                     'cast_names': Counter(data['cast_names'].sum()),
                     'crew_names': Counter(data['crew_names'].sum())}

        # Preprocess.create_embedding(self, data, 'production_companies_names', 'revenue', threshold=5, n_components=3)
        # Preprocess.create_embedding(self, data, 'cast_names', 'revenue', threshold=10, n_components=10)
        # Preprocess.create_embedding(self, data, 'crew_names', 'revenue', threshold=10, n_components=10)
        # Preprocess.create_embedding(self, data, 'crew_jobs', 'revenue', threshold=10, n_components=10)
        # Preprocess.create_embedding(self, data, 'Keywords_names', 'revenue', threshold=10, n_components=4)
        return self.transform(data, init=True)

    def transform(self, data, init=False):
        data = data.copy()
        if not init:
            data = Preprocess.extraced_json_columns(data)

        # original_language
        data.loc[data['original_language'].isin(self.tops['original_language']) == False, 'original_language'] = 'other'
        data = pd.get_dummies(data, columns=['original_language'])
        # release_date
        data['release_date'] = pd.to_datetime(data['release_date'])
        data['year'] = data['release_date'].dt.year
        data['year'] = data['year'].where(data['year'] >= 1940, 1939)
        data['month'] = data['release_date'].dt.month
        data['no_budget'] = (data['budget'] == 0) | (data['budget'].isna())
        data['low_budget'] = (data['budget'].between(1, 10000))
        data['budget'] = np.log(data['budget'] + 1)
        data['popularity'] = data['popularity'].where(data['popularity'] < 100, 100)
        data['collection'] = data['belongs_to_collection_len'] > 0
        data['homepage'] = data['homepage'].notna()
        data['poster_path'] = data['poster_path'].notna()
        if init:
            data['date_bin'], self.yearly_bins = pd.qcut(data['release_date'], q=40, retbins=True)
            self.yearly_df = pd.merge(data[['date_bin']].drop_duplicates().sort_values(by='date_bin'),
                                      data[data['low_budget'] == False][['date_bin', 'budget', 'popularity', 'runtime',
                                           'vote_average', 'vote_count']].groupby('date_bin').median().reset_index(),
                                         on='date_bin', how='left')
            self.yearly_df[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']] = self.yearly_df[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']].fillna(method='ffill')
        data['date_bin'] = pd.cut(data['release_date'], self.yearly_bins)
        data = pd.merge(data, self.yearly_df, on='date_bin', how='left', suffixes=('', '_yearly'))
        # filling nans
        data = pd.merge(data, self.new_budgets[['imdb_id', 'final_budget']], on='imdb_id', how='left')
        data.loc[(data['no_budget']) & (data['final_budget'].notna()), 'budget'] = data.loc[(data['no_budget']) & (data['final_budget'].notna()), 'final_budget']
        data.loc[((data['runtime'] == 0) | (data['runtime'].isna())), 'runtime'] = data.loc[((data['runtime'] == 0) | (data['runtime'].isna())), 'runtime_yearly']
        data['new_budget'] = (data['no_budget']) & (data['budget'] > 0)
        for col in ['popularity', 'vote_average', 'vote_count']:
            data.loc[data[col].isna(), col] = data.loc[data[col].isna(), col + '_yearly']
        # ratios
        for col in ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']:
            data[col + '_ratio'] = data[col] / (data[col + '_yearly'] + 1)
        # interactions
        for col_num in ['budget', 'popularity', 'vote_count']:
            for col_denom in ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']:
                if col_num == col_denom:
                    continue
                else:
                    if col_num < col_denom:
                        data[col_num + '_' + col_denom + '_m_interaction'] = data[col_num] * data[col_denom]
                        # data[col_num + '_' + col_denom + '_m_ratio_interaction'] = data[col_num] * data[col_denom + '_ratio']
                    data[col_num + '_' + col_denom + '_d_interaction'] = data[col_num] / (data[col_denom] + 1)
                    # data[col_num + '_' + col_denom + '_d_ratio_interaction'] = data[col_num] / (data[col_denom + '_ratio'] + 0.0001)
        # remove yearly
        data = data.drop(columns=[col for col in data.columns if col.endswith('_yearly')])

        # add binaries
        data = self.list2binary(data, 'genres_names')
        data = self.list2binary(data, 'production_countries_names')
        data = self.list2binary(data, 'production_companies_names')
        data = self.list2binary(data, 'cast_names')
        data = self.list2binary(data, 'crew_names')
        data = self.list2binary(data, 'Keywords_names')
        data = self.list2binary(data, 'crew_jobs')
        # add embeddings

        # data = self.embedding(data, 'production_companies_names')
        # data = self.embedding(data, 'cast_names')
        # data = self.embedding(data, 'crew_names')
        # data = self.embedding(data, 'Keywords_names')
        # data = self.embedding(data, 'crew_jobs')

        # add counts
        data = self.count_instances(data, 'production_companies_names', 1, 3)
        data = self.count_instances(data, 'production_companies_names', 4, 10)
        data = self.count_instances(data, 'production_countries_names', 0, 10)
        data = self.count_instances(data, 'production_countries_names', 11, 50)
        data = self.count_instances(data, 'cast_names', 4, 10000)

        if self.scale:
            if init:
                self.scaler = StandardScaler()
                self.scaler_columns = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'year',
                                       'month', 'genres_len', 'production_companies_len', 'production_countries_len',
                                       'Keywords_len', 'cast_len', 'crew_len', 'crew_jobs_len',
                                       'production_companies_names_1-3', 'production_companies_names_4-10',
                                       'production_countries_names_0-10', 'production_countries_names_11-50',
                                       'cast_names_4-10000'] + [col for col in data.columns if
                                                            (col.endswith('_interaction') or col.endswith('_ratio'))]
                self.scaler.fit(data[self.scaler_columns])

            data[self.scaler_columns] = self.scaler.transform(data[self.scaler_columns])

        data = data.drop(
            columns=['backdrop_path', 'original_title', 'overview', 'status', 'tagline', 'title', 'video',
                     'release_date', 'belongs_to_collection', 'genres', 'imdb_id', 'date_bin',
                     'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast',
                     'crew', 'genres_names', 'belongs_to_collection_names', 'belongs_to_collection_len',
                     'production_companies_names', 'production_countries_names', 'Keywords_names', 'cast_names',
                     'crew_names', 'crew_jobs'])
        data = data.set_index('id')
        if init:
            data = data.reindex(sorted(data.columns), axis=1)
            self.features = data.columns.tolist()
            if 'revenue' in self.features:
                self.features.remove('revenue')
                return data[self.features + ['revenue']].replace([np.inf, -np.inf], 0).fillna(0)
        else:
            for col in self.features:
                if col not in data.columns:
                    data[col] = 0
        return data[self.features].replace([np.inf, -np.inf], 0).fillna(0)


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
        mat = self.list2vector(data, col)
        self.mlkrs[col] = MLKR(n_components, verbose=True, max_iter=1000)
        self.mlkrs[col].fit(mat, data[target_col])

    def embedding(self, data, col):
        mat = self.list2vector(data, col)
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
            jobs_dict = dict(Counter([ele.get('job') for ele in data if 'job' in ele]))
            features.update({'jobs': jobs_dict, 'jobs_len': len(jobs_dict)})
        return pd.Series(features)

    @staticmethod
    def extraced_json_columns(data):
        size = 3000
        list_of_dfs = [data.iloc[i:i + size, :] for i in range(0, len(data), size)]
        for i, df in enumerate(list_of_dfs):
            col = 'genres'
            df = df.join(df[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
            col = 'belongs_to_collection'
            df = df.join(df[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
            col = 'production_companies'
            df = df.join(df[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
            col = 'production_countries'
            df = df.join(df[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
            col = 'Keywords'
            df = df.join(df[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
            col = 'cast'
            df = df.join(df[col].apply(lambda x: Preprocess.json_columns_handler(x)).add_prefix(f'{col}_'))
            col = 'crew'
            df = df.join(df[col].apply(lambda x: Preprocess.json_columns_handler(x, True)).add_prefix(f'{col}_'))
            list_of_dfs[i] = df
        data = pd.concat(list_of_dfs, axis=0)
        return data

    @staticmethod
    def json2counter(data, col, top=None):
        values = data[col].apply(lambda x: Preprocess.json_columns_handler(x, False))['names'].sum()
        return Counter(values).most_common(top)


    def list2binary(self, data, col):
        def list_handler(ls):
            series = {'{}_{}'.format(col, str(k).replace(' ', '_')): 1 if k in ls else 0 for k in self.tops[col]}
            if sum(series.values()) == 0:
                series[f'{col}_other'] = 1
            return pd.Series(series)
        return data.join(data[col].apply(lambda x: list_handler(x)).fillna(0).astype(int))

    def list2vector(self, data, col):
        def list_handler(ls):
            if isinstance(ls, dict):
                return pd.Series({self.ids[col][k]: ls[k] if k in ls else 0 for k in self.ids[col]})
            else:
                return pd.Series({self.ids[col][k]: 1 if k in ls else 0 for k in self.ids[col]})

        vectors = data[col].apply(lambda x: list_handler(x)).fillna(0).astype(int)
        for i in self.ids[col].values():
            if i not in vectors.columns:
                vectors[i] = 0
        return vectors[list(range(len(self.ids[col])))]

    def save(self, path):
        with open(path, 'wb') as file:
            return pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

