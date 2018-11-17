# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import pandas as pd

from ddf_utils.str import to_concept_id, format_float_digits
from ddf_utils.factory import ihme

from functools import partial
from itertools import product
from zipfile import ZipFile

import dask
import dask.dataframe as dd


source_dir = '../source/'
output_dir = '../../'

formatter = partial(format_float_digits, digits=2)

# below are configs for downloader when downloading the source files.
MEASURES = [1]
METRICS = [1]
AGES = [22, 27, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 30, 31, 32, 235, 23]


DTYPES = dict(location=np.int32, sex=np.int8, year=np.int16, cause=np.int16, age=np.int16, val=np.float,
              measure=np.int8, metric=np.int8)


def load_data(n):
    # print(n)
    f = osp.join(source_dir, n)
    zf = ZipFile(f)
    fname = osp.splitext(n)[0] + '.csv'
    data = pd.read_csv(zf.open(fname), dtype=DTYPES)
    data = data.drop(['upper', 'lower'], axis=1)

    # double check things
    assert set(data['metric'].unique().tolist()).issubset(set(METRICS))
    assert set(data['measure'].unique().tolist()).issubset(set(MEASURES))
    assert set(data['age'].unique().tolist()).issubset(set(AGES))

    return data


def serve_datapoint(df, concept):
    cols_new = ['location', 'sex', 'age', 'cause', 'year', concept]
    df.columns = cols_new

    causes = df.cause.unique()
    sexes = df.sex.unique()
    # if concept == 'mmr_rate':
    #     print(df.sex.unique())
    for g_ in product(causes, sexes):
        print(f"working on group {g_}")
        df_ = df[(df.cause == g_[0]) & (df.sex == g_[1])]
        if df_.empty:
            continue
        # print(g_)
        # print(df_.age.unique())
        # print(len(df_.year.unique()))
        # print(len(df_.location.unique()))
        df_ = df_.rename(columns={'val': concept})
        cause = 'cause-{}'.format(g_[0])
        sex = 'sex-{}'.format(g_[1])
        by = ['location', sex, 'age', cause, 'year']
        file_name = 'ddf--datapoints--' + concept + '--by--' + '--'.join(by) + '.csv'
        file_path = osp.join(output_dir, 'deaths', file_name)
        df_[concept] = df_[concept].map(formatter)
        df_[cols_new].sort_values(by=['location', 'sex', 'age', 'year']).to_csv(file_path, index=False)


def serve_entities(md):
    sexs = md['sex'].copy()
    sexs.columns = ['sex', 'name', 'short_name']
    sexs[['sex', 'name']].to_csv('../../ddf--entities--sex.csv', index=False)

    causes = md['cause'].copy()
    causes = causes.drop(['most_detailed', 'sort_order'], axis=1)
    causes.columns = ['cause', 'label', 'name', 'medium_name', 'short_name']
    causes.to_csv('../../ddf--entities--cause.csv', index=False)

    locations = md['location'].copy()
    locations = locations[locations.location_id != 'custom'].drop(['location_id', 'enabled'], axis=1)
    locations = locations.rename(columns={'id': 'location'})
    locations = locations[['location', 'type', 'name', 'medium_name', 'short_name']]
    locations.location = locations.location.map(lambda x: str(int(x)))
    locations.to_csv('../../ddf--entities--location.csv', index=False)

    ages = md['age'].copy()
    ages = ages.sort_values(by='sort')[['id', 'name', 'short_name', 'type']]
    ages.columns = ['age', 'name', 'short_name', 'type']
    ages.to_csv('../../ddf--entities--age.csv', index=False)


def main():
    md = ihme.load_metadata()
    metric = md['metric'].copy()
    measure = md['measure'].copy()

    # datapoints
    datapoint_output_dir = osp.join(output_dir, 'deaths')
    os.makedirs(datapoint_output_dir, exist_ok=True)

    data_full = dd.from_delayed([dask.delayed(load_data)(f) for f in os.listdir(source_dir) if f.endswith('.zip')], meta=DTYPES)

    metric = metric.set_index('id')['name'].to_dict()
    measure = measure.set_index('id')['short_name'].to_dict()

    all_measures = list()
    measure_metric_combinations = product(MEASURES, METRICS)
    for g in measure_metric_combinations:
        name = measure[g[0]] + ' ' + metric[g[1]]
        print(f'creating dattpoints for {name}')
        concept = to_concept_id(name)
        all_measures.append((concept, name))

        cols = ['location', 'sex', 'age', 'cause', 'year', 'val']
        df = data_full.loc[(data_full.measure == g[0]) & (data_full.metric == g[1]), cols].compute()
        serve_datapoint(df, concept)

    # entities
    serve_entities(md)

    # concepts
    cont_cdf = pd.DataFrame(all_measures, columns=['concept', 'name'])
    cont_cdf['concept_type'] = 'measure'
    cont_cdf.to_csv('../../ddf--concepts--continuous.csv', index=False)

    dis_cdf = pd.DataFrame([
        ['name', 'Name', 'string'],
        ['short_name', 'Short Name', 'string'],
        ['medium_name', 'Medium Name', 'string'],
        ['long_name', 'Long Name', 'string'],
        ['location', 'Location', 'entity_domain'],
        ['sex', 'Sex', 'entity_domain'],
        ['age', 'Age', 'entity_domain'],
        ['cause', 'Cause', 'entity_domain'],
        ['rei', 'Risk/Etiology/Impairment', 'entity_domain'],
        ['label', 'Label', 'string'],
        ['year', 'Year', 'time'],
        ['type', 'Type', 'string']
    ], columns=['concept', 'name', 'concept_type'])

    dis_cdf.sort_values(by='concept').to_csv('../../ddf--concepts--discrete.csv', index=False)

    print("Done.")


if __name__ == '__main__':
    main()
