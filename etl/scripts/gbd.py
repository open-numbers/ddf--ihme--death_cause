# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from decimal import Decimal
from ddf_utils.str import to_concept_id
from ddf_utils.index import create_index_file

# set floating point precision larger
pd.set_option('precision', 11)

# configuration
codebook = '../source/codebook/IHME_GBD_2013_CODEBOOK_Y2016M01D15.CSV'
source_dir = '../source'
out_dir = '../../'


def extract_concept_discrete(codebook):
    """extract discrete concepts from codebook"""

    # get all concepts and names
    dis_concept = codebook.ix[:1].T.ix[:11]

    # fill the columns: concept/name/type
    dis_concept.columns = ['concept', 'name']
    dis_concept = dis_concept.set_index('concept')
    dis_concept['concept_type'] = 'string'
    dis_concept['concept_type'].ix[['location', 'cause', 'risk', 'sex', 'age']] = 'entity_domain'
    dis_concept['concept_type'].ix['year'] = 'time'

    dis_concept = dis_concept.reset_index()
    dis_concept['concept'] = dis_concept['concept'].map(to_concept_id)

    # manually add name concept
    dis_concept = dis_concept.append(
        pd.DataFrame([['name', 'Name', 'string']], columns=dis_concept.columns)
    )

    return dis_concept


def extract_concept_continuous(codebook):
    """extract continuous concepts from codebook"""

    # get all concepts and names
    cont_concept = codebook.ix[:1].T.iloc[12:-2]
    # cont_concept = cont_concept.iloc[:-2]

    # fill the columns: concept/name/type/unit
    cont_concept.columns = ['concept', 'name']

    cont_df_list = []

    for i in codebook.iloc[2:, -1].dropna().values:
        cont_new = cont_concept.copy()
        cont_new['concept'] = i + '_' + cont_new['concept']
        cont_new['name'] = i + ' - ' + cont_new['name']
        cont_new['concept'] = cont_new['concept'].map(to_concept_id)

        cont_df_list.append(cont_new)

    cont_concept_all = pd.concat(cont_df_list)

    cont_concept_all['concept_type'] = 'measure'
    cont_concept_all['unit'] = 'years'

    return cont_concept_all


def extract_entities_location(codebook):
    """extract location entities from codebok"""
    # Note for 2016 version of data:
    # there are duplicated entities in locations. below are locations which ids are
    # not used in data file.
    #   * 137    North Africa and Middle East
    #   * 158                      South Asia
    #   * 241     Latin America and Caribbean
    #   * 242              Sub-Saharan Africa
    #   * 244                      South Asia
    cb = codebook.copy()
    cb.columns = cb.ix[0].map(to_concept_id)
    cb = cb.drop([0, 1])

    # get all columns for country
    ent_loc = cb.ix[:, :2]
    ent_loc = ent_loc.dropna(how='all')

    # manually remove duplicated
    ent_loc = ent_loc.set_index('location')
    ent_loc = ent_loc.drop(['137', '158', '241', '242', '244'])
    ent_loc = ent_loc.reset_index()

    if np.all(ent_loc['location_name'].duplicated()) is True:
        raise ValueError('duplicated entities detected! please check and remove them.')

    return ent_loc


def extract_entities_age_group(codebook):
    """extract age group entities from codebok"""

    ent_age = codebook.ix[:, 9:10]
    ent_age = ent_age.dropna(how='all')
    ent_age.columns = ent_age.ix[0]
    ent_age = ent_age.drop([0, 1])

    return ent_age


def extract_entities_sex(codebook):
    """extract sex entities from codebok"""

    ent_sex = codebook.ix[:, 7:8]
    ent_sex = ent_sex.dropna(how='all')
    ent_sex.columns = ent_sex.ix[0]
    ent_sex = ent_sex.drop([0, 1])

    return ent_sex


def extract_entities_risk(codebook):
    """extract risk entities from codebok"""

    ent_risk = codebook.ix[:, 5:6]
    ent_risk = ent_risk.dropna(how='all')
    ent_risk.columns = ent_risk.ix[0]
    ent_risk = ent_risk.drop([0, 1])

    return ent_risk


def extract_entities_cause(codebook):
    """extract cause entities from codebok"""

    ent_cause = codebook.ix[:, 3:4]
    ent_cause = ent_cause.dropna(how='all')
    ent_cause.columns = ent_cause.ix[0]
    ent_cause = ent_cause.drop([0, 1])

    return ent_cause


def _cleanup(df, m):
    df.columns = list(map(to_concept_id, df.columns))
    df = df.loc[:, ['location', 'cause', 'risk', 'sex', 'age', 'year',
                    'nm_mean', 'nm_lower', 'nm_upper', 'rt_mean', 'rt_lower', 'rt_upper',
                    'pc_mean', 'pc_lower', 'pc_upper', 'metric_name']]

    assert len(df['metric_name'].unique()) == 1
    assert df['metric_name'].unique()[0] == m

    for i in ['location', 'cause', 'risk', 'sex', 'age']:
        df[i] = df[i].astype('category')

    return df.drop('metric_name', axis=1).set_index(['location', 'cause', 'risk', 'sex', 'age', 'year'])


def extract_datapoints(source_dir, locs, chunksize=10**6, part=False):
    """
    This function loops through all location files and yield datapoints for each file.

    Each location file contains 9 columns of data points ('nm_mean' to 'pc_upper'),
    and totally there are 4 types of metrics. What this function does is, for each
    type of metric, find all related files, and for each file, extract the 9 columns
    along with the index columns, and return the concept_id and datapoints pairs.
    """

    d = {  # here are the filenames that don't match the location name in codebook.
        "Cote d'Ivoire": 'IHME-Data-Cote dIvoire-{}.zip',
        "East Asia & Pacific": 'IHME-Data-East Asia & Pacific - WB-{}.zip',
        "Europe & Central Asia": 'IHME-Data-Europe & Central Asia - WB-{}.zip',
        "Middle East & North Africa":  'IHME-Data-Middle East & North Africa - WB-{}.zip',
        "WB region":  'IHME-Data-World Bank Regions-{}.zip',
        "World Bank Lower Income":  'IHME-Data-World Bank Low Income-{}.zip'
    }

    m = ['Deaths', 'DALYs', 'YLDs', 'YLLs']
    concepts = ['nm_mean', 'nm_lower', 'nm_upper', 'rt_mean', 'rt_lower',
                'rt_upper', 'pc_mean', 'pc_lower', 'pc_upper']
    for k in m:
        print(k+' ...')
        for l in locs:
            if l in d.keys():
                path = os.path.join(source_dir, d[l].format(k))
            else:
                path = os.path.join(source_dir,
                                    u'IHME-Data-{}-{}.zip'.format(l, k))

            print('processing file: ' + path)
            for cnk in pd.read_csv(path, chunksize=chunksize):
                data = _cleanup(cnk, k)

                for c in concepts:
                    c_id = to_concept_id(k+'_'+c)

                    yield (c_id, data[c].reset_index().rename(columns={c:c_id}))

                if part:
                    break


def _format_num(n):
    """remove scientific notation"""
    g = format(n, 'g')
    if 'e' not in g:
        return g
    else:
        return (format(n, '.15f'))  # only keep 15 digits after decimal point.


if __name__ == '__main__':
    import os

    print('reading codebook...')
    cb = pd.read_csv(codebook, header=None, encoding='cp1252')
    cb = cb.iloc[:, 1:]

    print('creating concept files...')
    dis_concept = extract_concept_discrete(cb)
    dis_concept.to_csv(os.path.join(out_dir, 'ddf--concepts--discrete.csv'), index=False)

    cont_concept = extract_concept_continuous(cb)
    cont_concept.to_csv(os.path.join(out_dir, 'ddf--concepts--continuous.csv'), index=False)

    print('creating entities files...')
    ents = {
        'location': extract_entities_location,
        'risk': extract_entities_risk,
        'cause': extract_entities_cause,
        'age': extract_entities_age_group,
        'sex': extract_entities_sex,
    }

    for k, func in ents.items():
        path = os.path.join(out_dir, 'ddf--entities--{}.csv'.format(k))
        df = func(cb)
        df.to_csv(path, index=False, encoding='utf8')

    # create datapoints
    # for each piece of data return by extract_datapoints(),
    # we will write append to the file on the disk.
    # so we will remove all datapoints files first to avoid
    # appending to old datapoints files.
    print('creating datapoints files...will take a while')
    locs = extract_entities_location(cb)['location_name'].values

    # firstly we remove all datapoints
    for f in os.listdir(out_dir):
        if 'datapoints' in f:
            os.remove(os.path.join(out_dir, f))

    for c, df in extract_datapoints(source_dir, locs, chunksize=50e5):
        path = os.path.join(out_dir,
                            'ddf--datapoints--{}--by--location--risk--cause--sex--age--year.csv'.format(c))
        # TODO: find better solution for floating point persicion
        # there are numbers like 1.010625776e-13 in the source data.
        # after saving to datapoint file, it becomes 0.000000000000101.
        df[c] = df[c].map(_format_num)
        if os.path.exists(path):
            df.to_csv(path, mode='a', index=False, header=False)
        else:
            df.to_csv(path, index=False)

    print('creating index files...')
    create_index_file(out_dir)

    print('Done!')
