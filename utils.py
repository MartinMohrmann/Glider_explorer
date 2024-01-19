from erddapy import ERDDAP
import pprint
from ast import literal_eval
import pandas as pd

def load_metadata():
    server = "https://erddap.observations.voiceoftheocean.org/erddap"
    e = ERDDAP(
        server=server,
        protocol="tabledap",
        response="csv",
    )
    e.dataset_id = "meta_metadata_table"
    metadata = e.to_pandas(
        index_col="datasetID",
        parse_dates=True,
    )

    def obj_to_string(x):
        return pprint.pformat(x)

    def variable_exists(x, variable):
        return variable in x

    def basin_simplify(basin):
        if basin=='Eastern Gotland Basin, Northern Baltic Proper':
            return 'Eastern Gotland'
        if basin=='Northern Baltic Proper, Eastern Gotland Basin':
            return 'Eastern Gotland'
        elif basin=='Western Gotland Basin':
            return 'Western Gotland'
        elif basin=='Eastern Gotland Basin':
            return 'Eastern Gotland'
        elif basin=='Western Gotland Basin, Eastern Gotland Basin':
            return 'Western Gotland'
        elif basin=='Kattegat':
            return 'Skagerrak, Kattegat'
        elif basin=='Kattegat, Skagerrak':
            return 'Skagerrak, Kattegat'
        elif basin=='Skagerrak':
            return 'Skagerrak, Kattegat'
        elif basin=='Northern Baltic Proper':
            return 'Eastern Gotland'
            return 'Skagerrak, Kattegat'
        elif basin=='\\u00c3\\u0085land Sea':
            return 'Åland Sea'
        else:
            return basin

    metadata['optics_serial'] = metadata.optics_serial.apply(obj_to_string)
    metadata['irradiance_serial'] = metadata.irradiance_serial.apply(obj_to_string)
    metadata['altimeter_serial'] = metadata.altimeter_serial.apply(obj_to_string)
    metadata['glider_serial'] = metadata.glider_serial.apply(obj_to_string)
    metadata['basin'] = metadata.basin.apply(basin_simplify)

    # create list of all variables
    all_variables_set = set()
    menuentries = []
    menuentries_variables = []
    newmetadatacolumns = {}
    for index in range(0,len(metadata.index)):
        all_variables_set.update(literal_eval(metadata.iloc[index].variables))
    all_variables_set

    for variable in list(all_variables_set):
        newmetadatacolumns[variable+'_available'] = metadata.variables.apply(variable_exists, args=(variable,))
        menuentries.append({'label':variable+'_available', 'value':variable+'_available'})
        menuentries_variables.append({'label':variable,variable+'_available' 'value':variable})
    metadata = metadata.join(pd.DataFrame.from_dict(newmetadatacolumns))
    metadata['time_coverage_end (UTC)'] = pd.to_datetime(metadata['time_coverage_end (UTC)'])
    metadata['time_coverage_start (UTC)'] = pd.to_datetime(metadata['time_coverage_start (UTC)'])
    return metadata

def drop_overlaps(metadata):
    drop_overlap=True
    dropped_datasets = []
    for index in range(0, len(metadata)):
        glidercounter = 1
        maskedregions = []
        color = 'k'
        for index2 in range(0, index):
            r1 = dict(start=metadata.iloc[index]['time_coverage_start (UTC)'],
                      end=metadata.iloc[index]['time_coverage_end (UTC)'])
            r2 = dict(start=metadata.iloc[index2]['time_coverage_start (UTC)'],
                      end=metadata.iloc[index2]['time_coverage_end (UTC)'])
            latest_start = max(r1['start'], r2['start'])
            earliest_end = min(r1['end'], r2['end'])
            delta = (earliest_end - latest_start).days + 1
            overlap = max(0, delta)
            if overlap > 1:
                glidercounter += 1
                # if two Glider datasets are overlapping by more than a
                # day, they are plotted in multiple rows...
                if drop_overlap:
                    # ...and optionally dropped
                    dropped_datasets.append(metadata.index[index])
                    color = 'red'

    print('dropping datasets {}'.format(dropped_datasets))
    metadata = metadata.drop(dropped_datasets)
    return metadata

# this is a version were I only change the profile_nums, to try if no-concatenation helps with datashader performance
def voto_concat_datasets(datasets):
    """
    Concatenates multiple datasets along the time dimensions, profile_num
    and dives variable(s) are adapted so that they start counting from one
    for the first dataset and monotonically increase.

    Parameters
    ----------
    datasets : list of xarray.Datasets

    Returns
    -------
    xarray.Dataset
        concatenated Dataset containing all the data from the list of datasets
    """
    # in case the datasets have a different set of variables, emtpy variables are created
    # to allow for concatenation (concat with different set of variables leads to error)
    mlist = [set(dataset.variables.keys()) for dataset in datasets]
    allvariables = set.union(*mlist)
    for dataset in datasets:
        missing_vars = allvariables - set(dataset.variables.keys())
        for missing_var in missing_vars:
            dataset[missing_var] = np.nan

    # renumber profiles, so that profile_num still is unique in concat-dataset
    for index in range(1, len(datasets)):
        datasets[index]["profile_num"] += (
            datasets[index - 1].copy()["profile_num"].max()
        )
    return datasets
