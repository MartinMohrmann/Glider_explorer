# ULTIMATE!
import sys
print(sys.executable)

import cudf
import utils
import xarray
from download_glider_data import utils as dutils
import glidertools as gt
import hvplot.pandas  # noqa
import hvplot.cudf
import cmocean
import holoviews as hv
hv.extension('bokeh')

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

mode = 'all'
variables=['temperature', 'salinity', 'depth', 'potential_density', 'profile_num', 
           'profile_direction', 'chlorophyll', 'oxygen_concentration', 'longitude']
metadata = utils.load_metadata()
metadata = metadata[
    (metadata['basin']=='Bornholm Basin') &
    (metadata['time_coverage_start (UTC)'].dt.year==2023)
    ]
metadata = utils.drop_overlaps(metadata)
metadata = metadata#.iloc[0:12]

nrt_dataset_ids = list(metadata.index)
delayed_dataset_ids = [datasetid.replace('nrt', 'delayed') for datasetid in metadata.index]
all_dataset_ids = nrt_dataset_ids+delayed_dataset_ids#nrt_dataset_ids.extend(delayed_dataset_ids)

if mode=='nrt':
    dsdict = dutils.download_glider_dataset(dataset_ids=nrt_dataset_ids, variables=variables)
elif mode=='delayed':
    dsdict = dutils.download_glider_dataset(dataset_ids=delayed_dataset_ids,
                                            variables=variables)
elif mode=='all':
    dsdict = dutils.download_glider_dataset(dataset_ids=all_dataset_ids,
                                            variables=variables)

#for key in dsdict.keys():
#    dsdict[key] = gt.load.voto_seaexplorer_dataset(ds[variables])

    #dsdict.values = dsdict.values
#datasets_list = [gt.load.voto_seaexplorer_dataset(ds[variables]) for ds in [*dsdict.values()]]
#datasets_list = voto_concat_datasets(datasets_list)
dsdict = {key:gt.load.voto_seaexplorer_dataset(value) for key, value in dsdict.items()}

for key in dsdict.keys():
    df = dsdict[key].to_pandas().dropna(subset=['temperature', 'salinity'])
    df = df[['temperature', 'depth']].resample('1s').mean()
    dsdict[key] = df

ropts = dict(#colorbar=True, #width=350,
             toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],#, 'hover'], 
             default_tools=[], 
             active_tools=['xpan', 'xwheel_zoom'], 
             bgcolor="dimgrey")

cnorm = 'linear'
clim = tuple([0,20])#df.temperature.quantile([0.02, 0.98]).values)
width=1000
heigth=400

import pandas as pd
from holoviews.streams import RangeX
import numpy as np
from functools import reduce
from holoviews.streams import Pipe, Buffer
renderer = hv.renderer('bokeh')
import panel as pn


#df.index = cudf.to_datetime(df.index)

def get_xsection(x_range):
    (x0, x1) = x_range
    # print(x_range, x0)
    # import pdb; pdb.set_trace();
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    # grid timeline into n sections
    x_sampling = int(dtns/300)

    nrtkeys = [item for item in dsdict.keys() if item[0:3]=='nrt']
    meta = metadata.loc[nrtkeys]
    if (x1-x0)>np.timedelta64(90, 'D'):
        # activate sparse data mode to speed up reactivity
        # import pdb; pdb.set_trace();
        zoomed_out = True
        #x_sampling=8.64e13 # daily#30e12
        y_sampling=1
    else:
        # load delayed mode datasets for more detail
        delayedkeys = [item for item in dsdict.keys() if item[0:7]=='delayed']
        meta = metadata.loc[nrtkeys]
        zoomed_out = False
        x_sampling=8.64e13/24
        y_sampling=0.2

    # metadata.loc[dsdict.keys()]
    
    # import pdb; pdb.set_trace()
    # print(meta)
    #meta = meta[(meta['time_coverage_end (UTC)'].dt.date<x1) & (meta['time_coverage_start (UTC)'].dt.date<x0)]
    """
    meta = meta[((pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)>x0) & 
                (pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)<x1)) |
                ((pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)>x0) & 
                (pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)<x1))]

    """
    #meta = meta.loc['nrt_SEA055_M20']
    #print(
        # or it starts before our view and ends after
    #    (x0>=pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)),
    #    (x1<=pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)))
    #print(meta['time_coverage_start (UTC)'])
    #print(meta['time_coverage_end (UTC)'])
    buffer = np.timedelta64(30, 'D')
    meta = meta[
            #((pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)>=x0) & 
            #(pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)<=x1)) |
            # it starts within our view...
            (((pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)>=x0-buffer) & 
            (pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)<=x1+buffer)) |
            # or it ends within our view...
            ((pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)>=x0-buffer) & 
            (pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)<=x1+buffer))) #|
            # or it ends after our view but it starts before...
            #((x0>=pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)) &
            #(x1<=pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date))))
            # or it starts before our view and ends after
            #((pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)<=x0) & 
            #(pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)>=x1))
            ]
    
    #((pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)>=x0) & 
    #(pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)<=x1)) |
    # it starts within our view...
    

    
    #print(meta.index)

    
            # & (pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)<x0)]

    #import pdb; pdb.set_trace()
    #[datasetid.replace('nrt', 'delayed') for datasetid in metadata.index]
    if zoomed_out:
        dsids = [datasetid for datasetid in meta.index]
    else:
        dsids = [datasetid.replace('nrt', 'delayed') for datasetid in meta.index]
    print(dsids)
    datalist = [dsdict[key] for key in dsids]#df.loc[x0:x1]
    datalist = list(datalist)
    # import pdb; pdb.set_trace();
    
    
    #print(datalist)
    #return hv.Points(data, kdims=['time', 'depth'])
    #return hv.Points()
    plotslist = []
    for data in datalist:
        
        plotslist.append(data.hvplot.scatter(
            x='time', 
            y='depth',
            c='temperature',
            x_sampling=x_sampling,#20e12,#10e12,
            y_sampling=y_sampling,#5,#1.,
            flip_yaxis=True,
            dynamic=False,
            #hover=True,
            #hover_cols=['temperature'],
            cmap=cmocean.cm.thermal,
            width=800,
            height=400,
            #flip_yaxis=True,
            hover=False,
            rasterize=True,
            #cnorm=cnorm,
            #alpha=0.1,
            #clim=clim,
            ).opts(**ropts))
    
        """
        plotslist.append(data.hvplot.scatter(
            x='time', 
            y='depth',
            c='temperature',
            #x_sampling=2e12,#10e12,
            #y_sampling=0.1,#1.,
            flip_yaxis=True,
            dynamic=False,
            datashade=True,
            #hover=True,
            #hover_cols=['temperature'],
            cmap=cmocean.cm.thermal,
            width=width,
            height=heigth,
            #flip_yaxis=True,
            #rasterize=True,
            #cnorm=cnorm,
            #alpha=0.1,
            #clim=clim,
            ).opts(**ropts))
        """

    return reduce(lambda x, y: x*y, plotslist)
    

x_range=('2022-04-15', '2023-05-17')
range_stream = RangeX(x_range=(np.datetime64(x_range[0]), np.datetime64(x_range[1])))
#get_xsection(x_range)
finalplot = hv.DynamicMap(get_xsection, streams=[range_stream])
bokeh_server = pn.Row(finalplot).show(port=12345)
