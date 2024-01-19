# ToDo/Try panel/holoviews autoupdate
#
import xarray
import glidertools as gt
import hvplot.pandas
import hvplot.dask
import hvplot.xarray
import cmocean
import holoviews as hv
import pathlib
import pandas as pd
from holoviews.streams import RangeX
import numpy as np
from functools import reduce
import panel as pn
import param
# import cudf # works w. cuda, but slow.
# import hvplot.cudf

from download_glider_data import utils as dutils
import utils

########## download and filter metadata ##################
mode = 'all' # 'nrt', 'delayed'
variables=['temperature', 'salinity', 'depth',
           'potential_density', 'profile_num',
           'profile_direction', 'chlorophyll',
           'oxygen_concentration', 'longitude']
metadata = utils.load_metadata()
metadata = metadata[
    (metadata['basin']=='Bornholm Basin') &
    (metadata['time_coverage_start (UTC)'].dt.year==2022) &
    (metadata['time_coverage_start (UTC)'].dt.month<3)
    ]

metadata = utils.drop_overlaps(metadata)
nrt_dataset_ids = list(metadata.index)
delayed_dataset_ids = [
    datasetid.replace('nrt', 'delayed') for datasetid in metadata.index]
all_dataset_ids = nrt_dataset_ids+delayed_dataset_ids


###### download actual data ##############################
dutils.cache_dir = pathlib.Path('../voto_erddap_data_cache')
dsdict = dutils.download_glider_dataset(dataset_ids=all_dataset_ids,
                                        variables=variables)
#import pdb; pdb.set_trace();

# dsdict = {key:gt.load.voto_seaexplorer_dataset(value) for key, value in dsdict.items()}

"""
for key in dsdict.keys():
    df = dsdict[key].to_pandas().dropna(subset=['temperature', 'salinity'])
    df = df[['temperature', 'depth']].resample('1s').mean()
    dsdict[key] = df
"""

#cnorm = 'linear'
#clim = tuple([0,20])#df.temperature.quantile([0.02, 0.98]).values)

#df.index = cudf.to_datetime(df.index)

ropts = dict(#colorbar=True, #width=350,
             toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],#, 'hover'],
             default_tools=[],
             active_tools=['xpan', 'xwheel_zoom'],
             bgcolor="dimgrey",
             #framewise=True
             )


def get_xsection(x_range, variable):
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    # cnorm='linear'
    # variable='temperature'
    #cnorm='eq_hist' if cnorm else 'linear'
    #if cnorm=='eq_hist':
    #    dmap.reset()
    nrtkeys = [item for item in dsdict.keys() if item[0:3]=='nrt']
    meta = metadata.loc[nrtkeys]
    if (x1-x0)>np.timedelta64(180, 'D'):
        # grid timeline into n sections
        x_sampling = int(dtns/1000)
        # activate sparse data mode to speed up reactivity
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

    meta = meta[
            # x0 and x1 are the time start and end of our view, the other times
            # are the start and end of the individual datasets. To increase
            # perfomanceThe dataset is loaded only if visible, so if
            # 1. it starts within our view...
            ((pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)<=x1)) |
            # 2. it ends within our view...
            ((pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)<=x1)) |
            # 3. it starts before and ends after our view (zoomed in)...
            ((pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)<=x0) &
            (pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)>=x1)) |
            # 4. or it both, starts and ends within our view (zoomed out)...
            ((pd.to_datetime(meta['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(meta['time_coverage_end (UTC)'].dt.date)<=x1))
            ]
    text_opts  = hv.opts.Text(text_align='left', text_color='gray')
    # backend_opts = {"plot.toolbar.autohide": True}
    if zoomed_out:
        print([element for element in list(metadata.index)])
        dynfontsize=4
    else:
        print([element.replace('nrt', 'delayed') for element in list(metadata.index)])
        dynfontsize=10
    plotslist = []
    # note: only the first plot in the list needs the **ropts. Everything else might be overwritten
    print(variable)
    print(cnorm)
    for dsid in meta.index:
        if zoomed_out:
            data=dsdict[dsid]
        else:
            data=dsdict[dsid.replace('nrt', 'delayed')]

        plotslist.append(hv.Text(
            x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
            y=-2, text=dsid.replace('nrt_', ''),
            fontsize=dynfontsize,
                ).opts(text_opts
                ).opts(**ropts))

        plotslist.append(hv.VLine(metadata.loc[dsid]['time_coverage_start (UTC)']).opts(color='grey'))
        plotslist.append(hv.VLine(metadata.loc[dsid]['time_coverage_end (UTC)']).opts(color='grey'))

        # future development ideas: activate hover
        plotslist.append(data.hvplot.scatter(
            x='time',
            y='depth',
            c=variable,
            x_sampling=x_sampling,
            y_sampling=y_sampling,
            flip_yaxis=True,
            dynamic=False,
            cmap=cmocean.cm.thermal,
            #width=800,
            height=400,
            responsive=True,
            rasterize=True,
            cnorm=cnorm,#cnorm,
            #clim=clim,
            )
        )
    return reduce(lambda x, y: x*y, plotslist)


x_range=('2022-01-15', '2022-12-17')
range_stream = RangeX(x_range=(np.datetime64(x_range[0]), np.datetime64(x_range[1])))

variable_widget = pn.widgets.Select(
    name="icnorm",
    value="linear", options=['linear', 'eq_hist'])

def create_dynmap(icnorm):
    global cnorm
    cnorm = variable_widget.value
    dmap = hv.DynamicMap(get_xsection,
    kdims=['variable'],#, 'cnorm'],
    streams=[range_stream],
    )
    return dmap.redim.values(
        variable=('temperature', 'salinity'))

# one solution to the reset problem could be to change global values, e.g. set variable to "selected temperature" globally
dmap = pn.bind(
        create_dynmap,
        icnorm=variable_widget)

pn.Column(variable_widget,dmap).show(port=12345)
