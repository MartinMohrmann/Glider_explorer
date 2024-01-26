import xarray
import glidertools as gt
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

# unused imports
# import pandas as pd
# import hvplot.pandas
# import cudf # works w. cuda, but slow.
# import hvplot.cudf
# import holoviews as hv
import datashader as dsh
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread, spread



from download_glider_data import utils as dutils
import utils

#global cnorm
#cnorm='linear'


###### filter metadata to prepare download ##############
metadata = utils.filter_metadata()
all_dataset_ids = utils.add_delayed_dataset_ids(metadata) # hacky

###### download actual data ##############################
dutils.cache_dir = pathlib.Path('../voto_erddap_data_cache')
variables=['temperature', 'salinity', 'depth',
           'potential_density', 'profile_num',
           'profile_direction', 'chlorophyll',
           'oxygen_concentration', 'longitude']
dsdict = dutils.download_glider_dataset(dataset_ids=all_dataset_ids,
                                        variables=variables)

####### specify global plot variables ####################
#clim = tuple([0,20])#df.temperature.quantile([0.02, 0.98]).values) # beware, probably not global!
#df.index = cudf.to_datetime(df.index)
text_opts  = hv.opts.Text(text_align='left', text_color='gray')
ropts = dict(
             toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
             default_tools=[],
             active_tools=['xpan', 'xwheel_zoom'],
             bgcolor="dimgrey",
             )





def create_single_ds_plot(data, metadata, variable, dsid, plt_props):
    text_annotation = hv.Text(
        x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
        y=-2, text=dsid.replace('nrt_', ''),
        fontsize=plt_props['dynfontsize'],
            ).opts(text_opts
            ).opts(**ropts)

    startvline = hv.VLine(metadata.loc[dsid]['time_coverage_start (UTC)']).opts(color='grey')
    endvline = hv.VLine(metadata.loc[dsid]['time_coverage_end (UTC)']).opts(color='grey')
    # print(plt_props)
    return text_annotation*startvline*endvline

def create_single_ds_plot_raster(
        cnorm,
        data):
        #metadata=metadata,
        #variable=variable,
        #plt_props=plt_probs):
    # import pdb; pdb.set_trace()
    print('cnorm in create_single_ds_plot_raster is', cnorm)
    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c='cplotvar',
        cmap=cmocean.cm.thermal,
        cnorm=cnorm,

        #cnorm=icnorm.value
        )
    # print(icnorm)
    return raster


def get_xsection(x_range):
    variable='temperature'
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    plt_props = {}
    #delayedkeys = [item for item in dsdict.keys() if item[0:7]=='delayed']
    #nrtkeys =  [item for item in dsdict.keys() if item[0:3]=='nrt']
    meta = metadata[
            # x0 and x1 are the time start and end of our view, the other times
            # are the start and end of the individual datasets. To increase
            # perfomance, datasets are loaded only if visible, so if
            # 1. it starts within our view...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)<=x1)) |
            # 2. it ends within our view...
            ((pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)<=x1)) |
            # 3. it starts before and ends after our view (zoomed in)...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)<=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)>=x1)) |
            # 4. or it both, starts and ends within our view (zoomed out)...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)<=x1))
            ]
    # zoomed_out = False
    # zoomed_in = False
    print(f'len of meta is {len(meta)} in get_xsection')
    if (x1-x0)>np.timedelta64(180, 'D'):
        # activate sparse data mode to speed up reactivity
        zoomed_out = True
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = int(dtns/1000)
        plt_props['y_sampling']=1
        plt_props['dynfontsize']=4
    elif (x1-x0)<np.timedelta64(1, 'D'):
        # activate sparse data mode to speed up reactivity
        zoomed_in = True
        #variable='temperatu'
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = 1#int(dtns/10000)
        plt_props['y_sampling']=0.1
        plt_props['dynfontsize']=4
    else:
        # load delayed mode datasets for more detail
        zoomed_out = False
        plt_props['x_sampling']=8.64e13/24
        plt_props['y_sampling']=0.2
        plt_props['dynfontsize']=10

    plotslist = []
    # note: only the first plot in the list needs the **ropts. Everything else migh
    for dsid in meta.index:
        data=dsdict[dsid] if zoomed_out else dsdict[dsid.replace('nrt', 'delayed')]
        #data = data.isel(time=slice(0,-1,10), drop=True)#data.to_dask_dataframe().sample(0.1)
        single_plot = create_single_ds_plot(data, metadata, variable, dsid, plt_props)
        plotslist.append(single_plot)
    return reduce(lambda x, y: x*y, plotslist)



def get_xsection_raster(x_range, variable, cnorm):
    #global cnorm
    #cnorm=cnorm
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    plt_props = {}
    #delayedkeys = [item for item in dsdict.keys() if item[0:7]=='delayed']
    #nrtkeys =  [item for item in dsdict.keys() if item[0:3]=='nrt']
    meta = metadata[
            # x0 and x1 are the time start and end of our view, the other times
            # are the start and end of the individual datasets. To increase
            # perfomance, datasets are loaded only if visible, so if
            # 1. it starts within our view...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)<=x1)) |
            # 2. it ends within our view...
            ((pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)<=x1)) |
            # 3. it starts before and ends after our view (zoomed in)...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)<=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)>=x1)) |
            # 4. or it both, starts and ends within our view (zoomed out)...
            ((pd.to_datetime(metadata['time_coverage_start (UTC)'].dt.date)>=x0) &
            (pd.to_datetime(metadata['time_coverage_end (UTC)'].dt.date)<=x1))
            ]
    #zoomed_out = False
    #zoomed_in = False
    print(f'len of meta is {len(meta)} in get_xsection_raster')
    if (x1-x0)>np.timedelta64(180, 'D'):
        # activate sparse data mode to speed up reactivity
        zoomed_out = True
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = int(dtns/1000)
        plt_props['y_sampling']=1
        plt_props['dynfontsize']=4
    elif (x1-x0)<np.timedelta64(1, 'D'):
        # activate sparse data mode to speed up reactivity
        zoomed_in = True
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = 1#int(dtns/10000)
        plt_props['y_sampling']=0.1
        plt_props['dynfontsize']=4
    else:
        # load delayed mode datasets for more detail
        zoomed_out = False
        plt_props['x_sampling']=8.64e13/24
        plt_props['y_sampling']=0.2
        plt_props['dynfontsize']=10

    plotslist = []
    varlist = [dsdict[dsid.replace('nrt', 'delayed')][['time', 'depth', 'temperature', 'salinity']] for dsid in meta.index]
    dsconc = xarray.concat(varlist, dim='time')
    dsconc['cplotvar'] = dsconc[variable]
    #mplt = pn.bind(
    #    create_single_ds_plot_raster,
    #    icnorm=variable_widget,
    #    data=dsconc)

    mplt = create_single_ds_plot_raster(cnorm=cnorm, data=dsconc)
    """
    dmap_raster = rasterize(mplt,
                    aggregator='mean',
                    ).opts(
            invert_yaxis=True,
            colorbar=True,
            cmap=cmocean.cm.thermal,
            toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
            default_tools=[],
            responsive=True,
            height=400,
            cnorm=variable_widget.value,
            active_tools=['xpan', 'xwheel_zoom'],
            bgcolor="dimgrey",).opts(**ropts)
    """
    #mplt = rasterize(mplt)
    return mplt#dmap_raster






# on initial load, show all data
x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
         metadata['time_coverage_end (UTC)'].max().to_datetime64())
range_stream = RangeX(x_range=x_range)

def create_dynmap():
    # Change of global variables because range_stream does
    # not seem to work with passing variables (how?, why?)

    dmap = hv.DynamicMap(get_xsection,
    streams=[range_stream],)
    dmap_raster = hv.DynamicMap(get_xsection_raster,
    kdims=['variable', 'cnorm'],
    streams=[range_stream],).redim.values(
        variable=('temperature', 'salinity'),
        cnorm=('linear', 'eq_hist'))
    return (dmap, dmap_raster)

# one solution to the reset problem could be to change global values, e.g. set variable to "selected temperature" globally
#def combine_aggregate_dmaps(icnorm):
#    dmap, dmap_raster = create_dynmap() # 'linear'
#    dmap_raster = dmap_raster.redim.values(
#            variable=('temperature', 'salinity'))
means = dsh.mean('cplotvar')
#    print('icnorm in combine_aggregate_dmaps is', variable_widget.value)
#
#    dmap_combined = dmap_raster*dmap
#   return dmap_combined
dmap, dmap_raster = create_dynmap()

#slider = pn.widgets.FloatSlider(name='Amplitude', start=0, end=100)
variable_widget = pn.widgets.Select(
    name="icnorm",
    value="linear", options=['linear', 'eq_hist'])


#@pn.depends(variable_widget, watch=True)
def myrasterize(cnormvalue,dmap_raster, dmap):
    dmap_rasterized = rasterize(dmap_raster,
                aggregator=means,
                ).opts(
        #framewise=True,
        invert_yaxis=True,
        colorbar=True,
        cmap=cmocean.cm.thermal,
        toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
        default_tools=[],
        responsive=True,
        height=400,
        cnorm=cnormvalue,
        active_tools=['xpan', 'xwheel_zoom'],
        bgcolor="dimgrey",).opts(**ropts)
    return dmap_rasterized*dmap

dmap_rasterized = pn.bind(
    myrasterize,
    variable_widget,
    dmap_raster,
    dmap)

pn.Column(variable_widget, dmap_rasterized).show(port=12345)
#import pdb; pdb.set_trace()
#dmap_combined = dmap*myrasterize(dmap_raster)
#dmap_combined=combine_aggregate_dmaps(variable_widget)
#dmap_combined = pn.bind(
#    combine_aggregate_dmaps,
#    icnorm=variable_widget)
#dynamic map bin function holoviews
#pn.Column(variable_widget, dmap_combined).show(port=12345)

"""
Future development ideas:
* activate hover (for example dataset details, sensor specs, or point details)
* holoviews autoupdate for development
* write tests including timings benchmark for development
* implement async functionen documented in holoviews to not disturb user interaction
* throw out X_range_stream (possibly) and implement full data dynamic sampling instead. One solution could be to use a dynamic .sample(frac=zoomstufe)
...
"""
