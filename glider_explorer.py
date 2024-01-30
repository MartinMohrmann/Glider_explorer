import xarray
import glidertools as gt
import hvplot.dask
import hvplot.xarray
import cmocean
import holoviews as hv
import pathlib
import pandas as pd
import datashader as dsh
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread, spread
from holoviews.streams import RangeX, param
import numpy as np
from functools import reduce
import panel as pn

from download_glider_data import utils as dutils
import utils
import dictionaries

# unused imports
# import hvplot.pandas
# import cudf # works w. cuda, but slow.
# import hvplot.cudf

###### filter metadata to prepare download ##############
metadata = utils.filter_metadata()
metadata = metadata.drop('nrt_SEA067_M15', errors='ignore')

all_dataset_ids = utils.add_delayed_dataset_ids(metadata) # hacky
#import pdb; pdb.set_trace();
#all_dataset_ids.remove('nrt_SEA067_M15')
#all_dataset_ids.remove('delayed_SEA067_M15') #!!!!!!!!!!!!!!!!!!!!!S
#checkbox = pn.widgets.Checkbox(name='Logscale', value=True)
#select = pn.widgets.Select(name='Select', options=[cmocean.cm.thermal, cmocean.cm.haline])
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
             #ylim=(-5, None),
             #autorange='y',
             ylim=(-5,None)
             #autorange='y',
            )

def plot_limits(plot, element):
    plot.handles['x_range'].min_interval = np.timedelta64(2, 'h')
    plot.handles['x_range'].max_interval = np.timedelta64(int(5*3.15e7), 's') # 5 years
    plot.handles['y_range'].min_interval = 10
    plot.handles['y_range'].max_interval = 500
    #plot.handles['x_range'].start = np.datetime64('2020-01-01')
    #plot.handles['x_range'].end = np.datetime64('2026-01-01')
    #plot.handles['y_range'].start = -10
    #plot.handles['y_range'].end = 500


x_min_global = np.datetime64('2020-01-01')
x_max_global = np.datetime64('2024-01-01')

#class Style(param.Parameterized):
#    # could be expanded and take other parameter if it works
#    colormap = param.ObjectSelector(
#        default=cmocean.cm.thermal,
#        objects=[cmocean.cm.thermal, cmocean.cm.haline])

def create_single_ds_plot(data, metadata, variable, dsid, plt_props):
    text_annotation = hv.Text(
        x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
        y=-2, text=dsid.replace('nrt_', ''),
        fontsize=plt_props['dynfontsize'],
            ).opts(text_opts
            ).opts(**ropts)

    startvline = hv.VLine(metadata.loc[dsid][
        'time_coverage_start (UTC)']).opts(color='grey', line_width=1)
    endvline = hv.VLine(metadata.loc[dsid][
        'time_coverage_end (UTC)']).opts(color='grey', line_width=1)
    # print(plt_props)
    return text_annotation*startvline*endvline


def create_single_ds_plot_raster(
        cnorm,
        data):
    print('cnorm in create_single_ds_plot_raster is', cnorm)
    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c='cplotvar',
        #cmap=cmocean.cm.thermal,
        cnorm=cnorm,
        )
    return raster


def load_viewport_datasets(x_range, metadata):
    (x0, x1) = x_range
    global x_min_global
    global x_max_global
    x_min_global = x0
    x_max_global = x1
    print(x_range, x_min_global, x_max_global)
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
        plt_props['subsample_freq']=5
    elif (x1-x0)<np.timedelta64(1, 'D'):
        # activate sparse data mode to speed up reactivity
        zoomed_in = True
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = 1#int(dtns/10000)
        plt_props['y_sampling']=0.1
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=1
    else:
        # load delayed mode datasets for more detail
        zoomed_out = False
        plt_props['x_sampling']=8.64e13/24
        plt_props['y_sampling']=0.2
        plt_props['dynfontsize']=10
        plt_props['subsample_freq']=1
    return meta, plt_props


def get_xsection(x_range):
    variable='temperature'
    meta, plt_props = load_viewport_datasets(x_range, metadata)
    plotslist = []
    # note: only the first plot in the list needs the **ropts. Everything else migh
    for dsid in meta.index:
        #data=dsdict[dsid] if zoomed_out else dsdict[dsid.replace('nrt', 'delayed')]
        data=dsdict[dsid.replace('nrt', 'delayed')]
        single_plot = create_single_ds_plot(
            data, metadata, variable, dsid, plt_props)
        plotslist.append(single_plot)
    return reduce(lambda x, y: x*y, plotslist)


def get_xsection_raster(x_range, variable, cnorm):
    meta, plt_props = load_viewport_datasets(x_range, metadata)
    plotslist1 = []
    varlist = [dsdict[dsid.replace('nrt', 'delayed')][
        ['time', 'depth', 'temperature', 'salinity']] for dsid in meta.index]
    dsconc = xarray.concat(varlist, dim='time')
    dsconc['cplotvar'] = dsconc[variable]
    dsconc = dsconc.isel(time=slice(
        0,-1,plt_props['subsample_freq']), drop=True)#data.to_dask_dataframe().sample(0.1)
    mplt = create_single_ds_plot_raster(cnorm=cnorm, data=dsconc)
    # global combined
    # mplt = combined.apply.opts(cmap=dictionaries.cmap_dict['temperature'])
    return mplt#, combined

# on initial load, show all data
x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
         metadata['time_coverage_end (UTC)'].max().to_datetime64())
range_stream = RangeX(x_range=x_range)

def create_dynmap():
    # Change of global variables because range_stream does
    # not seem to work with passing variables (how?, why?)

    dmap = hv.DynamicMap(get_xsection,
    streams=[range_stream],).opts(hooks=[plot_limits])
    dmap_raster = hv.DynamicMap(get_xsection_raster,
    kdims=['variable', 'cnorm'],
    streams=[range_stream],).redim.values(
        variable=('temperature', 'salinity'),
        cnorm=('linear', 'eq_hist'))
    return dmap_raster, dmap

# style = Style()
means = dsh.mean('cplotvar')
dmap_raster, dmap = create_dynmap()
variable_widget = pn.widgets.Select(
    name="icnorm",
    value="linear", options=['linear', 'eq_hist'])


def myrasterize(cnormvalue,dmap_raster, dmap):
    # global dmap_rasterized
    dmap_rasterized = rasterize(dmap_raster,
                aggregator=means,
                x_sampling=8.64e13/48,
                ).opts(
        invert_yaxis=True,
        colorbar=True,
        cmap=cmocean.cm.thermal,#,cmap
        toolbar='above',
        tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
        default_tools=[],
        #responsive=True,
        width=800,
        height=400,
        cnorm=cnormvalue,
        active_tools=['xpan', 'xwheel_zoom'],
        bgcolor="dimgrey",)#.opts(**ropts,hooks=[plot_limits])

    #dmap_rasterized = dmap_rasterized.apply.opts(cmap=[dictionaries.cmap_dict['salinity'], dictionaries.cmap_dict['temperature']])
    combined = dmap_rasterized.apply.opts(cmap=dictionaries.cmap_dict['salinity'])
    combined = dmap_rasterized*dmap
    combined.opts(xlim=(x_min_global, x_max_global))
    return combined

# Here I could implement if I want a (spread-) scatterplot if zoomed in a lot.
# That could be shade instead of rasterize I believe? Or datashade? and then spread.
# combined = myrasterize('linear', dmap_raster, dmap)#.redim.values(cnorm=select)


#mholomap = hv.HoloMap(dmap_raster)
#pn.Column(select, variable_widget, combined).show(port=12345)


dmap_rasterized_bound = pn.bind(
    myrasterize,
    variable_widget,
    dmap_raster,
    dmap)


pn.Column(variable_widget, dmap_rasterized_bound).show(
    title='VOTO SAMBA data',
    websocket_origin='*',
    port=12345)


"""
Future development ideas:
* activate hover (for example dataset details, sensor specs, or point details)
* holoviews autoupdate for development
* write tests including timings benchmark for development
* implement async functionen documented in holoviews to not disturb user interaction
* throw out X_range_stream (possibly) and implement full data dynamic sampling instead. One solution could be to use a dynamic .sample(frac=zoomstufe)
* plot glidertools gridded data instead (optional, but good for interpolation)...
...
"""
