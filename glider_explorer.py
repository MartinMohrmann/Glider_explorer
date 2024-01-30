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
import param

from download_glider_data import utils as dutils
import utils
import dictionaries

# unused imports
# import hvplot.pandas
# import cudf # works w. cuda, but slow.
# import hvplot.cudf

x_min_global = np.datetime64('2022-01-01')
x_max_global = np.datetime64('2024-01-01')


###### filter metadata to prepare download ##############
metadata = utils.filter_metadata()
metadata = metadata.drop('nrt_SEA067_M15', errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
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
             ylim=(-5,None)
            )

def plot_limits(plot, element):
    plot.handles['x_range'].min_interval = np.timedelta64(2, 'h')
    plot.handles['x_range'].max_interval = np.timedelta64(int(5*3.15e7), 's') # 5 years
    plot.handles['y_range'].min_interval = 10
    plot.handles['y_range'].max_interval = 500

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
        data):
    #print('cnorm in create_single_ds_plot_raster is', cnorm)
    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c='cplotvar',
        #cmap=cmocean.cm.thermal,
        #cnorm=cnorm,
        )
    #mraster = myrasterize(cnormvalue='linear', dmap_raster=raster)
    return raster


def load_viewport_datasets(x_range, metadata):
    (x0, x1) = x_range
    print('load_viewport_datasets:', x_range)
    global x_min_global
    global x_max_global
    x_min_global = x0
    x_max_global = x1
    #print(x_range, x_min_global, x_max_global)
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
        plt_props['subsample_freq']=2000
    elif (x1-x0)<np.timedelta64(1, 'D'):
        # activate sparse data mode to speed up reactivity
        zoomed_in = True
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = 1#int(dtns/10000)
        plt_props['y_sampling']=0.1
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=100
    else:
        # load delayed mode datasets for more detail
        zoomed_out = False
        plt_props['x_sampling']=8.64e13/24
        plt_props['y_sampling']=0.2
        plt_props['dynfontsize']=10
        plt_props['subsample_freq']=100
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


def get_xsection_raster():
    print('execute stream')
    variable = variable_widget.value
    plt_props = {}
    plt_props['x_sampling']=8.64e13/24
    plt_props['y_sampling']=0.2
    plt_props['dynfontsize']=10
    plt_props['subsample_freq']=1
    meta, plt_props = metadata, plt_props#load_viewport_datasets(x_range, metadata)
    plotslist1 = []
    varlist = [dsdict[dsid.replace('nrt', 'delayed')][
        ['time', 'depth', 'temperature', 'salinity',
        'potential_density', 'chlorophyll', 'oxygen_concentration']
        ] for dsid in meta.index]
    dsconc = xarray.concat(varlist, dim='time')
    dsconc['cplotvar'] = dsconc[variable]
    dsconc = dsconc.isel(time=slice(
        0,-1,plt_props['subsample_freq']), drop=True)#data.to_dask_dataframe().sample(0.1)
    mplt = create_single_ds_plot_raster(data=dsconc)
    # global combined
    # mplt = combined.apply.opts(cmap=dictionaries.cmap_dict['temperature'])
    return mplt#, combined

# on initial load, show all data
x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
         metadata['time_coverage_end (UTC)'].max().to_datetime64())
#x_min_global, x_max_global = x_range
range_stream = RangeX(x_range=x_range)

cnorm_widget = pn.widgets.Select(
    name="icnorm",
    value="linear", options=['linear', 'eq_hist'])
variable_widget = pn.widgets.Select(
    name="variable",
    value="temperature", options=[
        'temperature', 'salinity', 'potential_density',
        'chlorophyll','oxygen_concentration'])

class GliderExplorer(param.Parameterized):

    pick_variable = param.ObjectSelector(
        default='temperature', objects=['temperature', 'salinity'])
    pick_cnorm = param.ObjectSelector(
        default='linear', objects=['linear', 'eq_hist'])

    @param.depends('pick_cnorm', 'pick_variable')
    def create_dynmap(self):
        dmap_raster = hv.DynamicMap(
            get_xsection_raster,
            #streams=[range_stream],
            )
        # Bis hierher habe ich einen stream.
        # das problem ist das jede bind auswahl meinen
        # stream zerstört, folgende daten zwar dynamic maps sind
        # aber leider ohne stream?

        #return dmap_raster#.opts(xlim=(x_min_global, x_max_global))
        means = dsh.mean('cplotvar')
        dmap_rasterized = rasterize(dmap_raster,
                    aggregator=means,
                    x_sampling=8.64e13/48,
                    ).opts(
            invert_yaxis=True,
            colorbar=True,
            cmap=dictionaries.cmap_dict[self.pick_variable],#,cmap
            toolbar='above',
            tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
            default_tools=[],
            #responsive=True,
            width=800,
            height=400,
            cnorm=self.pick_cnorm,#cnorm_value,
            active_tools=['xpan', 'xwheel_zoom'],
            bgcolor="dimgrey",
            clabel=variable_widget.value)
        dmap = hv.DynamicMap(get_xsection, streams=[range_stream])
        print('create_dynmap_set_ranges:',x_min_global, x_max_global)
        return dmap_rasterized.opts(xlim=(x_min_global, x_max_global))



#dmap_rasterized_bound = pn.bind(
#    create_dynmap,
#    cnorm_widget,
#    variable_widget)
glider_explorer=GliderExplorer()

pn.Column(glider_explorer.param, glider_explorer.create_dynmap).show(
    title='VOTO SAMBA data',
    websocket_origin='*',
    port=12345)

#pn.Column(cnorm_widget,variable_widget, dmap_rasterized_bound).show(
#    title='VOTO SAMBA data',
#    websocket_origin='*',
#    port=12345)

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
