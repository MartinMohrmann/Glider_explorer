import xarray
import glidertools as gt
#import hvplot.dask
#import hvplot.xarray
import hvplot.pandas
import cmocean
import holoviews as hv
import pathlib
import pandas as pd
import datashader as dsh
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread, spread
from holoviews.operation import decimate
from holoviews.streams import RangeX
import numpy as np
from functools import reduce
import panel as pn
import param
import datashader.transfer_functions as tf
import time

from download_glider_data import utils as dutils
import utils
import dictionaries

# unused imports
# import hvplot.pandas
#import cudf # works w. cuda, but slow.
try:
    import hvplot.cudf
except:
    print('no cudf available, that is fine but slower')

###### filter metadata to prepare download ##############
metadata, all_datasets = utils.filter_metadata()
metadata = metadata.drop(['nrt_SEA067_M15', 'nrt_SEA079_M14', 'nrt_SEA061_M63'], errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
all_dataset_ids = utils.add_delayed_dataset_ids(metadata, all_datasets) # hacky

###### download actual data ##############################
dutils.cache_dir = pathlib.Path('../voto_erddap_data_cache')
variables=['temperature', 'salinity', 'depth',
           'potential_density', 'profile_num',
           'profile_direction', 'chlorophyll',
           'oxygen_concentration', 'longitude']
dsdict = dutils.download_glider_dataset(dataset_ids=all_dataset_ids,
                                        variables=variables)
#import pdb; pdb.set_trace();

####### specify global plot variables ####################
#df.index = cudf.to_datetime(df.index)
text_opts  = hv.opts.Text(text_align='left', text_color='black') #OOOOOOOOOOOOOOO
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

def create_single_ds_plot(data, metadata, variable, dsid, plt_props):
    text_annotation = hv.Text(
        x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
        y=-2, text=dsid.replace('nrt_', ''),
        fontsize=plt_props['dynfontsize'],
            ).opts(**ropts).opts(text_opts)

    startvline = hv.VLine(metadata.loc[dsid][
        'time_coverage_start (UTC)']).opts(color='grey', line_width=1)
    endvline = hv.VLine(metadata.loc[dsid][
        'time_coverage_end (UTC)']).opts(color='grey', line_width=1)
    return text_annotation*startvline*endvline


def create_single_ds_plot_raster(
        data):
    t1 = time.perf_counter()
    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c='cplotvar',
        )
    t2 = time.perf_counter()
    return raster


def load_viewport_datasets(x_range):
    t1 = time.perf_counter()
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    plt_props = {}
    meta = metadata[metadata['basin']==currentobject.pick_basin]
    meta = meta[
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

    print(f'len of meta is {len(meta)} in load_viewport_datasets')
    if (x1-x0)>np.timedelta64(360, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=50
    elif (x1-x0)>np.timedelta64(180, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=20
    elif (x1-x0)<np.timedelta64(1, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=1
    else:
        # load delayed mode datasets for more detail
        plt_props['zoomed_out'] = False
        plt_props['dynfontsize']=10
        plt_props['subsample_freq']=1
    t2 = time.perf_counter()
    return meta, plt_props


def get_xsection():
    t1 = time.perf_counter()
    variable='temperature'
    meta, plt_props = load_viewport_datasets((x_min_global,x_max_global))
    plotslist = []
    for dsid in meta.index:
        # this is just plotting lines and meta, no need for 'delayed' data (?)
        data=dsdict[dsid]
        single_plot = create_single_ds_plot(
            data, metadata, variable, dsid, plt_props)
        plotslist.append(single_plot)
    t2 = time.perf_counter()
    return reduce(lambda x, y: x*y, plotslist)


def get_xsection_mld(x_range):
    t1 = time.perf_counter()
    variable='temperature'
    meta, plt_props = load_viewport_datasets(x_range)
    # activate this for high delayed resolution
    # metakeys = [element if plt_props['zoomed_out'] else element.replace('nrt', 'delayed') for element in meta.index]
    metakeys = meta.index
    dslist = utils.voto_concat_datasets(varlist)
    dslist = [utils.add_dive_column(ds) for ds in dslist]
    plotslist = []
    for ds in dslist:
        mld = gt.physics.mixed_layer_depth(ds.to_xarray(), 'temperature', thresh=0.1, verbose=False, ref_depth=10)
        gtime = ds.reset_index().groupby(by='profile_num').mean().time
        #gt.utils.group_by_profiles(ds, variables=['time', 'temperature']).mean().time.values
        gmld = mld.values
        dfmld = pd.DataFrame.from_dict(dict(time=gtime, mld=gmld))
        #dfmld['mld'] = dfmld.mld.rolling(window=10, min_periods=5, center=True).mean()
        mldscatter = dfmld.hvplot.line(
            x='time',
            y='mld',
            color='white',
            alpha=0.5,
        )
        plotslist.append(mldscatter)
    t2 = time.perf_counter()
    return reduce(lambda x, y: x*y, plotslist)


def get_xsection_raster(x_range):
    (x0, x1) = x_range
    global x_min_global
    global x_max_global
    x_min_global = x0
    x_max_global = x1
    meta, plt_props = load_viewport_datasets(x_range)
    plotslist1 = []
    #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
    # activate this for high res data
    if plt_props['zoomed_out']:
        metakeys = [element.replace('nrt', 'delayed') for element in meta.index]
    else:
        metakeys = [element.replace('nrt', 'delayed') if
            element.replace('nrt', 'delayed') in all_datasets.index else element for element in meta.index]

    varlist = [dsdict[dsid] for dsid in metakeys]
    dsconc = pd.concat(varlist)
    dsconc['cplotvar'] = dsconc[currentobject.pick_variable]
    dsconc = dsconc.iloc[0:-1:plt_props['subsample_freq']]
    mplt = create_single_ds_plot_raster(data=dsconc)
    t2 = time.perf_counter()
    return mplt


def get_xsection_points(x_range):
    # currently not activated, but almost completely working.
    # only had some slight problems to keep zoom settings on variable change,
    # but that should be easy to solve...

    (x0, x1) = x_range

    if (x1-x0)<np.timedelta64(4, 'D'):
        meta, plt_props = load_viewport_datasets(x_range)
        plotslist1 = []
        #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
        metakeys = [element if plt_props['zoomed_out'] else element.replace('nrt', 'delayed') for element in meta.index]
        varlist = [dsdict[dsid] for dsid in metakeys]
        dsconc = pd.concat(varlist)
        dsconc['cplotvar'] = dsconc[glider_explorer.pick_variable]
        points = dsconc.hvplot.points(
            x='time',
            y='depth',
            c='cplotvar',
            )
    else:
        dsconc = pd.DataFrame.from_dict(dict(time=[x0], depth=[0], cplotvar=[10]))
        points = dsconc.hvplot.points(
            x='time',
            y='depth',
            c='cplotvar',
            )
    return points


# on initial load, show all data
x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
         metadata['time_coverage_end (UTC)'].max().to_datetime64())


global x_min_global
global x_max_global
x_min_global, x_max_global = x_range

class GliderExplorer(param.Parameterized):

    pick_variable = param.ObjectSelector(
        default='temperature', objects=[
        'temperature', 'salinity', 'potential_density',
        'chlorophyll','oxygen_concentration'])
    pick_basin = param.ObjectSelector(
        default='Bornholm Basin', objects=[
        'Bornholm Basin', 'Eastern Gotland',
        'Western Gotland', 'Skagerrak, Kattegat',
        'Åland Sea']
    )
    pick_cnorm = param.ObjectSelector(
        default='linear', objects=['linear', 'eq_hist'])
    pick_aggregation = param.ObjectSelector(
        default='mean', objects=['mean', 'std', 'var'])
    pick_mld = param.Boolean(
        default=False)
    #button_inflow = pn.widgets.Button(name='Tell me about inflows', button_type='primary')

    #@param.depends('pick_cnorm','pick_variable', 'pick_basin', 'pick_aggregation', 'pick_mld') # outcommenting this means just depend on all, redraw always

    def create_dynmap(self):
        global currentobject
        currentobject = self
        t1 = time.perf_counter()
        x_range=(x_min_global,
                 x_max_global)

        range_stream = RangeX(x_range=x_range)
        pick_cnorm='linear'

        dmap_raster = hv.DynamicMap(
            get_xsection_raster,
            streams=[range_stream],
            #cache_size=1,)
        )

        if self.pick_mld:
            dmap_mld = hv.DynamicMap(get_xsection_mld, streams=[range_stream], cache_size=1)
        if self.pick_aggregation=='mean':
            means = dsh.mean('cplotvar')
        if self.pick_aggregation=='std':
            means = dsh.std('cplotvar')
        if self.pick_aggregation=='var':
            means = dsh.var('cplotvar')
        dmap_rasterized = rasterize(dmap_raster,
                    aggregator=means,
                    x_sampling=8.64e13/24,
                    y_sampling=.5,
                    ).opts(
            #alpha=0.2,
            invert_yaxis=True,
            colorbar=True,
            cmap=dictionaries.cmap_dict[self.pick_variable],#,cmap
            toolbar='above',
            tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
            default_tools=[],
            #responsive=True,
            width=800,
            height=400,
            cnorm=self.pick_cnorm,
            active_tools=['xpan', 'xwheel_zoom'],
            bgcolor="dimgrey",
            clabel=self.pick_variable)

        dmap = hv.DynamicMap(get_xsection, cache_size=1)
        t2 = time.perf_counter()

        """
        THIS PART IS WORKING PERFECTLY FINE, should reactivate itm but check zoom ranges on variable change...
        dmap_points = hv.DynamicMap(
            get_xsection_points,
            streams=[range_stream],
            cache_size=1
            )
        dmap_points = spread(datashade(
            dmap_points,
            aggregator=means,
            cnorm=self.pick_cnorm,
            cmap=dictionaries.cmap_dict[self.pick_variable],), px=4).opts(
                invert_yaxis=True,
                toolbar='above',
                tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
                default_tools=[],
                width=800,
                height=400,
                active_tools=['xpan', 'xwheel_zoom'],
                bgcolor="dimgrey",)
        """


        if self.pick_mld:
            #return (dmap_rasterized*dmap_points).opts(xlim=(x_min_global, x_max_global))*dmap*dmap_mld
            return (dmap_rasterized).opts(xlim=(x_min_global, x_max_global))*dmap*dmap_mld
        else:
            #return (dmap_rasterized*dmap_points).opts(xlim=(x_min_global, x_max_global))*dmap
            return (dmap_rasterized).opts(xlim=(x_min_global, x_max_global))*dmap
        #return dmap*dmap_mld


glider_explorer=GliderExplorer()

# usefull to create secondary plot, but not fully indepentently working yet:
# glider_explorer2=GliderExplorer()


pn.Column(
    pn.Row(
        glider_explorer.param,
        glider_explorer.create_dynmap),).show(
    title='VOTO SAMBA data',
    websocket_origin='*',
    port=12345,
    #admin=True,
    #profiler=True
    )

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
* good example to follow is the AdvancedStockExplorer class in the documentation
* add secondary plot or the option for secondary linked plot
* disentangle interactivity, so that partial refreshes (e.g. mixed layer calculation only) don't trigger complete refresh
* otpimal colorbar range (percentiles?)
* on selection of a new basin, I should reset the ranges. Otherwise it could come up with an error when changing while having unavailable x_range.
...
"""
