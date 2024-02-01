import xarray
import glidertools as gt
import hvplot.dask
import hvplot.xarray
import hvplot.pandas
import cmocean
import holoviews as hv
import pathlib
import pandas as pd
import datashader as dsh
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread, spread
from holoviews.streams import RangeX
import numpy as np
from functools import reduce
import panel as pn
import param
import datashader.transfer_functions as tf
pn.extension('plotly')

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
metadata = utils.filter_metadata()
metadata = metadata.drop(['nrt_SEA067_M15', 'nrt_SEA079_M14', 'nrt_SEA061_M63'], errors='ignore') #!!!!!!!!!!!!!!!!!!!! # temporary data inconsistency
all_dataset_ids = utils.add_delayed_dataset_ids(metadata) # hacky

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
    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c='cplotvar',
        )
    return raster


def load_viewport_datasets(x_range):
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    plt_props = {}
    #delayedkeys = [item for item in dsdict.keys() if item[0:7]=='delayed']
    #nrtkeys =  [item for item in dsdict.keys() if item[0:3]=='nrt']
    meta = metadata[metadata['basin']==glider_explorer.pick_basin]
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
    #import pdb; pdb.set_trace();
    #variable = variable_widget.value

    #print('loading data for basin', basin_widget.value)
    #zoomed_out = False
    #zoomed_in = False
    print(f'len of meta is {len(meta)} in load_viewport_datasets')
    if (x1-x0)>np.timedelta64(360, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = True
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = 8.64e13#int(dtns/1000)
        plt_props['y_sampling']=1
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=1
    elif (x1-x0)>np.timedelta64(180, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = 8.64e13/2#int(dtns/1000)
        plt_props['y_sampling']=1
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=1
    elif (x1-x0)<np.timedelta64(1, 'D'):
        # activate sparse data mode to speed up reactivity
        plt_props['zoomed_out'] = False
        #x_sampling=8.64e13 # daily
        # grid timeline into n sections
        plt_props['x_sampling'] = 1#int(dtns/10000)
        plt_props['y_sampling']=0.1
        plt_props['dynfontsize']=4
        plt_props['subsample_freq']=1
    else:
        # load delayed mode datasets for more detail
        plt_props['zoomed_out'] = False
        plt_props['x_sampling']=8.64e13/24
        plt_props['y_sampling']=0.2
        plt_props['dynfontsize']=10
        plt_props['subsample_freq']=1
    #import pdb; pdb.set_trace();
    return meta, plt_props


def get_xsection():
    variable='temperature'
    meta, plt_props = load_viewport_datasets((x_min_global,x_max_global))
    plotslist = []
    for dsid in meta.index:
        #this is just plotting lines and meta, so I don't need 'delayed'
        #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
        data=dsdict[dsid]
        single_plot = create_single_ds_plot(
            data, metadata, variable, dsid, plt_props)
        plotslist.append(single_plot)
    return reduce(lambda x, y: x*y, plotslist)


def get_xsection_mld(x_range):
    print('EXCUTE MLD')
    variable='temperature'
    meta, plt_props = load_viewport_datasets(x_range)
    metakeys = [element if plt_props['zoomed_out'] else element.replace('nrt', 'delayed') for element in meta.index]
    #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
    varlist = [utils.voto_seaexplorer_dataset(dsdict[dsid]) for dsid in meta.index]
    #dsconc = xarray.concat(varlist, dim='time')
    #dsconc = utils.voto_seaexplorer_dataset(dsconc)
    dslist = utils.voto_concat_datasets(varlist)
    dslist = [utils.add_dive_column(ds) for ds in dslist]
    # import pdb; pdb.set_trace();
    #dsconc = xarray.concat(dsconc, dim='time')
    #dsconc['dives'] = dsconc['profile_num']
    plotslist = []
    for ds in dslist:
        mld = gt.physics.mixed_layer_depth(ds, 'temperature', thresh=0.1, verbose=False, ref_depth=10)
        #import pdb; pdb.set_trace();
        ds = ds.to_pandas() # alternatively to cudf?
        #import pdb; pdb.set_trace();
        gtime = ds.reset_index().groupby(by='profile_num').mean().time#gt.utils.group_by_profiles(ds, variables=['time', 'temperature']).mean().time.values
        gmld = mld.values
        dfmld = pd.DataFrame.from_dict(dict(time=gtime, mld=gmld))
        #dfmld['mld'] = dfmld.mld.rolling(window=10, min_periods=5, center=True).mean()
        mldscatter = dfmld.hvplot.line(
            x='time',
            y='mld',
            color='white',
            alpha=0.5,
            #datashade=True,
        )
        plotslist.append(mldscatter)
    return reduce(lambda x, y: x*y, plotslist)



def get_xsection_raster(x_range):
    #import pdb; pdb.set_trace();
    #print('here things go wrong:',x_range)
    #variable = glider_explorer
    (x0, x1) = x_range
    global x_min_global
    global x_max_global
    x_min_global = x0
    x_max_global = x1
    meta, plt_props = load_viewport_datasets(x_range)
    plotslist1 = []
    #data=dsdict[dsid] if plt_props['zoomed_out'] else dsdict[dsid.replace('nrt', 'delayed')]
    metakeys = [element if plt_props['zoomed_out'] else element.replace('nrt', 'delayed') for element in meta.index]
    #import pdb; pdb.set_trace();
    varlist = [dsdict[dsid] for dsid in metakeys]
    dsconc = xarray.concat(varlist, dim='time')
    #mld = gt.physics.mixed_layer_depth(
    #    dsconc, 'temperature', thresh=0.3, verbose=False, ref_depth=5)
    #times = gt.utils.group_by_profiles(ds).mean().time.values
    #dfmld.hvplot.line(x='time', y='mld', color='white').opts(default_tools=[])
    dsconc['cplotvar'] = dsconc[glider_explorer.pick_variable]
    #dsconc = dsconc.isel(time=slice(
    #    0,-1,plt_props['subsample_freq']), drop=True)#data.to_dask_dataframe().sample(0.1)
    mplt = create_single_ds_plot_raster(data=dsconc)
    return mplt#*mldscatter

# on initial load, show all data
x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
         metadata['time_coverage_end (UTC)'].max().to_datetime64())

range_stream = RangeX(x_range=x_range)

""" cnorm_widget = pn.widgets.Select(
    name="icnorm",
    value="linear", options=['linear', 'eq_hist'])
variable_widget = pn.widgets.Select(
    name="variable",
    value="temperature", options=[
        'temperature', 'salinity', 'potential_density',
        'chlorophyll','oxygen_concentration'])
basin_widget = pn.widgets.Select(
    name="basin",
    value="Bornholm Basin", options=[
        'Bornholm Basin', 'Eastern Gotland']) """

global x_min_global
global x_max_global
x_min_global, x_max_global = x_range
#x_min_global = metadata['time_coverage_start (UTC)'].min().to_datetime64()
#x_max_global = metadata['time_coverage_end (UTC)'].max().to_datetime64()

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

    #x_range=(x_min_global,
    #         x_max_global)

    #@param.depends('pick_cnorm','pick_variable', 'pick_basin', 'pick_aggregation', 'pick_mld') # outcommenting this means just depend on all, redraw always
    def create_dynmap(self):

        #x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
        #        metadata['time_coverage_end (UTC)'].max().to_datetime64())
        #global x_min_global
        #global x_max_global
        #import pdb; pdb.set_trace();
        #global x_min_global
        #global x_max_global
        x_range=(x_min_global,
                 x_max_global)
        # print('dynmap VALUES!!!!!!',
        #    x_min_global,
        #    x_max_global)
        range_stream = RangeX(x_range=x_range)



        pick_cnorm='linear'
        print('execute dynmap',range_stream)
        # here everything is alright!


        #global meta
        #meta = hv.DynamicMap(load_viewport_datasets,streams=[range_stream])
        #dmap = hv.DynamicMap(get_xsection, streams=[range_stream])
        dmap_raster = hv.DynamicMap(
            get_xsection_raster,#(self.pick_variable),
            streams=[range_stream],
            #self.pick_variable,
            )
        #import pdb; pdb.set_trace();
        #range_stream = hv.streams.RangeX(source=dmap_raster)
        # Bis hierher habe ich einen stream.
        # das problem ist das jede bind auswahl meinen
        # stream zerstört, folgende daten zwar dynamic maps sind
        # aber leider ohne stream?

        #return dmap_raster#.opts(xlim=(x_min_global, x_max_global))
        if self.pick_mld:
            dmap_mld = hv.DynamicMap(get_xsection_mld, streams=[range_stream])
        #dmap_mld_rasterized = spread(tf.shade(dmap_mld),px=5)
        if self.pick_aggregation=='mean':
            means = dsh.mean('cplotvar')
        if self.pick_aggregation=='std':
            means = dsh.std('cplotvar')
        if self.pick_aggregation=='var':
            means = dsh.var('cplotvar')
        dmap_rasterized = rasterize(dmap_raster,
                    aggregator=means,
                    x_sampling=8.64e13/24,
                    y_sampling=0.2,
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
            cnorm=self.pick_cnorm,#cnorm_value,
            active_tools=['xpan', 'xwheel_zoom'],
            bgcolor="dimgrey",
            clabel=self.pick_variable)

        dmap = hv.DynamicMap(get_xsection)


        #x_min_global = x0
        #x_max_global = x1
        #global x_range
        #x_range = range_stream.x_range
        #(x0, x1) = #x_range
        #import pdb; pdb.set_trace()
        if self.pick_mld:
            return (dmap_rasterized).opts(xlim=(x_min_global, x_max_global))*dmap*dmap_mld
        else:
            return (dmap_rasterized).opts(xlim=(x_min_global, x_max_global))*dmap
        #return dmap*dmap_mld


#dmap_rasterized_bound = pn.bind(
#    create_dynmap,
#    cnorm_widget,
#    variable_widget)
glider_explorer=GliderExplorer()


pn.Row(glider_explorer.param, glider_explorer.create_dynmap).show(
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
* good example to follow is the AdvancedStockExplorer class in the documentation
...
"""
