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

###### filter metadata to prepare download ##############
metadata = utils.filter_metadata()
all_dataset_ids = utils.add_delayed_dataset_ids(metadata) # hacky
#import pdb; pdb.set_trace();

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
    # plot_options should be a dictionary
    #import pdb; pdb.set_trace()
    text_annotation = hv.Text(
        x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
        y=-2, text=dsid.replace('nrt_', ''),
        fontsize=plt_props['dynfontsize'],
            ).opts(text_opts
            ).opts(**ropts)

    startvline = hv.VLine(metadata.loc[dsid]['time_coverage_start (UTC)']).opts(color='grey')
    endvline = hv.VLine(metadata.loc[dsid]['time_coverage_end (UTC)']).opts(color='grey')
    print(plt_props)


    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c=variable,
        #x_sampling=plt_props['x_sampling'],
        #y_sampling=plt_props['y_sampling'],
        flip_yaxis=True,
        dynamic=False,
        cmap=cmocean.cm.thermal,
        height=400,
        responsive=True,
        #datashade=True,
        rasterize=True,
        cnorm=cnorm,

        #clim=clim,
        )

    """
    means = dsh.mean(variable)
    points = hv.Points(data, ['time', 'depth'])
    raster = rasterize(points,
                  aggregator=means,
                  cmap=cmocean.cm.thermal,
                  #interpolation='bilinear',
                  #dynamic=False,
                  datashade=True,
                  #x_sampling=1,#4e12,
                  #y_sampling=0.2,
                  cnorm='eq_hist').opts(invert_yaxis=True)#.opts(**ropts)

    return points
    """
    return text_annotation*startvline*endvline#*raster

def create_single_ds_plot_raster(data, metadata, variable, plt_props):
    # plot_options should be a dictionary
    #import pdb; pdb.set_trace()
    """
    text_annotation = hv.Text(
        x=metadata.loc[dsid]['time_coverage_start (UTC)'] ,
        y=-2, text=dsid.replace('nrt_', ''),
        fontsize=plt_props['dynfontsize'],
            ).opts(text_opts
            ).opts(**ropts)

    startvline = hv.VLine(metadata.loc[dsid]['time_coverage_start (UTC)']).opts(color='grey')
    endvline = hv.VLine(metadata.loc[dsid]['time_coverage_end (UTC)']).opts(color='grey')
    print(plt_props)

    """
    raster = data.hvplot.scatter(
        x='time',
        y='depth',
        c=variable,
        #x_sampling=plt_props['x_sampling'],
        #y_sampling=plt_props['y_sampling'],
        #flip_yaxis=True,
        #dynamic=False,
        cmap=cmocean.cm.thermal,
        #height=400,
        #responsive=True,
        #datashade=True,
        #rasterize=True,
        #cnorm=cnorm,*rasterize(

        #clim=clim,
        )

    """
    means = dsh.mean(variable)
    points = hv.Points(data, ['time', 'depth'])
    raster = rasterize(points,
                  aggregator=means,
                  cmap=cmocean.cm.thermal,
                  #interpolation='bilinear',
                  #dynamic=False,
                  datashade=True,
                  #x_sampling=1,#4e12,
                  #y_sampling=0.2,
                  cnorm='eq_hist').opts(invert_yaxis=True)#.opts(**ropts)

    return points
    """
    return raster


def get_xsection(x_range):
    variable='temperature'
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    #import pdb; pdb.set_trace()
    #nrtkeys = [item for item in dsdict.keys() if item[0:3]=='nrt']
    #meta = metadata.loc[nrtkeys]
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
    zoomed_out = False
    # grid timeline into n sections
    #plt_props['x_sampling'] = int(dtns/1000) # effective horizontal resolution (px)
    #plt_props['y_sampling']=.2
    #plt_props['dynfontsize']=4
    #plt_props['x_sampling']=8.64e13/240

    zoomed_in = False
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
        #data = data.drop
        #import pdb; pdb.set_trace()
        single_plot = create_single_ds_plot(data, metadata, variable, dsid, plt_props)
        plotslist.append(single_plot)
    return reduce(lambda x, y: x*y, plotslist)



def get_xsection_raster(x_range, variable):
    global plotvar
    plotvar = variable
    (x0, x1) = x_range
    dt = x1-x0
    dtns = dt/np.timedelta64(1, 'ns')
    #import pdb; pdb.set_trace()
    #nrtkeys = [item for item in dsdict.keys() if item[0:3]=='nrt']
    #meta = metadata.loc[nrtkeys]
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
    zoomed_out = False
    # grid timeline into n sections
    #plt_props['x_sampling'] = int(dtns/1000) # effective horizontal resolution (px)
    #plt_props['y_sampling']=.2
    #plt_props['dynfontsize']=4
    #plt_props['x_sampling']=8.64e13/240

    zoomed_in = False
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
    varlist = [dsdict[dsid.replace('nrt', 'delayed')][['time', 'depth', 'temperature']] for dsid in meta.index]
    dsconc = xarray.concat(varlist, dim='time')
    mplt = create_single_ds_plot_raster(dsconc, metadata, variable, plt_props)
    # note: only the first plot in the list needs the **ropts. Everything else migh
    #for dsid in meta.index:
    #    import pdb; pdb.set_trace();
    #    data=dsdict[dsid] if zoomed_out else dsdict[dsid.replace('nrt', 'delayed')]
    #    data = data.isel(time=slice(0,-1,10), drop=True)#data.to_dask_dataframe().sample(0.1)
        #data = data.drop
        #import pdb; pdb.set_trace()
        #single_plot = create_single_ds_plot_raster(data, metadata, variable, dsid, plt_props)
        #plotslist.append(single_plot)
    return mplt#reduce(lambda x, y: x*y, plotslist)






# on initial load, show all data
x_range=(metadata['time_coverage_start (UTC)'].min().to_datetime64(),
         metadata['time_coverage_end (UTC)'].max().to_datetime64())
range_stream = RangeX(x_range=x_range)

variable_widget = pn.widgets.Select(
    name="icnorm",
    value="linear", options=['linear', 'eq_hist'])

def create_dynmap(icnorm):
    # Change of global variables because range_stream does
    # not seem to work with passing variables (how?, why?)
    global cnorm
    #global variable
    cnorm = variable_widget.value
    dmap = hv.DynamicMap(get_xsection,
    #kdims=['variable'],#, 'cnorm'],
    streams=[range_stream],)
    dmap_raster = hv.DynamicMap(get_xsection_raster,
    kdims=['variable'],#, 'cnorm'],
    streams=[range_stream],)
    #import pdb; pdb.set_trace()

    return dmap, dmap_raster#.redim.values(
        #variable=('temperature', 'salinity'))

# one solution to the reset problem could be to change global values, e.g. set variable to "selected temperature" globally
dmap = pn.bind(
        create_dynmap,
        icnorm=variable_widget)

#import pdb; pdb.set_trace()
#pn.Column(plotslist2*create_dynmap('linear')).show(port=12345)
dmap, dmap_raster = create_dynmap('linear')
dmap_raster = dmap_raster.redim.values(
        variable=('temperature', 'salinity'))
means = dsh.mean(dmap_raster.current_key)
import pdb; pdb.set_trace()
#pn.Column((rasterize(dmap_raster, aggregator=means)*dmap).opts(**ropts)).show(port=12345)
pn.Column(rasterize(dmap_raster,
                    aggregator=means,
                    ).opts(
             invert_yaxis=True,
             colorbar=True,
             cmap=cmocean.cm.thermal,
             toolbar='above', tools=['xwheel_zoom', 'reset', 'xpan', 'ywheel_zoom', 'ypan'],
             default_tools=[],
             responsive=True,
             height=400,
             active_tools=['xpan', 'xwheel_zoom'],
             bgcolor="dimgrey",).opts(**ropts)*dmap).show(port=12345)
#pn.Column(variable_widget,dmap).show(port=12345)

"""
Future development ideas:
* activate hover
* holoviews autoupdate for development
* write tests including timings benchmark for development
* implement async functionen documented in holoviews to not disturb user interaction
* throw out X_range_stream (possibly) and implement full data dynamic sampling instead. One solution could be to use a dynamic .sample(frac=zoomstufe)
...
"""
