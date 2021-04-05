import numpy as np 
import pandas as pd 
import bokeh 
from bokeh.plotting import curdoc, figure, output_file, show, save
from bokeh.models import ColumnDataSource, ColorMapper, ColorBar, Ticker, Whisker
from bokeh.transform import linear_cmap, factor_cmap, log_cmap
from bokeh.models.widgets import Tabs, Panel, CheckboxGroup
from bokeh.models.tickers import FixedTicker
from bokeh.models import CustomJS, RadioGroup, CheckboxGroup
from bokeh.layouts import row, column
from bokeh.models.tools import HoverTool, TapTool
from bokeh.models.ranges import Range1d

def phase_fold_lc(period, t0, times, fluxes):
    t0 = 0 if np.isnan(t0) else t0
    phases = np.mod((times-t0)/period, 1.0)
    phases[phases > 0.5] -= 1

    return phases


def plot_lc(attr, old, new):
    # The index of the selected glyph is : new['1d']['indices'][0]
    lc_ind = new[0]
    tic = results['TIC'][lc_ind]
    lc = np.loadtxt('data/lcs_ascii/tic'+f'{tic:010}'+'.norm.lc')
    try:
        lc_2g = np.loadtxt('data/lcs_2g/tic'+f'{tic:010}'+'.2g.lc')
    except:
        lc_2g = np.array([np.linspace(-0.5,0.5,2000), np.ones(2000)]).T
    try:
        lc_pf = np.loadtxt('data/lcs_pf/tic'+f'{tic:010}'+'.pf.lc')
    except:
        lc_pf = np.array([np.linspace(-0.5,0.5,2000), np.ones(2000)]).T
    
    # lc_plot.name = 'TIC'+str(tic)
    
    lc_plot.data_source.data = {
    'times': lc[:,0],
    'fluxes': lc[:,1],
    'phases_triage': phase_fold_lc(results['period_triage'][lc_ind], 
                            results['t0_triage'][lc_ind], 
                            lc[:,0], lc[:,1]),
    'phases_2g': phase_fold_lc(results['period_2g'][lc_ind], 
                            results['t0_triage'][lc_ind], 
                            lc[:,0], lc[:,1]),
    'phases_pf': phase_fold_lc(results['period_pf'][lc_ind], 
                            results['t0_triage'][lc_ind], 
                            lc[:,0], lc[:,1]),
    }

    lc_plot_2g.data_source.data = {
    'phases': lc_2g[:,0],
    'fluxes': lc_2g[:,1]
    }

    lc_plot_pf.data_source.data = {
    'phases': lc_pf[:,0],
    'fluxes': lc_pf[:,1]
    }

    # check whether to phase fold or not and whether to plot 2g and pf models or not
    if checkbox4.active == 0:
        lc_plot.glyph.x = 'times'
        p4.x_range = Range1d(start=min(lc[:,0]), end=min(lc[:,0]))
    elif checkbox4.active == 1:
        lc_plot.glyph.x = 'phases_triage'
        p4.x_range = Range1d(start=-0.5, end=0.5)
    elif checkbox4.active == 2:
        lc_plot.glyph.x = 'phases_2g'
        p4.x_range = Range1d(start=-0.5, end=0.5)
    else:
        lc_plot.glyph.x = 'phases_pf'
        p4.x_range = Range1d(start=-0.5, end=0.5)

    if checkbox5.active == [0]:
        lc_plot_2g.visible = True
        lc_plot_pf.visible = False
    
    if checkbox5.active == [1]:
        lc_plot_2g.visible = False
        lc_plot_pf.visible = True
    
    if checkbox5.active == [0,1]:
        lc_plot_2g.visible = True
        lc_plot_pf.visible = True

    if checkbox5.active == []:
        lc_plot_2g.visible = False
        lc_plot_pf.visible = False

def update_phase_fold(attr, old, new):
    if checkbox4.active == 0:
        lc_plot.glyph.x = 'times'
        times_lc = lc_plot.data_source.data['times']
        p4.x_range = Range1d(start=min(times_lc), end=max(times_lc))
    elif checkbox4.active == 1:
        lc_plot.glyph.x = 'phases_triage'
        p4.x_range = Range1d(start=-0.5, end=0.5)
    elif checkbox4.active == 2:
        lc_plot.glyph.x = 'phases_2g'
        p4.x_range = Range1d(start=-0.5, end=0.5)
    else:
        lc_plot.glyph.x = 'phases_pf'
        p4.x_range = Range1d(start=-0.5, end=0.5)



def plot_models(attr, old, new):
    if checkbox5.active == [0]:
        lc_plot_2g.visible = True
        lc_plot_pf.visible = False
    
    if checkbox5.active == [1]:
        lc_plot_2g.visible = False
        lc_plot_pf.visible = True
    
    if checkbox5.active == [0,1]:
        lc_plot_2g.visible = True
        lc_plot_pf.visible = True

    if checkbox5.active == []:
        lc_plot_2g.visible = False
        lc_plot_pf.visible = False

def plot_diffs(attr, old, new):
    if checkbox6.active == [0]:
        diffs_tr2g.visible = True
        diffs_trpf.visible = False
        diffs_2gpf.visible = False
    
    if checkbox6.active == [1]:
        diffs_tr2g.visible = False
        diffs_trpf.visible = True
        diffs_2gpf.visible = False
    
    if checkbox6.active == [2]:
        diffs_tr2g.visible = False
        diffs_trpf.visible = False
        diffs_2gpf.visible = True
    
    if checkbox6.active == [0,1]:
        diffs_tr2g.visible = True
        diffs_trpf.visible = True
        diffs_2gpf.visible = False

    if checkbox6.active == [0,2]:
        diffs_tr2g.visible = True
        diffs_trpf.visible = False
        diffs_2gpf.visible = True

    if checkbox6.active == [1,2]:
        diffs_tr2g.visible = False
        diffs_trpf.visible = True
        diffs_2gpf.visible = True

    if checkbox6.active == [0,1,2]:
        diffs_tr2g.visible = True
        diffs_trpf.visible = True
        diffs_2gpf.visible = True

    if checkbox6.active == []:
        diffs_tr2g.visible = False
        diffs_trpf.visible = False
        diffs_2gpf.visible = False

def plot_hist(attr, old, new):
    if checkbox7.active == [0]:
        hist_tr2g.visible = True
        hist_trpf.visible = False
        hist_2gpf.visible = False
    
    if checkbox7.active == [1]:
        hist_tr2g.visible = False
        hist_trpf.visible = True
        hist_2gpf.visible = False

    if checkbox7.active == [2]:
        hist_tr2g.visible = False
        hist_trpf.visible = False
        hist_2gpf.visible = True
    
    if checkbox7.active == [0,1]:
        hist_tr2g.visible = True
        hist_trpf.visible = True
        hist_2gpf.visible = False

    if checkbox7.active == [0,2]:
        hist_tr2g.visible = True
        hist_trpf.visible = False
        hist_2gpf.visible = True

    if checkbox7.active == [1,2]:
        hist_tr2g.visible = False
        hist_trpf.visible = True
        hist_2gpf.visible = True

    if checkbox7.active == [0,1,2]:
        hist_tr2g.visible = True
        hist_trpf.visible = True
        hist_2gpf.visible = True

    if checkbox7.active == []:
        hist_tr2g.visible = False
        hist_trpf.visible = False
        hist_2gpf.visible = False


def update_tab1(attr, old, new):
    if checkbox1.active == 0:
        new_data = {'x': results['tsne_x'], 
                    'y': results['tsne_y'], 
                    'color': results['period_triage']}
        cs_tab1.data_source.data = new_data
        colormapper1['transform'].palette = 'Viridis256'
        colormapper1['transform'].low = new_data['color'].min()
        colormapper1['transform'].high = new_data['color'].max()

    elif checkbox1.active == 1:
        new_data = {'x': results['tsne_x'], 
                    'y': results['tsne_y'], 
                    'color': results['period_2g']}
        cs_tab1.data_source.data = new_data
        colormapper1['transform'].palette = 'Plasma256'
        colormapper1['transform'].low = new_data['color'].min()
        colormapper1['transform'].high = new_data['color'].max()

    else:
        new_data = {'x': results['tsne_x'], 
                    'y': results['tsne_y'], 
                    'color': results['period_pf']}
        cs_tab1.data_source.data = new_data
        colormapper1['transform'].palette = 'Turbo256'
        colormapper1['transform'].low = new_data['color'].min()
        colormapper1['transform'].high = new_data['color'].max()

'''
PREPARE THE DATAFRAME WITH ALL THE RELEVANT INFORMATION FOR PLOTTING
- load two-Gaussian, polyfit and triage table results
- append polyfit and triage results to the 2g table
- append coordinates from tsne projection for plotting
- clean up: replace infs with nans
'''
# output_file('tess_periods.html')
# load all files
results = pd.read_csv("data/tess_refined_2g.csv") # this is just the 2g results
triage = pd.read_csv('data/triage_period_table.txt', sep=" ", header=None)
triage.columns = ["TIC", "period", "t0"]
tsne_map = np.load('data/proj_norm.npy')

df_pf = pd.read_csv('data/tess_refined_polyfits.csv')
df_pf = df_pf.replace([np.inf, -np.inf], np.nan)
df_pf = df_pf[df_pf['logprob_mean_pf'].notnull()]

results = pd.merge(results, df_pf, on=['#ind'], how='outer')
# prepare data with extra needed columns for plotting
# will also need to insert polyfits as well once I have the table
results = results.drop('#ind',axis=1)
results = results.rename(columns={'TIC_x':'TIC'})
results = results.drop('TIC_y',axis=1)


results.insert(1, "tsne_x", tsne_map[0], True) 
results.insert(2, "tsne_y", tsne_map[1], True) 
results.insert(4, "period_triage", triage['period'], True)
results.insert(5, "t0_triage", triage['t0'], True)
results = results.replace([np.inf, -np.inf], np.nan)
results['period_2g'][np.isnan(results['logprob_mean_2g'])] = np.nan*np.ones(len(results['period_2g'][np.isnan(results['logprob_mean_2g'])]))
results.insert(len(results.columns), 'diff_triage_2g', 100*np.abs(results['period_triage']-results['period_2g'])/results['period_triage'])
results.insert(len(results.columns), 'diff_triage_pf', 100*np.abs(results['period_triage']-results['period_pf'])/results['period_triage'])
results.insert(len(results.columns), 'diff_2g_pf', 100*np.abs(results['period_2g']-results['period_pf'])/results['period_2g'])
# results = results[results['period_reldiff'] <= 200] 
source = ColumnDataSource(results)

lc_source = {
    'times': [],
    'fluxes': [],
    'phases_triage': [],
    'phases_2g': [],
    'phases_pf': []
}

models_source_2g = {
    'phases': [],
    'fluxes': []
}

models_source_pf = {
    'phases': [],
    'fluxes': []
}

'''
PREPARE THE LAYOUT FOR ALL OF THE PLOTS
4x4 figures: top left: periods, top right: period diffs, bottom left: chi2, bottom right: light curves
'''

TOOLTIPS = [
    ("index", "$index"),
    ("TIC", "@TIC"),
    ("Period_triage", "@period_triage"),
    ("Period_2g", "@period_2g"),
    ("Period_pf", "@period_pf")
]

p1 = figure(plot_width=600, plot_height=400, min_border=15, 
            tools="tap,box_zoom,reset,save,wheel_zoom,hover",
            tooltips = TOOLTIPS)
p2 = figure(plot_width=600, plot_height=400, min_border=15, 
            tools="tap,box_zoom,reset,save,wheel_zoom,hover",
            tooltips = TOOLTIPS)
p3 = figure(plot_width=600, plot_height=400, min_border=15, 
            tools="tap,box_zoom,reset,save,wheel_zoom,hover",
            tooltips = TOOLTIPS)
p4 = figure(plot_width=600, plot_height=400, min_border=15, title='TIC', 
            tools="tap,box_zoom,reset,save,wheel_zoom,hover")

lc_plot = p4.scatter(x='times', y='fluxes', source=lc_source, size=1, name='')
lc_plot_2g = p4.line(x='phases', y='fluxes', source=models_source_2g, line_width=3, line_color='orange', name='')
lc_plot_pf = p4.line(x='phases', y='fluxes', source=models_source_pf, line_width=3, line_color='yellow', name='')
lc_plot_2g.visible = False 
lc_plot_pf.visible = False

p4.title.text = lc_plot.name

checkbox1 = RadioGroup(labels=["triage", "2g", "pf"], active=0)
# checkbox2 = RadioGroup(labels=["triage-2g"], active=0)
# checkbox3 = RadioGroup(labels=["logprob_mean 2g"], active=0)
checkbox4 = RadioGroup(labels=["2g model"], active=0)
checkbox4 = RadioGroup(labels=["none", "triage period", "2g period", "pf period"], active=0)
checkbox4.on_change('active', update_phase_fold)
checkbox5 = CheckboxGroup(labels=["2g model", "polyfit"], active=[])
checkbox5.on_change('active', plot_models)
checkbox6 = CheckboxGroup(labels=["triage vs 2g", 
                                    "triage vs pf", 
                                    "2g vs pf"], 
                                    active=[0,1,2])
checkbox6.on_change('active', plot_diffs)
checkbox7 = CheckboxGroup(labels=["triage vs 2g", 
                                    "triage vs pf", 
                                    "2g vs pf"], 
                                    active=[0,1,2])
checkbox7.on_change('active', plot_hist)

tab1 = Panel(child = row(column(p1, checkbox1), column(p4,row(checkbox4, checkbox5))), 
            title = 'Periods')
tab2 = Panel(child = row(column(p2,checkbox6), column(p3, checkbox7)), 
            title = 'Stats')
# tab3 = Panel(child = row(column(p3, checkbox3), column(p4,row(checkbox4, checkbox5))), 
#             title = 'Logprobs')
layout = Tabs(tabs = [tab1, tab2])


'''
PLOTTING TAB 1
'''
#all systems in triage
# p1.scatter(x=tsne_map[0], y=tsne_map[1], size=1, color='gray')

source_tab1 = ColumnDataSource({'x': results['tsne_x'], 
                                'y': results['tsne_y'], 
                                'color': results['period_triage'],
                                'period_triage': results['period_triage'],
                                'period_2g': results['period_2g'],
                                'period_pf': results['period_pf'],
                                'TIC': results['TIC']})

colormapper1 = log_cmap(field_name = "color", palette='Viridis256',
                        low = source_tab1.data['color'].min(), 
                        high=source_tab1.data['color'].max())

cs_tab1 = p1.scatter(x='x', y='y',
         color=colormapper1, source = source_tab1, size=5)
color_bar1 = ColorBar(color_mapper=colormapper1['transform'], 
                      width=500, height=10, location=(0,0),
                      orientation="horizontal")
                      #ticker = FixedTicker(ticks=[0, 0.1, 1, 10, 100]))
p1.add_layout(color_bar1, 'above')
checkbox1.on_change('active', update_tab1)
cs_tab1.data_source.selected.on_change('indices', plot_lc)



'''
PLOTTING TAB2
'''

source_tab2 = ColumnDataSource({'x': results['period_triage'], 
                                'y': results['period_2g'], 
                                'z': results['period_pf'],
                                'color': np.abs(results['logprob_mean_2g'])})

# colormapper2 = linear_cmap(field_name = "color", palette='Viridis256', low = 0, high=100)

diffs_tr2g = p2.scatter(x='x', y='y', source = source_tab2, legend_label='P_triage vs. P_2g', size=10)
diffs_trpf = p2.scatter(x='x', y='z', source = source_tab2, color='orange', legend_label='P_triage vs. P_pf', size=7.5)
diffs_2gpf = p2.scatter(x='y', y='z', source = source_tab2, color='purple', legend_label='P_2g vs. P_pf', size=5)
p2.xaxis.axis_label = 'P_x'
p2.yaxis.axis_label = 'P_y'
p2.line(x='x', y='x', source = source_tab2, line_color='red', line_width=1, line_dash='dotted', line_alpha=0.5)
p2.legend.location = "top_left"

# base_2g, lower_2g, upper_2g = [], [], []

# for i in range(len(results)):
#     try:
#         period_2g = results['period_2g'][i]
#         period_2g_low = results['period_2g_sigma_low']
#         period_2g_high = results['period_2g_sigma_high']
#         lower_2g.append(results['period_2g'][i] - results['period_2g_sigma_low'][i])
#         upper_2g.append(results['period_2g'][i] + results['period_2g_sigma_high'][i])
#         base_2g.append(results['period_2g'][i])
#     except:
#         pass

# source_error_2g = ColumnDataSource(data=dict(base=base_2g, lower=lower_2g, upper=upper_2g))

# p2.add_layout(
#     Whisker(source=source_error_2g, base="base", upper="upper", lower="lower")
# )

# color_bar2 = ColorBar(color_mapper=colormapper2['transform'], width=500, height=10, location=(0,0),
#                       title="chi2", orientation="horizontal")
# p2.add_layout(color_bar2, 'above')



# diff_triage_2g
# diff_triage_pf
# diff_2g_pf

twog_cut = results['diff_triage_2g'][results['diff_triage_2g'] < 1].values
pf_cut = results['diff_triage_pf'][results['diff_triage_pf'] < 1].values
twogpf_cut = results['diff_2g_pf'][results['diff_2g_pf'] < 1].values


hist_2g, edges_2g = np.histogram(twog_cut, density=True, bins=50)
hist_pf, edges_pf = np.histogram(pf_cut, density=True, bins=50)
hist_2gpf, edges_2gpf = np.histogram(twogpf_cut, density=True, bins=50)

hist_tr2g = p3.quad(top=hist_2g, bottom=0, left=edges_2g[:-1], right=edges_2g[1:],
           fill_color="blue", line_color="white", alpha=0.5, legend_label='triage - 2g')
hist_trpf = p3.quad(top=hist_pf, bottom=0, left=edges_pf[:-1], right=edges_pf[1:],
           fill_color="orange", line_color="white", alpha=0.5, legend_label='triage - pf')
hist_2gpf = p3.quad(top=hist_2gpf, bottom=0, left=edges_2gpf[:-1], right=edges_2gpf[1:],
           fill_color="purple", line_color="white", alpha=0.5, legend_label='2g - pf')
p3.xaxis.axis_label = "| triage - model | / triage x 100 %"


curdoc().add_root(layout)












