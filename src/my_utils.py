### My module


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



### plot related functions
def create_category_colormap(base_color, n_steps=256):
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        f"modulated_{base_color}",
        [mcolors.to_rgba(base_color, alpha=0.), mcolors.to_rgba(base_color, alpha=1.0)],
        N=n_steps,)

def create_mix_colormap(base_color_1, base_color_2, n_steps=256):
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        f"modulated_{base_color_1}",
        [mcolors.to_rgba(base_color_1, alpha=1), mcolors.to_rgba(base_color_2, alpha=1)],
        N=n_steps,)

def create_mix_colormap_N(list_of_colors, n_steps=256):
    import matplotlib.colors as mcolors
    temp = [mcolors.to_rgba(color, alpha=1) for color in list_of_colors]
    return mcolors.LinearSegmentedColormap.from_list(
        f"custom_mix_cm", temp, N=n_steps)

def generate_colorbar_svg(scores, cmap, output_path, title=None, background_color='white'):
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import MaxNLocator
    """
    Generate a colorbar as an SVG file from a set of scores.

    Parameters:
        scores (array-like): Array of scores to generate the colorbar.
        cmap (Colormap): Matplotlib colormap to use for the visualization or cmap object
        output_path (str): Path to save the SVG file.
        tick_format (str): Format string for tick labels.
    """
    # Normalize the scores
    norm = Normalize(vmin=np.min(scores), vmax=np.max(scores))

    # Create a figure and axis for the colormap
    fig, ax = plt.subplots(figsize=(1, 6))
    fig.subplots_adjust(right=.4,bottom=0.1)

    # Set background color
    fig.patch.set_facecolor(background_color)

    # Create a colorbar
    cbar = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    # Format the tick labels
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    fig.tight_layout()
    # Save the colormap as an SVG file
    plt.savefig(output_path+f'/{title}.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)


### manipulate strings (or arrays of strings)
def from_chname_to_subjects(ch_names_list_or_array):
    """
    ch_names_list_or_array : list or array of strings, each string being a channel name, composed as follows : <subject id>_<shaft><number>
    """
    subs = []
    n_by_sub = []
    
    for ch_name in ch_names_list_or_array:
        subj_id = ch_name.split('_')[0]
        last_id = subs[-1] if len(subs) > 0 else None
        if subj_id == last_id:
            n_by_sub[-1] += 1
        else:
            subs.append(subj_id)
            n_by_sub.append(1)
        if subj_id not in subs:
            subs.append(ch_name.split('_')[0])

    return subs, n_by_sub



##########################################
########## a bit more technical ##########
##########################################

# function to remove isolated channels
def searchlight(loc, name, distance_threshold = 5, n_sub=2, n_repeat = 1, metrics = 'euclidean', weight_axis = None):
    '''
    name: list of channel names (e.g. ['001-shaft_01', '001-shaft'_02', '002-shaft_02'])
    if minkowski, weight_axis is the weighting of different dimensions

    returns: list of booleans, True if the channel is to be kept, False if it is to be ignored, can be used as a mask
    '''
    from scipy.spatial.distance import cdist

    temp_loc, temp_name = loc.copy(), name.copy()
    
    channels_ignore = []  

    for i_repeat in range(n_repeat):
        dist_matrix = cdist(temp_loc,temp_loc, metrics, w = weight_axis)  # computes distance matrix between all channels

        for channel_index in range(dist_matrix.shape[0]):
            if not channel_index in channels_ignore:
                subject_channel = temp_name[channel_index].split('-')[0]
                subject_neighbour_list = []
                for neighbour_index in range(dist_matrix.shape[0]):
                    distance_neighbour = dist_matrix[channel_index,neighbour_index]
                    subject_neighbour = temp_name[neighbour_index].split('-')[0]
                    if subject_neighbour != subject_channel and not subject_neighbour in subject_neighbour_list and distance_neighbour < distance_threshold:
                        subject_neighbour_list.append(subject_neighbour)
                if len(subject_neighbour_list) < n_sub:
                    channels_ignore.append(channel_index)
            else:
                True
    channels_keep = [not (i in (channels_ignore)) for i in np.arange(loc.shape[0])]
    return channels_keep


def set_theme(theme = 'clement'):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    if theme == 'clement':
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["b", "r", "g", "orange", "cyan", "m", "olive", "brown", "darkblue", "coral", "deeppink"])
        mpl.rcParams['axes.grid'] = False
        mpl.rcParams['axes.linewidth'] = 1.8
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['boxplot.boxprops.linewidth'] = 1.5
        mpl.rcParams['boxplot.capprops.linewidth'] = 1.5
        mpl.rcParams['boxplot.flierprops.linewidth'] = 1.5
        mpl.rcParams['boxplot.flierprops.markeredgewidth'] = 1.5

        mpl.rcParams['font.size'] = 16
        # set the format of strings of numerals (like : custom_ticks_labels = ['0.5', '2', '4', '10', '35', '80', '150'])
        # mpl.rcParams['xtick.major.formatter'] = 'plain'
        # mpl.rcParams['ytick.major.formatter'] = 'plain'

        mpl.rcParams['grid.alpha'] = 0

        mpl.rcParams['grid.linewidth'] = 0.5

        mpl.rcParams['lines.linewidth'] = 2

        mpl.rcParams['figure.figsize'] = [5,5]

        mpl.rcParams['hist.bins'] = 30

        mpl.rcParams['legend.frameon'] = False

        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.format'] = 'svg'

        mpl.rcParams['xtick.major.width'] = 2
        mpl.rcParams['ytick.major.width'] = 2
    elif theme == 'pierre':
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["b", "r", "g", "orange", "cyan", "m", "olive", "brown", "darkblue", "coral", "deeppink"])
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['axes.linewidth'] = 1.8
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['boxplot.boxprops.linewidth'] = 1.5
        mpl.rcParams['boxplot.capprops.linewidth'] = 1.5
        mpl.rcParams['boxplot.flierprops.linewidth'] = 1.5
        mpl.rcParams['boxplot.flierprops.markeredgewidth'] = 1.5

        mpl.rcParams['font.size'] = 16

        mpl.rcParams['grid.alpha'] = 0.5

        mpl.rcParams['grid.linewidth'] = 0.5

        mpl.rcParams['lines.linewidth'] = 2

        mpl.rcParams['figure.figsize'] = [5,5]

        # mpl.rcParams['font.family'] = ['Calibri']
        mpl.rcParams['font.size'] = 14

        mpl.rcParams['hist.bins'] = 30

        mpl.rcParams['legend.frameon'] = False

        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.format'] = 'svg'

        mpl.rcParams['xtick.major.width'] = 2
        mpl.rcParams['ytick.major.width'] = 2

    else:
        print(theme, 'not found, set to default')