# import nibabel
import nibabel.freesurfer.io as fsio
import copy
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from os.path import join
import mne
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps as cmaps
import imageio
import os
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D



def mask_distance(positions, vertices, max_dist):
    #mask based on distance to cortical surface
    mask = []
    for index_position, position in enumerate(positions.T):
        dist = np.sqrt(np.sum((position-vertices)**2, axis=1))
        min_dist = dist[np.argmin(dist)]
        if min_dist <= max_dist:
            mask.append(index_position)
    return mask

def place_surface(positions, vertices, dist_proj):
    #stick channel coordinates to nearby surface
    for index_position, position in enumerate(positions.T):
        dist = np.sqrt(np.sum((position-vertices)**2, axis=1))
        idx = np.argmin(dist)
        if dist[idx] <= dist_proj:
            closest_position = vertices[idx,:]
            positions[:,index_position] = closest_position
    return positions

def color_array(scores, color_type = 'cool'):
    color_map = dict()
    for i, s in enumerate(scores):
        color_map[i] = cmaps[color_type](s/np.max(scores))
    return color_map

def add_3d_axes_to_scene(plotter, position='lower_left', scale=0.1, origin=None):
    """
    Ajoute des axes 3D qui tournent avec la scène.
    
    Parameters
    ----------
    plotter : pyvista.Plotter
        L'instance du plotter PyVista
    position : str
        Position relative dans la boîte englobante
    scale : float
        Échelle des axes (relative à la taille de la scène)
    origin : list or None
        Position personnalisée de l'origine. Si None, calculée automatiquement.
    """
    import pyvista as pv
    
    bounds = plotter.bounds
    
    if origin is None:
        # Calculer la position selon le paramètre position
        position_map = {
            'lower_left': (0.1, 0.1, 0.1),
            'lower_right': (0.9, 0.1, 0.1),
            'upper_left': (0.1, 0.9, 0.1),
            'upper_right': (0.9, 0.9, 0.1),
            'center': (0.5, 0.5, 0.5)
        }
        
        ratios = position_map.get(position, (0.1, 0.1, 0.1))
        
        origin = [
            bounds[0] + (bounds[1] - bounds[0]) * ratios[0],
            bounds[2] + (bounds[3] - bounds[2]) * ratios[1],
            bounds[4] + (bounds[5] - bounds[4]) * ratios[2]
        ]
    # Calculer la longueur des axes
    scene_size = np.sqrt(
        (bounds[1] - bounds[0])**2 + 
        (bounds[3] - bounds[2])**2 + 
        (bounds[5] - bounds[4])**2
    )
    axes_length = scene_size * scale
    
    # Définir les axes
    axes_data = [
        ([1, 0, 0], 'black', 'X'),
        ([0, 1, 0], 'black', 'Y'),
        ([0, 0, 1], 'black', 'Z')
    ]
    
    for direction, color, label in axes_data:
        # Créer une flèche
        arrow = pv.Arrow(
            start=origin,
            direction=direction,
            scale=axes_length,
            tip_length=0.15,
            tip_radius=0.025 ,
            shaft_radius=0.007


        )
        plotter.add_mesh(arrow, color=color)
        
        # Ajouter le label
        label_pos = np.array(origin) + np.array(direction) * axes_length * 1.2
        plotter.add_point_labels(
            [label_pos],
            [label],
            font_size=13,
            text_color=color,
            bold=False,
            always_visible=False
        )
    
    return plotter    

def brainscatter_static(scores, locations, channels,
                        background,
                        min_scores = -1, dist_max = 5, dist_proj = 0,
                        cmap = 'rainbow',
                        left_viewpoint = False, right_viewpoint = False, upper_viewpoint = False, hems = ['lh','rh'],
                        surf_path = '/home/sauvage/Sorciere_v0/DataSEEG_Sorciere/BIDS/freesurfer', path_save = '/home/sauvage/Documents/Code/Wildly_Convoluted_Brains/figures/broadband', 
                        close = True,
                        savefig = False,
                        fig_label = None,
                        add_axes = False,
                        axes_position = 'lower_left',
                        axes_scale = 0.1):
    
    if not left_viewpoint:
        left_viewpoint = [(-280, 240, 20),
                    (-35, -30, 0),
                    (0.5, 1, 2)]
    if not right_viewpoint:
        right_viewpoint = [(280, 240, 20),
                        (35, -30, 0),
                        (-0.5, 1, 2)]
    if not upper_viewpoint:
        upper_viewpoint = [(0, 0, 360),
                        (0, 0, 0),
                        (0, 1, 2)]
    
    scores[np.where(scores<min_scores)] = min_scores
    clim = [scores[scores > min_scores].min(), scores[scores > min_scores].max()]
    locs = locations.copy()
    s = scores.copy()

    #retrieve coordinates of surface vertices
    vertices = [None, None]
    vertices[0], _ = fsio.read_geometry(join(surf_path + '/fsaverage/surf/','lh.pial'))
    vertices[1], _ = fsio.read_geometry(join(surf_path + '/fsaverage/surf/','rh.pial'))
    vertices = np.concatenate(vertices, 0)

    # mask on maximum distance
    mask = mask_distance(locs, vertices, dist_max)
    s = s[mask]
    chans = channels[mask]
    locs = locs[:,mask]

    # projection on surface
    locs = place_surface(locs, vertices, dist_proj)

    #mask on treshold
    locs_empty = locs[:,s <= min_scores] #points without scalar
    locs = locs[:,s > min_scores] #points with scalar
    s = s[s > min_scores]


    for hem in hems:
        if hem == 'lh':
            hem_locs_empty = locs_empty[:,locs_empty[0,:]<0]
            index_interest = list(np.arange(locs.shape[1])[locs[0,:] < 0])

            #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #                                                        channel_select = ["'"], strict = True, exclude = False)
        elif hem == 'rh':
            hem_locs_empty = locs_empty[:,locs_empty[0,:]>0]
            #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #                                                        channel_select = ["'"], strict = True, exclude = True)
            index_interest = list(np.arange(locs.shape[1])[locs[0,:] > 0])

        elif hem == 'both':
            hem_locs_empty = locs_empty
            index_interest = list(np.arange(locs.shape[1]))

        hem_s = s[np.newaxis,index_interest]
        hem_chan = chans[index_interest]
        hem_locs = locs[:,index_interest]

        #plot brain
        if dist_proj == 0:
            alpha = 0.3
            proj = 'trans'
        else:
            alpha = .3
            proj = 'proj'

        brain = mne.viz.Brain(subjects_dir=surf_path, subject = 'fsaverage',
                            surf='pial', #'pial', 'inflated', 'white',...
                            hemi=hem,  # hem,
                            background=background,
                            alpha = alpha,
                            cortex='high_contrast',  # ['w','k'], #classic
                            offscreen=True,
                            theme=None)
        if fig_label is not None:
            brain.add_text(0.1, 0.9, fig_label, 'title', font_size=16)

        # labels = ("ctx-lh-inferiortemporal", "ctx-lh-middletemporal", "ctx-lh-superiortemporal", "ctx-lh-temporalpole", "ctx-lh-insula", "ctx-lh-transversetemporal")
        # brain.add_volume_labels(aseg="aparc+aseg", labels=labels, alpha = 0.25)
        if hem_s.shape[0] > 0:
            # brain.plotter.add_points(hem_locs.T, render_points_as_spheres=True, point_size = 10, scalars = hem_s[0], cmap = cmap)  #lighting = False
            brain.plotter.add_points(hem_locs.T, render_points_as_spheres=True, point_size = 15, scalars = hem_s[0], cmap = cmap)  #lighting = False
            brain.plotter.update_scalar_bar_range(clim)

        if hem_locs_empty.shape[1] > 0:
            brain.plotter.add_points(hem_locs_empty.T, render_points_as_spheres=True, point_size = 3, color = 'black')

        if hem == 'lh':
            brain.plotter.camera_position = left_viewpoint

        elif hem == 'rh':
            brain.plotter.camera_position = right_viewpoint

        elif hem == 'both':
            brain.plotter.camera_position = upper_viewpoint

        if savefig == True:
            brain.save_image(path_save + hem + '.png')
        if close:
            brain.plotter.close()

    left_str = 'lh'
    right_str =  'rh'

    # left_h = mpimg.imread(path_save + left_str + '.png')
    # right_h = mpimg.imread(path_save + right_str + '.png')

    # Créer une figure et des sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(1, 1))

    # # Afficher la première image dans le premier sous-graphique
    # axes[0].imshow(right_h)
    # axes[0].set_title(right_str)

    # # Afficher la deuxième image dans le deuxième sous-graphique
    # axes[1].imshow(left_h)
    # axes[1].set_title(left_str)

    # After creating plotter but before showing:
    if add_axes:
        add_3d_axes_to_scene(brain.plotter, axes_position, axes_scale)
    

    fig.show()
    #plt.title(out_names[index])
    if savefig == True:
        fig.savefig(path_save + '.png')

    return fig,axes

def brainscatter_static_(scores, locations, channels,
                        background,
                        cmap,
                        min_scores = -1, dist_max = 5, dist_proj = 0,
                        left_viewpoint = False, right_viewpoint = False, hems = 'both',  # ['lh','rh'],
                        surf_path = '/home/sauvage/Sorciere_v0/DataSEEG_Sorciere/BIDS/freesurfer', path_save = '/home/sauvage/Documents/Code/Wildly_Convoluted_Brains/figures/broadband/brain_plots', 
                        close = True):
    
    if not left_viewpoint:
        left_viewpoint = [(-280, 240, 20),
                    (-35, -30, 0),
                    (0.5, 1, 2)]
    if not right_viewpoint:
        right_viewpoint = [(280, 240, 20),
                        (35, -30, 0),
                        (-0.5, 1, 2)]
    
    scores[np.where(scores<min_scores)] = min_scores
    clim = [scores[scores > min_scores].min(), scores[scores > min_scores].max()]
    locs = locations.copy()
    s = scores.copy()

    #retrieve coordinates of surface vertices
    vertices = [None, None]
    vertices[0], _ = fsio.read_geometry(join(surf_path + '/fsaverage/surf/','lh.pial'))
    vertices[1], _ = fsio.read_geometry(join(surf_path + '/fsaverage/surf/','rh.pial'))
    vertices = np.concatenate(vertices, 0)

    # mask on maximum distance
    mask = mask_distance(locs, vertices, dist_max)
    s = s[mask]
    chans = channels[mask]
    locs = locs[:,mask]

    # projection on surface
    locs = place_surface(locs, vertices, dist_proj)

    #mask on treshold
    locs_empty = locs[:,s <= min_scores] #points without scalar
    locs = locs[:,s > min_scores] #points with scalar
    s = s[s > min_scores]



    for hem in hems:
        if hem == 'lh':
            hem_locs_empty = locs_empty[:,locs_empty[0,:]<0]
            index_interest = list(np.arange(locs.shape[1])[locs[0,:] < 0])
            hem_s = s[np.newaxis,index_interest]
            hem_chan = chans[index_interest]
            hem_locs = locs[:,index_interest]
            #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #                                                        channel_select = ["'"], strict = True, exclude = False)
        elif hem == 'rh':
            hem_locs_empty = locs_empty[:,locs_empty[0,:]>0]
            #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #                                                        channel_select = ["'"], strict = True, exclude = True)
            index_interest = list(np.arange(locs.shape[1])[locs[0,:] > 0])
            hem_s = s[np.newaxis,index_interest]
            hem_chan = chans[index_interest]
            hem_locs = locs[:,index_interest]

        #plot brain
        if dist_proj == 0:
            alpha = 0.3
            proj = 'trans'
        else:
            alpha = .3
            proj = 'proj'

    brain = mne.viz.Brain(subjects_dir=surf_path, subject = 'fsaverage',
                            surf='pial', #'pial', 'inflated', 'white',...
                            hemi=hem,  # hem,
                            background=background,
                            alpha = alpha,
                            cortex='high_contrast',  # ['w','k'], #classic
                            offscreen=True,
                            theme=None)

        # # labels = ("ctx-lh-inferiortemporal", "ctx-lh-middletemporal", "ctx-lh-superiortemporal", "ctx-lh-temporalpole", "ctx-lh-insula", "ctx-lh-transversetemporal")
        # # brain.add_volume_labels(aseg="aparc+aseg", labels=labels, alpha = 0.25)
        # if hem_s.shape[0] > 0:
        #     brain.plotter.add_points(hem_locs.T, render_points_as_spheres=True, point_size = 10, scalars = hem_s[0], cmap = cmap)  #lighting = False
        #     brain.plotter.update_scalar_bar_range(clim)

        # if hem_locs_empty.shape[1] > 0:
        #     brain.plotter.add_points(hem_locs_empty.T, render_points_as_spheres=True, point_size = 3, color = 'black')

    if hem == 'lh':
        brain.plotter.camera_position = left_viewpoint

    elif hem == 'rh':
        brain.plotter.camera_position = right_viewpoint

    brain.plotter.add_points(locs.T, render_points_as_spheres=True, point_size = 10, scalars = s, cmap = cmap)
    # brain.plotter.add_points(locs.T, render_points_as_spheres=True, point_size = 13, scalars = s, color = 'violet')
    brain.plotter.update_scalar_bar_range(clim)
    # brain.save_image(path_save + 'bothhems.png')
    if close:
        brain.plotter.close()
    # fig, axes = plt.subplots(1,1,figsize=(20,10))
    # both_h = mpimg.imread(path_save + hem + '.png')
    # axes.imshow(both_h)



        # brain.save_image(path_save + hem + '.png')
        # if close:
        #     brain.plotter.close()

    # left_str = 'lh'
    # right_str =  'rh'

    # left_h = mpimg.imread(path_save + left_str + '.png')
    # right_h = mpimg.imread(path_save + right_str + '.png')

    # # Créer une figure et des sous-graphiques
    # fig, axes = plt.subplots(1, 2, figsize=(20, 15))

    # # Afficher la première image dans le premier sous-graphique
    # axes[0].imshow(right_h)
    # axes[0].set_title(right_str)

    # # Afficher la deuxième image dans le deuxième sous-graphique
    # axes[1].imshow(left_h)
    # axes[1].set_title(left_str)

    fig.show()
    #plt.title(out_names[index])
    # fig.savefig(path_save + '.png')

    return fig,axes

def create_matshow_gif(data, timesteps, output_file='matshow_animation.gif', 
                       labels = None, regressor_name = '',
                       cmap = 'viridis', duration = 1, loop = 5):
    """
    Create a GIF from a matshow changing over the specified number of timesteps.

    Parameters:
    data (ndarray): A 3D numpy array of shape (timesteps, rows, columns).
    timesteps (int): The number of timesteps to animate.
    output_file (str): The name of the output GIF file.
    """
    # Ensure the output directory exists
    os.makedirs('frames', exist_ok=True)
    
    if labels is None:
        labels = np.arange(data.shape[1])

    # Generate and save each frame
    for t in range(timesteps):
        plt.matshow(data[t], cmap='viridis')
        plt.xticks(np.arange(len(labels)),labels,rotation = 90)
        plt.yticks(np.arange(len(labels)),labels)
        plt.title(regressor_name + ' ' + f'Timestep {t + 1}')
        plt.colorbar()
        plt.savefig(f'frames/frame_{t:03d}.png')
        plt.close()
    
    # Read the saved frames and compile them into a GIF
    with imageio.get_writer(output_file, mode='I', duration=duration,loop=loop) as writer:
        for t in range(timesteps):
            image = imageio.imread(f'frames/frame_{t:03d}.png')
            writer.append_data(image)
    
    # Clean up the frames directory
    for t in range(timesteps):
        os.remove(f'frames/frame_{t:03d}.png')
    os.rmdir('frames')
    
    print(f'GIF saved as {output_file}')



def create_collection_gif(data, output_file='matshow_animation.gif', 
                    duration = 1, loop = 5, write_image = False):
    """
    Create a GIF from a collections of prebuilt images.

    Parameters:
    data (list): list of figures, they will be the frames of the gif
    timesteps (int): The number of timesteps to animate.
    output_file (str): The name of the output GIF file.
    """
    # Ensure the output directory exists
    os.makedirs('frames', exist_ok=True)
    

    # Generate and save each frame
    for t,fig in enumerate(data):
        if write_image:
            fig.write_image(file=f'frames/frame_{t:03d}.png', format='.png')
        else:
            fig.savefig(f'frames/frame_{t:03d}.png', dpi=300)
        

    
    # Read the saved frames and compile them into a GIF
    with imageio.get_writer(output_file, mode='I', duration=duration,loop=loop) as writer:
        for t in range(len(data)):
            image = imageio.imread(f'frames/frame_{t:03d}.png')
            writer.append_data(image)
    
    # Clean up the frames directory
    for t in range(len(data)):
        os.remove(f'frames/frame_{t:03d}.png')
    os.rmdir('frames')
    
    print(f'GIF saved as {output_file}')


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


def brainscatter_rotation(scores, locations, channels, surf_path, path_save, title_tag = '',
                        min_scores = -1, dist_max = 5, dist_proj = 0,
                        cmap = 'rainbow', background = 'black',
                        left_viewpoint = False, right_viewpoint = False, #hems = ['lh','rh'],
                        close = True, n_angles = 10):
    
    if not left_viewpoint:
        left_viewpoint = [(-400, 400, 20),
                    (-15, 15, 0),
                    (0, 0, 5)]


    if not right_viewpoint:
        right_viewpoint = [(400, 400, 20),
                    (15, 15, 0),
                    (0, 0, 5)]
        
    modulus = np.sqrt(left_viewpoint[0][0]**2 + left_viewpoint[0][1]**2)
    left_viewpoint_list = [left_viewpoint]
    right_viewpoint_list = [right_viewpoint]
    for i in range(n_angles):
        new_left_viewpoint = [(left_viewpoint_list[0][0][0] * np.cos(2*np.pi/n_angles*i), left_viewpoint_list[0][0][1] * -np.sin(2*np.pi/n_angles*i), left_viewpoint_list[0][0][2]), 
                        (left_viewpoint_list[0][1][0] * np.cos(2*np.pi/n_angles*i), left_viewpoint_list[0][1][1] * -np.sin(2*np.pi/n_angles*i), left_viewpoint_list[0][1][2]), 
                        left_viewpoint_list[0][2]]
        left_viewpoint_list.append(new_left_viewpoint)

        new_right_viewpoint = [(right_viewpoint_list[0][0][0] * np.cos(2*np.pi/n_angles*i), right_viewpoint_list[0][0][1] * -np.sin(2*np.pi/n_angles*i), right_viewpoint_list[0][0][2]), 
                        (right_viewpoint_list[0][1][0] * np.cos(2*np.pi/n_angles*i), right_viewpoint_list[0][1][1] * -np.sin(2*np.pi/n_angles*i), right_viewpoint_list[0][1][2]), 
                        right_viewpoint_list[0][2]]
        right_viewpoint_list.append(new_right_viewpoint)
    
    scores[np.where(scores<min_scores)] = min_scores
    clim = [scores[scores > min_scores].min(), scores[scores > min_scores].max()]
    locs = locations.copy()
    s = scores.copy()

    #retrieve coordinates of surface vertices
    vertices = [None, None]
    vertices[0], _ = fsio.read_geometry(join(surf_path + '/fsaverage/surf/','lh.pial'))
    vertices[1], _ = fsio.read_geometry(join(surf_path + '/fsaverage/surf/','rh.pial'))
    vertices = np.concatenate(vertices, 0)

    # mask on maximum distance
    mask = mask_distance(locs, vertices, dist_max)
    s = s[mask]
    chans = channels[mask]
    locs = locs[:,mask]

    # projection on surface
    locs = place_surface(locs, vertices, dist_proj)

    #mask on treshold
    locs_empty = locs[:,s <= min_scores] #points without scalar
    locs = locs[:,s > min_scores] #points with scalar
    s = s[s > min_scores]

    img_list = []
    for view_index, left_viewpoint, right_viewpoint in zip(np.arange(len(left_viewpoint_list) - 1),left_viewpoint_list[1:], right_viewpoint_list[1:]):
        # for hem in hems:
            # if hem == 'lh':
            #     hem_locs_empty = locs_empty[:,locs_empty[0,:]<0]
            #     index_interest = list(np.arange(locs.shape[1])[locs[0,:] < 0])
            #     hem_s = s[np.newaxis,index_interest]
            #     hem_chan = chans[index_interest]
            #     hem_locs = locs[:,index_interest]
            #     #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #     #                                                        channel_select = ["'"], strict = True, exclude = False)
            # elif hem == 'rh':
            #     hem_locs_empty = locs_empty[:,locs_empty[0,:]>0]
            #     #hem_s, hem_chan, hem_locs = select_channels(s[np.newaxis, :], chans, locs, 
            #     #                                                        channel_select = ["'"], strict = True, exclude = True)
            #     index_interest = list(np.arange(locs.shape[1])[locs[0,:] > 0])
            #     hem_s = s[np.newaxis,index_interest]
            #     hem_chan = chans[index_interest]
            #     hem_locs = locs[:,index_interest]

        #plot brain
        if dist_proj == 0:
            alpha = 0.3
            proj = 'trans'
        else:
            alpha = .3
            proj = 'proj'

        brain = mne.viz.Brain(subjects_dir=surf_path, subject = 'fsaverage',
                            surf='pial', #'pial', 'inflated', 'white',...
                            hemi= 'both',
                            background=background,
                            alpha = 0.3,
                            cortex='classic',  # 'classic'
                            offscreen=True,
                            theme=None)

            
        # if hem_s.shape[0] > 0:
        if s.shape[0] > 0:
            # brain.plotter.add_points(hem_locs.T, render_points_as_spheres=True, point_size = 10, scalars = hem_s[0], cmap = cmap)  #lighting = False
            brain.plotter.add_points(locs.T, render_points_as_spheres=True, point_size = 10, scalars = s, cmap = cmap, show_scalar_bar=False)  #lighting = False
            brain.plotter.update_scalar_bar_range(clim)

        # if hem_locs_empty.shape[1] > 0:
        if locs_empty.shape[1] > 0:
            brain.plotter.add_points(locs_empty.T, render_points_as_spheres=True, point_size = 3, color = 'black')
            # brain.plotter.add_points(hem_locs_empty.T, render_points_as_spheres=True, point_size = 3, color = 'black')

        # if hem == 'lh':
        #     brain.plotter.camera_position = left_viewpoint

        # elif hem == 'rh':
        #     brain.plotter.camera_position = right_viewpoint
        brain.plotter.camera_position = left_viewpoint
        brain.save_image(path_save + title_tag + str(view_index) +  '.png')
        brain.plotter.close()

        # fig, ax = plt.subplots(1,2,figsize=(20,10))
        # left_h = mpimg.imread(path_save + title_tag + 'lh' + str(view_index) + '.png')
        # right_h = mpimg.imread(path_save + title_tag + 'rh' + str(view_index) + '.png')
        # ax[0].imshow(left_h)
        # ax[1].imshow(right_h)

        # # clean up
        # ax[0].spines[['top','bottom','right','left']].set_visible(False)
        # ax[0].set_xticks([])
        # ax[0].set_xticklabels([])
        # ax[0].set_yticks([])
        # ax[0].set_yticklabels([])

        # ax[1].spines[['top','bottom','right','left']].set_visible(False)
        # ax[1].set_xticks([])
        # ax[1].set_xticklabels([])
        # ax[1].set_yticks([])
        # ax[1].set_yticklabels([])

        fig, ax = plt.subplots(1,1,figsize=(17,10))
        both_h = mpimg.imread(path_save + title_tag + str(view_index) + '.png')
        ax.imshow(both_h)
        ax.spines[['top','bottom','right','left']].set_visible(False)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        img_list.append(fig)
        plt.close()

    return img_list
