# Ref: https://github.com/dy1901/ecg_plot

import io
from PIL import Image

import os
import cv2
import numpy as np
from math import ceil 
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def signal_to_image(ecg_signal, sample_rate=500, title="", show_grid=False, show_lead_name=False, is_crop=True, is_bin=False, is_12_individual=False, **kwargs):
    if not is_12_individual:
        if is_crop:
            plot_func = plot_crop
        else:
            plot_func = plot
        plot_func(ecg_signal, 
                sample_rate=sample_rate, 
                title=title, 
                show_grid=show_grid,
                show_lead_name=show_lead_name,
                **kwargs)

        buf = io.BytesIO()
        plt.ioff()
        plt.savefig(buf, format='jpg', dpi=100)
        plt.close()
        buf.seek(0)

        if not is_bin:
            image = np.array(Image.open(buf).convert("RGB")) # (H, W, 3)
            # this is neccessary
            image[0] = image[20]
            image[:, 0] = image[:, 20]
            return image
        else:
            image = np.array(Image.open(buf).convert("L"))  # to gray (H, W)
            image[0] = image[20]
            image[:, 0] = image[:, 20]

            bin_image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV + 
                                                cv2.THRESH_OTSU)[1]
            return np.expand_dims(bin_image / 255.0, -1).astype(np.float32)    # unsqueeze(-1) add channel dimension
    else:
        images = []
        for lead in ecg_signal:
            plot_1(lead, 500, "", show_grid=show_grid)

            buf = io.BytesIO()
            plt.ioff()
            plt.savefig(buf, format='jpg', dpi=100)
            plt.close()
            buf.seek(0)

            image = np.array(Image.open(buf).convert("L"))  # to gray (H, W)
            image[0] = image[20]
            image[:, 0] = image[:, 20]

            if is_bin:
                image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV + 
                                                cv2.THRESH_OTSU)[1]

            images.append((image / 255.0).astype(np.float32))    # normalize

        return np.stack(images, axis=-1)     # H, W, 12
        

def plot_1(
        ecg,
        sample_rate = 500,
        title = "ECG 1",
        style = None,
        row_height = 6,
        show_grid = True,
    ):
    '''
    ecg: 1d array
    '''
    secs = len(ecg) / sample_rate
    line_width = 0.5
    display_factor = 1
    fig, ax = plt.subplots(figsize=(secs * display_factor, row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = secs
    y_min = row_height/4 - (1/2)*row_height
    y_max = row_height/4

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0.7)

    if(show_grid):
        ax.set_xticks(np.arange(x_min, x_max, 0.2))    
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)

    step = 1.0/sample_rate
    ax.plot(
        np.arange(0, len(ecg)*step, step), 
        ecg,
        linewidth=line_width * display_factor, 
        color=color_line
        )
    

lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot(
        ecg, 
        sample_rate    = 500, 
        title          = 'ECG 12', 
        lead_index     = lead_index, 
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 0.5
    fig, ax = plt.subplots(figsize=(secs*columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns*secs
    y_min = row_height/4 - (rows/2)*row_height
    y_max = row_height/4

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0.7)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.5, ecg[t_lead][0] + y_offset + 0.5], linewidth=line_width * display_factor * 2, color=color_line)

         
                t_lead = lead_order[c * rows + i]
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step) + x_offset, 
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line
                    )

def plot_crop(
        ecg, 
        sample_rate    = 500, 
        title          = 'ECG 12', 
        lead_index     = lead_index, 
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        extra_lead     = [],
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """
    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    ecg_length = int(len(ecg[0]) / columns)
    secs  = ecg_length / sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    all_rows = rows + len(extra_lead)
    # display_factor = 2.5
    display_factor = 1
    line_width = 0.5
    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, all_rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns * secs
    y_min = row_height/4 - (all_rows/2)*row_height
    y_max = row_height/4

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0.7)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [y_offset - 0.35, y_offset + 0.35], linewidth=line_width * display_factor, color=color_line)
         
                t_lead = lead_order[c * rows + i]
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)

                wav_offset = c * ecg_length
                ax.plot(
                    np.arange(0, len(ecg[t_lead, :ecg_length])*step, step) + x_offset, 
                    ecg[t_lead, wav_offset : wav_offset + ecg_length] + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line
                    )

    for i in range(rows, all_rows):
        y_offset = -(row_height/2) * ceil(i % all_rows)
        # if (y_offset < -5):
        #     y_offset = y_offset + 0.25

        x_offset = 0
        lead_name = extra_lead[i % rows]
        t_lead = lead_index.index(lead_name)
    
        step = 1.0/sample_rate
        if(show_lead_name):
            ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)

        ax.plot(
            np.arange(0, len(ecg[t_lead])*step, step) + x_offset, 
            ecg[t_lead] + y_offset,
            linewidth=line_width * display_factor, 
            color=color_line
            )

def plot5s(
        ecg, 
        sample_rate    = 500, 
        title          = 'ECG 12', 
        lead_index     = lead_index, 
        lead_order     = None,
        style          = None,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """
    columns = 2
    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    ecg_length = int(len(ecg[0]) / 2)
    secs  = ecg_length / sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 0.5
    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns * secs
    y_min = row_height/4 - (rows/2)*row_height
    y_max = row_height/4

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0.7)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, 2):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [y_offset - 0.35, y_offset + 0.35], linewidth=line_width * display_factor, color=color_line)

         
                t_lead = lead_order[c * rows + i]
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead, :ecg_length])*step, step) + x_offset, 
                    ecg[t_lead, x_offset : ecg_length + x_offset] + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line
                    )
                
 
DEFAULT_PATH = './'
show_counter = 1
def show_svg(tmp_path = DEFAULT_PATH):
    """Plot multi lead ECG chart.
    # Arguments
        tmp_path: path for temporary saving the result svg file
    """ 
    global show_counter
    file_name = tmp_path + "show_tmp_file_{}.svg".format(show_counter)
    plt.savefig(file_name)
    os.system("open {}".format(file_name))
    show_counter += 1
    plt.close()

def show():
    plt.show()


def save_as_png(file_name, path = DEFAULT_PATH, dpi = 100, layout='tight'):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
        dpi      : set dots per inch (dpi) for the saved image
        layout   : Set equal to "tight" to include ax labels on saved image
    """
    plt.ioff()
    plt.savefig(path + file_name + '.png', dpi = dpi, bbox_inches=layout)
    plt.close()

def save_as_svg(file_name, path = DEFAULT_PATH):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
    """
    plt.ioff()
    plt.savefig(path + file_name + '.svg')
    plt.close()

def save_as_jpg(file_name, path = DEFAULT_PATH):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
    """
    plt.ioff()
    plt.savefig(path + file_name + '.jpg')
    plt.close()
