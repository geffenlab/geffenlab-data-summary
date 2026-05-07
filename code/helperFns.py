import math
import os

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def leave_one_out_svm(X, Y):
    n_samples = X.shape[0]
    predictions = []

    # Loop over each row (sample) in X
    for i in range(n_samples):
        # Create train/test sets: all rows except i for training, i-th row for testing
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i)

        X_test = X[i].reshape(1, -1)  # Reshape to keep it as a 2D array

        X_train = StandardScaler().fit_transform(X_train)

        # Train SVM with a linear kernel
        clf = SVC(kernel='linear')
        clf.fit(X_train, Y_train)

        # Predict the label for the held-out row
        prediction = clf.predict(X_test)
        predictions.append(prediction[0])

    # Calculate overall accuracy
    accuracy = accuracy_score(Y, predictions)
    return predictions, accuracy

def compute_psth(spike_times, event_times, bins, window=(-0.2, 0.8), comparison_window=0.05):
    """
    Vectorized computation of PSTH for each neuron and performs a t-test to compare the 50ms
    before and after the event time. Also returns whether spikes increased or decreased. [ChatGPT]
    
    Parameters:
    - spike_times: List of arrays, where each array contains spike times for a neuron.
    - event_times: 1D array of event times (stimulus or task events).
    - bins: Number of bins for the PSTH.
    - window: Time window around each event (default is -0.5 to 0.5 seconds).
    - comparison_window: Time window for comparison before and after event (default is 50ms).
    
    Returns:
    - psth_list: 2D array of PSTHs for each neuron (neurons x bins).
    - bin_edges: The edges of the bins used for histogram.
    - ttest_results: List of p-values from t-tests for each neuron.
    - change_results: List indicating whether the spike rate increased, decreased, or stayed the same.
    """
    
    if not isinstance(event_times, np.ndarray):
        event_times = np.array(event_times)

    bin_edges = np.linspace(window[0], window[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    psth_list = []
    ttest_results = []
    change_results = []
    
    # Define the time window for the 50ms before and after the event
    pre_window = (-comparison_window, 0)
    post_window = (0, comparison_window)
    
    for neuron_spikes in spike_times:

        neuron_spikes = np.array(neuron_spikes)

        # Align spike times to events
        aligned_spikes = neuron_spikes[:, np.newaxis] - event_times[np.newaxis, :]
        aligned_spikes = aligned_spikes.ravel()  # Flatten the array
        
        # Compute PSTH for the neuron
        spike_counts, _ = np.histogram(aligned_spikes, bins=bin_edges)
        spike_counts = spike_counts / (len(event_times) * (bin_edges[1] - bin_edges[0]))
        psth_list.append(spike_counts)
        
        # Compare 50ms before and after the event
        pre_counts = np.histogram(aligned_spikes, bins=np.linspace(pre_window[0], pre_window[1], bins + 1))[0]
        post_counts = np.histogram(aligned_spikes, bins=np.linspace(post_window[0], post_window[1], bins + 1))[0]
        
        # Perform paired t-test
        _, p_value = ttest_rel(pre_counts, post_counts)
        ttest_results.append(p_value)
        
        # Check if spikes increased, decreased, or stayed the same
        pre_mean = np.mean(pre_counts)
        post_mean = np.mean(post_counts)
        
        if post_mean > pre_mean:
            change_results.append("Increased")
        elif post_mean < pre_mean:
            change_results.append("Decreased")
        else:
            change_results.append("No change")
    
    return np.array(psth_list), np.array(bin_centers), np.array(ttest_results), np.array(change_results)

def computeAndPlotPCs(X, labels = [], n_components = 2):

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components = n_components)

    principalComponents = pca.fit_transform(X)

    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

    fig, ax = plt.subplots()

    if len(labels) == 0:
        im = ax.scatter(principalDf['pc1'], principalDf['pc2'])
    else:
        im = ax.scatter(principalDf['pc1'], principalDf['pc2'], c = labels)
        fig.colorbar(im, ax=ax)

    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.grid()

    plt.show()
 
    return fig, ax, principalDf

def computePCs(X, n_components = 2):

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components = n_components)

    principalComponents = pca.fit_transform(X)

    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

    return principalDf

def gen_tensor(time_edges, neurons, events, spikes_df, nSpikes = False):
    '''
    Generates an i x j x k tensor, where i = # events, j = # neurons, k = # time bins

    Args:
    time_edges: temporal edges to bin region around each event
    neurons: IDs of neuronal clusters to include in the tensor
    events: array of times associated with events of interest (say, stimulus onsets)
    spikes_df: pandas dataframe that has one row for each spike recorded. The first column is the ID
        associated with the cluster that spiked, while the second column is the time (in seconds) when
        the spike was recorded

    Note: If time_edges has length 2, this function returns a i x j matrix, not a tensor.    

    Returns:
    mat: i x j x k tensor
    '''
    stepT = time_edges[:-1]

    mat = np.zeros((len(events),len(neurons), len(stepT)))

    for isp, spi in enumerate(neurons):
        sp = spikes_df.loc[spikes_df['cluster'] == spi, 'time'].values
        for ie, ev in enumerate(events):
            for s, step in enumerate(stepT):
                winTemp = (time_edges[s], time_edges[s+1])
                mat[ie,isp,s]  = np.sum(np.logical_and(sp > (ev + winTemp[0]),sp < (ev + winTemp[1])))

    if nSpikes == False:
        mat /= (winTemp[1] - winTemp[0])

    if len(stepT) == 1:
        mat = np.squeeze(mat)
    
    return mat 

def plot_trial_aligned_raster(spike_df, stim_df, selected_clusters=None, ax=None,
                              time_col="time", cluster_col="cluster", 
                              stim_time_col="stim_time", stim_freq_col="stim",
                              time_window=(-0.2, 0.8)):  
    """
    Plots a trial-aligned raster with horizontal lines separating stimulus frequencies and one label per frequency. [ChatGPT]
    
    Parameters:
    spike_df (pd.DataFrame): DataFrame with spike times and cluster labels.
    stim_df (pd.DataFrame): DataFrame with stimulus times and their frequencies.
    selected_clusters (list or None): List of cluster IDs to plot (if None, plot all).
    ax (matplotlib axis or None): Axis to plot on (if None, creates a new figure).
    time_col (str): Column name for spike times.
    cluster_col (str): Column name for cluster (neuron ID).
    stim_time_col (str): Column name for stimulus presentation times.
    stim_freq_col (str): Column name for stimulus frequency.
    time_window (tuple): Start and end time relative to stimulus onset (e.g., (-0.2, 0.8)).
    """
    # If specific clusters are requested, filter the DataFrame
    if selected_clusters is not None:
        spike_df = spike_df[spike_df[cluster_col].isin(selected_clusters)]

    trial_data = []

    # Iterate over each stimulus presentation
    for trial_idx, (stim_time, stim_freq) in enumerate(zip(stim_df[stim_time_col], stim_df[stim_freq_col])):
        trial_start = stim_time + time_window[0]
        trial_end = stim_time + time_window[1]

        # Extract spikes that occur in the time window around this stimulus
        trial_spikes = spike_df[(spike_df[time_col] >= trial_start) & (spike_df[time_col] <= trial_end)].copy()
        trial_spikes["relative_time"] = trial_spikes[time_col] - stim_time  # Align to stimulus onset
        trial_spikes["trial"] = trial_idx  # Assign trial number
        trial_spikes["stim_freq"] = stim_freq  # Store stimulus frequency
        trial_data.append(trial_spikes)

    # Combine all trial data
    aligned_spikes = pd.concat(trial_data, ignore_index=True)

    # Sort trials by stimulus frequency
    stim_df_sorted = stim_df.sort_values(by=stim_freq_col, ascending=True).reset_index()
    trial_order = {idx: i for i, idx in enumerate(stim_df_sorted["index"])}
    aligned_spikes["sorted_trial"] = aligned_spikes["trial"].map(trial_order)

    # Create figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot spikes
    for trial in sorted(aligned_spikes["sorted_trial"].unique()):
        trial_spikes = aligned_spikes[aligned_spikes["sorted_trial"] == trial]

        temp = np.array([trial] * len(trial_spikes))
        ax.vlines(trial_spikes["relative_time"], temp - 1, temp + 1, color="black", linewidth=1)

        #ax.scatter(trial_spikes["relative_time"], [trial] * len(trial_spikes), s=2, color="black")

    # Add stimulus onset as a vertical line at time 0
    #x.axvline(0, color="red", linestyle="--", alpha=0.8, label="Stimulus Onset")
    #ax.axvline(0.5, color="red", linestyle="--", alpha=0.8, label="Stimulus Offset")

    # Add horizontal lines separating stimulus frequencies
    unique_freqs = stim_df_sorted[stim_freq_col].unique()
    trial_bounds_all = np.cumsum(np.bincount(stim_df_sorted[stim_freq_col].rank(method="dense").astype(int)))
    top = trial_bounds_all[-1]
    trial_bounds = trial_bounds_all[:-1]
    for ib, bound in enumerate(trial_bounds):
        if ib < len(trial_bounds): # - 1:
            ax.axhline(bound - 0.5, color="blue", linestyle="--", alpha=0.6)
        frac = np.round((unique_freqs[ib] - 12.551)/(14.773 - 12.551), 3)
        rect = Rectangle((-0.01, bound), width=0.01, height= trial_bounds_all[ib+1] - bound, facecolor=[frac, 0.2, 1-frac])
        ax.add_patch(rect)

    # Assign one label per unique stimulus frequency
    first_trial_per_freq = stim_df_sorted.groupby(stim_freq_col)["index"].first()

    # Compute the center of each frequency's trial block
    y_ticks = stim_df_sorted.groupby(stim_freq_col)["index"].apply(lambda trials: (trial_order[trials.iloc[0]] + trial_order[trials.iloc[-1]]) / 2).values

    #y_labels = [f"{np.round((2**freq)/1000,1)} kHz" for freq in first_trial_per_freq.index]
    y_labels = [f"{np.round((2**freq)/1000,1)}" for freq in first_trial_per_freq.index]

    # Set y-axis labels with only one per frequency
    ax.set_yticks(y_ticks[1::2])
    ax.set_yticklabels(y_labels[1::2])

    # Formatting
    ax.set_xlabel("Time From Stim Onset (s)")
    ax.set_ylabel("Stimulus Frequency (kHz)")
    #ax.set_title("Trial-Aligned Raster Plot with Single Frequency Labels")
    #ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(time_window)
    ax.set_ylim(0, top)

    #ax.axvspan(0, 0.1, color='gray', alpha=0.3)
    ax.axvspan(0, 0.5, color='gray', alpha=0.3)
    #ax.axvspan(0.5, 0.6, color='blue', alpha=0.3)

    return ax  # Return the axis for further customization if needed

##

def plot_factors(U, labs, times, plots='line', fig=None, axes=None, scatter_kw=dict(),
                 line_kw=dict(), bar_kw=dict(), **kwargs):
    
    """Plots a KTensor. Adapted from Alex Williams's github: https://github.com/neurostatslab/tensortools

    Note: Each keyword option is broadcast to all modes of the KTensor. For
    example, if `U` is a 3rd-order tensor (i.e. `U.ndim == 3`) then
    `plot_factors(U, plots=['line','bar','scatter'])` plots all factors for the
    first mode as a line plot, the second as a bar plot, and the third mode as
    a scatterplot. But, thanks to broadcasting semantics,
    `plot_factors(U, color='line')` produces line plots for each mode.

    Parameters
    ----------
    U : KTensor
        Kruskal tensor to be plotted.

    plots : str or list
        One of {'bar','line','scatter'} to specify the type of plot for each
        factor. The default is 'line'.
    fig : matplotlib Figure object
        If provided, add plots to the specified figure. The figure must have a
        sufficient number of axes objects.
    axes : 2d numpy array of matplotlib Axes objects
        If provided, add plots to the specified figure.
    scatter_kw : dict or sequence of dicts
        Keyword arguments provided to scatterplots. If a single dict is
        provided, these options are broadcasted to all modes.
    line_kw : dict or sequence of dicts
        Keyword arguments provided to line plots. If a single dict is provided,
        these options are broadcasted to all modes.
    bar_kw : dict or sequence of dicts
        Keyword arguments provided to bar plots. If a single dict is provided,
        these options are broadcasted to all modes.
    **kwargs : dict
        Additional keyword parameters are passed to the `subplots(...)`
        function to specify options such as `figsize` and `gridspec_kw`. See
        `matplotlib.pyplot.subplots(...)` documentation for more info.
    """

    # ~~~~~~~~~~~~~
    # PARSE OPTIONS
    # ~~~~~~~~~~~~~
    kwargs.setdefault('figsize', (8, U.rank))

    # parse optional inputs
    # plots = _broadcast_arg(U, plots, str, 'plots')
    # bar_kw = _broadcast_arg(U, bar_kw, dict, 'bar_kw')
    # line_kw = _broadcast_arg(U, line_kw, dict, 'line_kw')
    # scatter_kw = _broadcast_arg(U, scatter_kw, dict, 'scatter_kw')

    # default scatterplot options
    for sckw in scatter_kw:
        sckw.setdefault('edgecolor', 'none')
        sckw.setdefault('s', 10)

    colormap = np.array(['b', 'purple', 'r'])

    # ~~~~~~~~~~~~~~
    # SETUP SUBPLOTS
    # ~~~~~~~~~~~~~~
    if fig is None and axes is None:
        fig, axes = plt.subplots(U.rank, U.ndim, **kwargs)
        # make sure axes is a 2d-array
        if U.rank == 1:
            axes = axes[None, :]

    # if axes are passed in, identify figure
    elif fig is None:
        fig = axes[0, 0].get_figure()

    # if figure is passed, identify axes
    else:
        axes = np.array(fig.get_axes(), dtype=object).reshape(U.rank, U.ndim)

    # main loop, plot each factor
    plot_obj = np.empty((U.rank, U.ndim), dtype=object)
    for r in range(U.rank):
        for i, f in enumerate(U):
            # start plots at 1 instead of zero
            x = np.arange(1, f.shape[0]+1)

            # determine type of plot
            if plots[i] == 'bar':
                plot_obj[r, i] = axes[r, i].bar(x, f[:, r])#, **bar_kw[i])
                x_label = 'Neuron'
            elif plots[i] == 'scatter':
                plot_obj[r, i] = axes[r, i].scatter(x, f[:, r], c = colormap[labs-1], s = 10)#, **scatter_kw[i])
                x_label = 'Trial'
            elif plots[i] == 'line':
                plot_obj[r, i] = axes[r, i].plot(times, f[:, r], '-')#, **line_kw[i])
                axes[r,i].axvspan(0, 0.5, color='gray', alpha=0.3)
                x_label = 'Time (s)'
            else:
                raise ValueError('invalid plot type')

            # format axes
            axes[r, i].locator_params(nbins=4)
            axes[r, i].spines['top'].set_visible(False)
            axes[r, i].spines['right'].set_visible(False)
            axes[r, i].xaxis.set_tick_params(direction='out')
            axes[r, i].yaxis.set_tick_params(direction='out')
            axes[r, i].yaxis.set_ticks_position('left')
            axes[r, i].xaxis.set_ticks_position('bottom')

            # remove xticks on all but bottom row
            if r != U.rank-1:
                plt.setp(axes[r, i].get_xticklabels(), visible=False)
            else:
                axes[r,i].set_xlabel(x_label)

    # link y-axes within columns
    # for i in range(U.ndim):
    #     yl = [a.get_ylim() for a in axes[:, i]]
    #     y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
    #     [a.set_ylim((y0, y1)) for a in axes[:, i]]

    # format y-ticks
    for r in range(U.rank):
        for i in range(U.ndim):
            # only two labels
            ymin, ymax = np.round(axes[r, i].get_ylim(), 2)
            axes[r, i].set_ylim((ymin, ymax))

            # remove decimals from labels
            if ymin.is_integer():
                ymin = int(ymin)
            if ymax.is_integer():
                ymax = int(ymax)

            # update plot
            axes[r, i].set_yticks([ymin, ymax])

    plt.tight_layout()

    return fig, axes, plot_obj

from scipy.ndimage import gaussian_filter1d
def smooth(data, window_size = 0.7):
    return gaussian_filter1d(data, window_size)

def plotNeuron(kIdx, tensor, xvals, onset_response, offset_response, trial_events, spikes_df, kept_clusters, effect_direction, cat_effect_direction, tag, sess_date, save_fig = False, save_name = None, show = True):

    temp = trial_events[np.logical_and(trial_events['resp'] == 1, trial_events['cat'] == 1)].head(1)
    if temp['acc'].values[0] == 1:
        lowDir = temp['dir'].values[0]
        highDir = 3 - temp['dir'].values[0]
    else:
        lowDir = 3 - temp['dir'].values[0]
        highDir = temp['dir'].values[0]

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1, 1])  # First column wider

    ax1 = fig.add_subplot(gs[0, 0])  # Top-left subplot
    ax2 = fig.add_subplot(gs[1, 0])  # Large subplot spanning middle-left & bottom-left
    ax3 = fig.add_subplot(gs[2, 0])  # Top-right subplot
    ax4 = fig.add_subplot(gs[3, 0])  # Middle-right subplot

    ax5 = fig.add_subplot(gs[0:2, 1])  # Top-right subplot
    ax6 = fig.add_subplot(gs[2:, 1])  # Bottom-right subplot

    freqs = np.unique(trial_events['stim'])
    freq_resp_onset = np.zeros(len(freqs))
    freq_resp_offset = np.zeros(len(freqs))

    for i, f in enumerate(freqs):
        freq_resp_onset[i] = np.mean(onset_response[trial_events['stim'] == f, kIdx],0)
        freq_resp_offset[i] = np.mean(offset_response[trial_events['stim'] == f, kIdx],0)

    ax1.plot(freqs, freq_resp_onset, 'ko-', label = 'onset')
    #ax1.plot(freqs, freq_resp_offset, 'bo-', label = 'offset')
    ax1.set_ylabel('Spikes/s')
    ax1.set_xlabel('Stimulus Frequency (kHz)')

    xts = np.array([6, 8, 12, 16, 24])
    ax1.set_xticks(np.log2(xts*1000))
    ax1.set_xticklabels(xts.astype(int))

    ax1.legend()

    low = np.mean(tensor[trial_events['cat'] == 1,kIdx,:],0)
    med = np.mean(tensor[trial_events['cat'] == 2,kIdx,:],0)
    high = np.mean(tensor[trial_events['cat'] == 3,kIdx,:],0)
    ax2.plot(xvals,smooth(low), color = [0,.2,1], label = 'low')
    ax2.plot(xvals,smooth(med), color = [0.5,.2,0.5], label = 'probe')
    ax2.plot(xvals,smooth(high), color = [1,.2,0], label = 'high')

    chose_low_probe = np.mean(tensor[np.logical_and(trial_events['cat'] == 2, trial_events['dir'] == lowDir),kIdx,:],0)
    chose_high_probe = np.mean(tensor[np.logical_and(trial_events['cat'] == 2, trial_events['dir'] == highDir),kIdx,:],0)
    ax3.plot(xvals,smooth(chose_low_probe), color = [0,.2,1], label = 'probe (chose low)')
    ax3.plot(xvals,smooth(chose_high_probe), color = [1,.2,0], label = 'probe (chose high)')

    left_correct = np.mean(tensor[np.logical_and(trial_events['acc'] == 1, trial_events['dir'] == 1),kIdx,:],0)
    right_correct = np.mean(tensor[np.logical_and(trial_events['acc'] == 1, trial_events['dir'] == 2),kIdx,:],0)
    left_error = np.mean(tensor[np.logical_and(trial_events['acc'] == 0, trial_events['dir'] == 1),kIdx,:],0)
    right_error = np.mean(tensor[np.logical_and(trial_events['acc'] == 0, trial_events['dir'] == 2),kIdx,:],0)
    ax4.plot(xvals,smooth(left_correct), color = 'orange', label = 'left corr')
    ax4.plot(xvals,smooth(right_correct), color = 'green', label = 'right corr')
    ax4.plot(xvals,smooth(left_error), color = 'orange', linestyle = '--', label = 'left error')
    ax4.plot(xvals,smooth(right_error), color = 'green', linestyle = '--', label = 'right error')

#    no_resp = np.mean(tensor[trial_events['resp'] == 0,kIdx,:],0)
#    resp = np.mean(tensor[trial_events['resp'] == 1,kIdx,:],0)
#    ax4.plot(xvals,smooth(no_resp), label = 'no response')
#    ax4.plot(xvals,smooth(resp), label = 'response')

    for ax in (ax2, ax3, ax4):
        ax.axvspan(0, 0.5, color='gray', alpha=0.3)
        ax.axvline(0, color = 'k', linestyle = '--')
        ax.legend(loc='upper right')
        ax.set_ylabel('Spikes/s')
        ax.set_xlabel('Time From Stim Onset (s)')
        ax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False,         # ticks along the top edge are off
            ) # labels along the bottom edge are off
        ax.spines[['right', 'top']].set_visible(False)

    plot_trial_aligned_raster(spikes_df, trial_events, selected_clusters = [kept_clusters[kIdx]], time_window = [-0.1, 0.6], ax = ax5)

    dyn = 0.7*np.max(np.mean(tensor[:,kIdx,:],0))
    stims = np.unique(trial_events['stim'])
    offsets = []
    for si, stim in enumerate(stims):
        frac = np.round((stim - 12.551)/(14.773 - 12.551), 3)
        offsets.append(dyn*si)
        ax6.plot(xvals, dyn*si + smooth(np.mean(tensor[trial_events['stim'] == stim,kIdx,:],0)), color = [frac, 0.2, 1-frac])
    for off in offsets:
        ax6.axhline(off, color = 'k', linestyle = '--', linewidth = 0.5)

    ax6.set_yticks(offsets)

    #y_labels = [f"{np.round((2**freq)/1000,1)} kHz" for freq in stims]
    y_labels = [f"{np.round((2**freq)/1000,1)}" for freq in stims]
    ax6.set_yticklabels(y_labels)

    ax6.axvspan(0, 0.5, color='gray', alpha=0.3)
    ax6.set_xlim(-0.1,0.6)
    ax6.set_ylabel('Stimulus Frequency (kHz)')
    ax6.set_xlabel('Time From Stim Onset (s)')
    ax6.spines[['right', 'top']].set_visible(False)

    if effect_direction[kIdx] == -1:
        effect_dir = 'Onset Suppressed'
    elif effect_direction[kIdx] == 1:
        effect_dir = 'Onset Enhanced'
    else:
        effect_dir = 'No Onset Effect'

    if cat_effect_direction[kIdx] == 0:
        cat_effect_dir = ', Not Category Selective'
    else:
        cat_effect_dir = ', Category Selective'

    title_text = tag + ', ' + sess_date + ': Cluster #' + str(kept_clusters[kIdx]) + ', ' + effect_dir + cat_effect_dir

    fig.suptitle(title_text)
    fig.tight_layout()

    if show:
        plt.show()

    if save_fig: 
        plt.savefig(save_name)
    
    plt.close(fig)

    ##

def quick_plot(spikes_df, cID, trial_events, bin_size = 0.005, ax = None):

    steps = np.arange(-0.1, 0.6, bin_size)
    xvals = (steps[0:-1] + steps[1:])/2

    tensor = gen_tensor(steps, (cID,), trial_events['stim_time'], spikes_df)
    tensor = np.squeeze(tensor)

    low = np.mean(tensor[trial_events['cat'] == 1,:],0)
    med = np.mean(tensor[trial_events['cat'] == 2,:],0)
    high = np.mean(tensor[trial_events['cat'] == 3,:],0)
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(xvals, smooth(low), color = [0,.2,1], label = 'low')
    ax.plot(xvals, smooth(med), color = [0.5,.2,0.5], label = 'probe')
    ax.plot(xvals, smooth(high), color = [1,.2,0], label = 'high')
    ax.axvspan(0, 0.5, color = 'grey', alpha = 0.5)

    ax.set_title(str(cID))
    ax.legend()
    ax.spines[['right','top']].set_visible(False)

    ax.set_ylabel('Spikes/s')
    ax.set_xlabel('Time From Stim Onset (s)')

    #plt.tight_layout()
    #plt.show()

def quick_plot_resp(spikes_df, cID, trial_events, bin_size = 0.005, ax = None, show_legend = False):

    steps = np.arange(-0.4, 0.2, bin_size)
    xvals = (steps[0:-1] + steps[1:])/2

    tensor = gen_tensor(steps, (cID,), trial_events['resp_time'], spikes_df)
    tensor = np.squeeze(tensor)

    low = np.mean(tensor[trial_events['cat'] == 1,:],0)
    med = np.mean(tensor[trial_events['cat'] == 2,:],0)
    high = np.mean(tensor[trial_events['cat'] == 3,:],0)
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(xvals, smooth(low), color = [0,.2,1], label = 'low')
    ax.plot(xvals, smooth(med), color = [0.5,.2,0.5], label = 'probe')
    ax.plot(xvals, smooth(high), color = [1,.2,0], label = 'high')

    ax.axvline(0, color = 'k', linestyle = '--')

    ax.set_title(str(cID))

    if show_legend:
        ax.legend(loc = 0, prop={'size': 12})

    ax.spines[['right','top']].set_visible(False)

    ax.set_ylabel('Spikes/s')
    ax.set_xlabel('Time From Response (s)')

def simple_condition_plot(spikes_df, cID, trial_events, bin_size = 0.005, ax = None, show_legend = False):

    steps = np.arange(-0.1, 1, bin_size)
    xvals = (steps[0:-1] + steps[1:])/2

    tensor = gen_tensor(steps, (cID,), trial_events['stim_time'], spikes_df)
    tensor = np.squeeze(tensor)

    # med = np.mean(tensor[trial_events['cat'] == 2,:],0)

    low = np.mean(tensor[trial_events['cat'] == 1,:],0)
    high = np.mean(tensor[trial_events['cat'] == 3,:],0)

    n_trials, n_timepoints = tensor.shape
    min_trials = 3

    def safe_condition_mean(cat_val, dir_val):
        # Align trial metadata with tensor
        cat_array = np.asarray(trial_events['cat'])[:n_trials]
        dir_array = np.asarray(trial_events['dir'])[:n_trials]

        # Find matching trials
        mask = np.logical_and(cat_array == cat_val, dir_array == dir_val)
        trial_idxs = np.where(mask)[0]

        if len(trial_idxs) >= min_trials:
            return np.mean(tensor[trial_idxs, :], axis=0)  # average across trials
        else:
            return np.full((n_timepoints,), np.nan)

    # Condition-specific traces
    low_tl  = safe_condition_mean(1, 1)
    low_tr  = safe_condition_mean(1, 2)
    low_nr  = safe_condition_mean(1, 0)

    high_tl = safe_condition_mean(3, 1)
    high_tr = safe_condition_mean(3, 2)
    high_nr = safe_condition_mean(3, 0)

    if ax is None:
        fig, ax = plt.subplots()

    #ax.plot(xvals, hf.smooth(med), color = [0.5,.2,0.5], label = 'probe')

    # Get the parent figure and the position of the given ax
    fig = ax.figure
    pos = ax.get_position()
    ax.remove()  # Remove the original Axes from the figure

    # Compute 3 equally spaced vertical axes inside the original ax slot
    spacing = 0.02  # small vertical space between panels
    height = (pos.height - 2 * spacing) / 3

    # Compute new axes positions
    pos1 = [pos.x0, pos.y0 + 2 * (height + spacing), pos.width, height]
    pos2 = [pos.x0, pos.y0 +     (height + spacing), pos.width, height]
    pos3 = [pos.x0, pos.y0,                      pos.width, height]

    ax1 = fig.add_axes(pos1)
    ax2 = fig.add_axes(pos2)
    ax3 = fig.add_axes(pos3)

    ax1.text(0.45, 0.7, str(cID), transform=ax1.transAxes,
         ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.plot(xvals, smooth(low), color = '#377eb8', label = 'low', linewidth = 3)
    ax1.plot(xvals, smooth(high), color = '#e41a1c', label = 'high', linewidth = 3)
    ax1.set_xticklabels([])
    
    ax2.plot(xvals, smooth(low_tl), color = '#6baed6', label = 'low (lick left)')
    ax2.plot(xvals, smooth(low_tr), color = '#08519c', label = 'low (lick right)')
    ax2.set_xticklabels([])
    
    ax3.plot(xvals, smooth(high_tl), color = '#fb6a4a', label = 'high (lick left)')
    ax3.plot(xvals, smooth(high_tr), color = '#a50f15', label = 'high (lick right)')
    
    ax3.set_xlabel('Time From Stim Onset (s)')
    
    for sub_ax in [ax1, ax2, ax3]:
        sub_ax.axvline(0, color = 'k', linestyle = '--')
        sub_ax.axvline(0.5, color = 'k', linestyle = '--')
        sub_ax.set_ylabel('Spikes/s')
        sub_ax.spines[['right','top']].set_visible(False)
        sub_ax.set_xticks((-0.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1))
        sub_ax.tick_params(labelsize = 10)
    
    if show_legend:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()

        handles = handles1 + handles2 + handles3
        labels = labels1 + labels2 + labels3
    else:
        handles, labels = None, None

    return handles, labels

def simple_condition_plot_resp(spikes_df, cID, trial_events, bin_size = 0.005, ax = None, show_legend = False):

    steps = np.arange(-0.7, 0.3, bin_size)
    xvals = (steps[0:-1] + steps[1:])/2

    tensor = gen_tensor(steps, (cID,), trial_events['resp_time'], spikes_df)
    tensor = np.squeeze(tensor)

    left = np.mean(tensor[trial_events['dir'] == 1,:],0)
    right = np.mean(tensor[trial_events['dir'] == 2,:],0)

    n_trials, n_timepoints = tensor.shape
    min_trials = 3

    def safe_condition_mean(cat_val, dir_val):
        # Align trial metadata with tensor
        cat_array = np.asarray(trial_events['cat'])[:n_trials]
        dir_array = np.asarray(trial_events['dir'])[:n_trials]

        # Find matching trials
        mask = np.logical_and(cat_array == cat_val, dir_array == dir_val)
        trial_idxs = np.where(mask)[0]

        if len(trial_idxs) >= min_trials:
            return np.mean(tensor[trial_idxs, :], axis=0)  # average across trials
        else:
            return np.full((n_timepoints,), np.nan)

    # Condition-specific traces
        
    left_sl  = safe_condition_mean(1, 1)
    left_sh  = safe_condition_mean(3, 1)
    left_sp  = safe_condition_mean(2, 1)

    right_sl = safe_condition_mean(1, 2)
    right_sh = safe_condition_mean(3, 2)
    right_sp = safe_condition_mean(2, 2)

    if ax is None:
        fig, ax = plt.subplots()

    #ax.plot(xvals, hf.smooth(med), color = [0.5,.2,0.5], label = 'probe')

    # Get the parent figure and the position of the given ax
    fig = ax.figure
    pos = ax.get_position()
    ax.remove()  # Remove the original Axes from the figure

    # Compute 3 equally spaced vertical axes inside the original ax slot
    spacing = 0.02  # small vertical space between panels
    height = (pos.height - 2 * spacing) / 3

    # Compute new axes positions
    pos1 = [pos.x0, pos.y0 + 2 * (height + spacing), pos.width, height]
    pos2 = [pos.x0, pos.y0 +     (height + spacing), pos.width, height]
    pos3 = [pos.x0, pos.y0,                      pos.width, height]

    ax1 = fig.add_axes(pos1)
    ax2 = fig.add_axes(pos2)
    ax3 = fig.add_axes(pos3)

    ax1.text(0.45, 0.7, str(cID), transform=ax1.transAxes,
         ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.plot(xvals, smooth(left), color = '#7f7f7f', label = 'left') #, linewidth = 3)
    ax1.plot(xvals, smooth(right), color = '#4d4d4d', label = 'right') #, linewidth = 3)
    ax1.set_xticklabels([])
    
    ax2.plot(xvals, smooth(left_sl), color = '#377eb8', label = 'left (stim low)')
    ax2.plot(xvals, smooth(left_sh), color = '#e41a1c', label = 'left (stim high)')
    ax2.set_xticklabels([])
    
    ax3.plot(xvals, smooth(right_sl), color = '#08519c', label = 'right (stim low)')
    ax3.plot(xvals, smooth(right_sh), color = '#a50f15', label = 'right (stim high)')
    
    ax3.set_xlabel('Time From Response (s)')
    
    for sub_ax in [ax1, ax2, ax3]:
        sub_ax.axvline(0, color = 'k', linestyle = '--')
        sub_ax.axvline(0.5, color = 'k', linestyle = '--')
        sub_ax.set_ylabel('Spikes/s')
        sub_ax.spines[['right','top']].set_visible(False)
        sub_ax.set_xticks((-.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3))
        sub_ax.tick_params(labelsize = 10)
    
    if show_legend:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()

        handles = handles1 + handles2 + handles3
        labels = labels1 + labels2 + labels3
    else:
        handles, labels = None, None

    return handles, labels

def plot_behavior_over_trials(ax, trial_events, smoothing_width=20):

    trial_nums = np.arange(len(trial_events))
    acc_raw = trial_events['acc'].values.astype(float)

    # Only count 0 = incorrect and 1 = correct responses (exclude 2 = NR, 99 = probe)
    is_valid = trial_events['acc'].isin([0, 1])
    is_correct = trial_events['acc'] == 1

    # Smooth numerator and denominator separately
    signal = is_correct.astype(float)
    weights = is_valid.astype(float)

    smooth_signal = uniform_filter1d(signal, size=smoothing_width)
    smooth_weights = uniform_filter1d(weights, size=smoothing_width)

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_smooth = smooth_signal / smooth_weights
        acc_smooth[smooth_weights == 0] = np.nan

    is_nr = trial_events['acc'] == 2
    nr_smooth = uniform_filter1d(is_nr.astype(float), size=smoothing_width)

    # Color each dot by actual accuracy
    def acc_to_color(val):
        if val == 1:
            return 'green'
        elif val == 0:
            return 'red'
        elif val == 2:
            return 'blue'
        elif val == 99:
            return 'gray'
        else:
            return 'lightgray'  # fallback

    acc_colors = trial_events['acc'].apply(acc_to_color).values

    # Plot only dots (no line), at smoothed value, colored by real outcome
    ax.scatter(trial_nums, acc_smooth, c=acc_colors, s=15, alpha=0.8, zorder=3)

    # Plot NR trace
    ax.plot(trial_nums, nr_smooth, label='P(No Response)', color='blue', linewidth=1.2, linestyle='--')

    # ---- Smoothed Lick Probabilities ----
    dir_vals = trial_events['dir'].values
    cat_vals = trial_events['cat'].values

    def smoothed_single_prob(cat_vals, dir_vals, target_cat, target_dir, width):
        """
        Compute smoothed P(dir == target_dir | cat == target_cat).
        Returns probability and smoothed denominator for masking.
        """
        is_cat = cat_vals == target_cat
        responded = dir_vals > 0
        valid = is_cat & responded
        is_target = (dir_vals == target_dir) & valid

        signal = is_target.astype(float)
        denom = valid.astype(float)

        smooth_signal = uniform_filter1d(signal, size=width)
        smooth_denom  = uniform_filter1d(denom, size=width)

        with np.errstate(divide='ignore', invalid='ignore'):
            prob = smooth_signal / smooth_denom
            prob[smooth_denom == 0] = np.nan

        return prob, smooth_denom

    # Smooth for low cat
    p_left_low, denom_low = smoothed_single_prob(cat_vals, dir_vals, 1, 1, smoothing_width)
    p_right_low = 1 - p_left_low
    p_left_low[denom_low == 0] = np.nan
    p_right_low[denom_low == 0] = np.nan

    # Smooth for high cat
    p_left_high, denom_high = smoothed_single_prob(cat_vals, dir_vals, 3, 1, smoothing_width)
    p_right_high = 1 - p_left_high
    p_left_high[denom_high == 0] = np.nan
    p_right_high[denom_high == 0] = np.nan

    # Smooth for probe cat
    # p_left_probe, denom_probe = smoothed_single_prob(cat_vals, dir_vals, 2, 1, smoothing_width)
    # p_right_probe = 1 - p_left_probe
    # p_left_probe[denom_probe == 0] = np.nan
    # p_right_probe[denom_probe == 0] = np.nan

    # Plot the traces
    ax.plot(trial_nums, p_left_high,  label='P(Left Lick|High)',  color='#fb6a4a', linewidth=1.5)
    ax.plot(trial_nums, p_left_low,   label='P(Left Lick|Low)',  color='#6baed6', linewidth=1.5)
    #ax.plot(trial_nums, p_left_probe,   label='P(Left Lick|Probe)',  color='#cab2d6', linewidth=1.5)
    
    ax.plot(trial_nums, p_right_high, label='P(Right Lick|High)', color='#a50f15', linewidth=1.5)
    ax.plot(trial_nums, p_right_low,  label='P(Right Lick|Low)', color='#08519c', linewidth=1.5)
    #ax.plot(trial_nums, p_right_probe,   label='P(Right Lick|Probe)',  color='#6a3d9a', linewidth=1.5)

    # ---- Styling ----
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, len(trial_events))
    ax.set_xlabel("Trial Number", fontsize=9)
    ax.set_ylabel("Smoothed Values", fontsize=9)
    ax.set_title("Behavior Over Trials", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.spines[['right','top']].set_visible(False)

    ax.legend(
        loc='center left',
        bbox_to_anchor = (1.00, 0.5),
        fontsize=6,
        frameon=False,
        ncol=1,
        handlelength=2
    )

def plot_response_time_histograms(ax_top, ax_bottom, trial_events, bin_width=0.02, max_rt=1):

    # Filter valid directional lick trials
    valid = (
        trial_events['resp_time'].notna() &
        trial_events['stim_time'].notna() &
        (trial_events['dir'].isin([1, 2]))
    )

    te = trial_events[valid].copy()
    te['rt'] = te['resp_time'] - te['stim_time']

    # Define histogram bins
    bins = np.arange(0, max_rt + bin_width, bin_width)

    # Category color map
    cat_colors = {
        1: '#377eb8',  # Low
        2: '#984ea3',  # Probe
        3: '#e41a1c'   # High
    }

    # Helper to plot all categories for one response direction
    def plot_for_response(ax, dir_val, title):
        for cat_val, color in cat_colors.items():
            subset = te[(te['dir'] == dir_val) & (te['cat'] == cat_val)]
            if not subset.empty:
                ax.hist(subset['rt'], bins=bins, alpha=0.6, color=color, #density = True, stacked = True,
                        label=f"{['Low', 'Probe', 'High'][cat_val - 1]}")
        
        ax.set_xlim(0, max_rt)
        ax.set_ylabel("Trial Count", fontsize=9)
        #ax.set_title(f"{title} Licks", fontsize=9)
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False, title="Stim Category", title_fontsize=8)

    # Top: left licks, Bottom: right licks
    plot_for_response(ax_top,    dir_val=1, title="Left")
    plot_for_response(ax_bottom, dir_val=2, title="Right")
    ax_bottom.set_xlabel("Response Time (s)", fontsize=9)


def batch_plot(identifier, neuron_ids, spikes_df, trial_events, plot_fn, rows=1, cols=5, save_dir = "batch_plots"):

    os.makedirs(save_dir, exist_ok=True)
    neuron_ids = list(neuron_ids)
    per_page = rows * cols
    total_batches = math.ceil(len(neuron_ids) / per_page)
    dpi = plt.rcParams['figure.dpi']
    figsize = (1920 / dpi, 1080 / dpi)


# Need to reorganize the code slightly so that the analyses for the top plots aren't done for each batch.

    # Session-Level Decoding
    steps = np.linspace(-0.1, 1.0, 41)
    tensor = gen_tensor(steps, neuron_ids, trial_events['stim_time'], spikes_df)
    trial_events_temp = trial_events[trial_events['cat'] != 2]
    catt = (np.array(trial_events_temp['cat'])-1)/2
    choicet = np.array(trial_events_temp['dir']) - 1
    accs_cat = np.zeros(np.size(tensor,2))
    accs_choice = accs_cat.copy()
    for ts, x in enumerate(accs_cat):
        temp = tensor[trial_events['cat'] != 2,:,ts]
        temp = StandardScaler().fit_transform(temp)
        pred, accs_cat[ts] = leave_one_out_svm(temp, catt)
        pred, accs_choice[ts] = leave_one_out_svm(temp[choicet >= 0,:], choicet[choicet >= 0])
    timestamps = 0.5*(steps[1:] + steps[:-1])


# Loop starts
    for batch_idx in range(total_batches):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(rows + 1, 5, figure=fig, height_ratios=[1] + [5] * rows)

        # === Manually defined top-row plots ===
        ax_top1 = fig.add_subplot(gs[0, 0])
        ax_top2 = fig.add_subplot(gs[0, 1:3])  # twice as wide
        ax_top4 = fig.add_subplot(gs[0, 4])


        outer_ax = fig.add_subplot(gs[0, 3])
        fig.delaxes(outer_ax)

        # Create nested GridSpec
        inner_gs = GridSpecFromSubplotSpec(
            2, 1,            # 2 vertical subplots
            height_ratios=[1, 1],
            hspace=0.3,
            subplot_spec=gs[0, 3]  # link to parent slot
        )

        ax_top_high = fig.add_subplot(inner_gs[0])
        ax_top_low  = fig.add_subplot(inner_gs[1])

        # Psychometric Function

        beh_temp = trial_events.loc[trial_events['dir'] > 0,:]
        dirs = beh_temp.groupby('stim')['dir'].mean() - 1

        if beh_temp[np.logical_and(beh_temp['acc'] == 1, beh_temp['dir'] == 1)]['cat'].head(1).values == 3:
            dirs = 1 - dirs

        ax_top1.axvline(np.log2(10000), color = 'k', linestyle = '--', linewidth = 1)
        ax_top1.axvline(np.log2(17000), color = 'k', linestyle = '--', linewidth = 1)
        ax_top1.plot(dirs, 'ko-')

        xts = np.array([6, 8, 12, 16, 24])
        ax_top1.set_xticks(np.log2(xts*1000))
        ax_top1.set_xticklabels(xts)

        ax_top1.set_xlabel('Stimulus Frequency (kHz)', fontsize = 9)
        ax_top1.set_ylabel('P("High")', fontsize = 9)
        ax_top1.spines[['right','top']].set_visible(False)
        ax_top1.set_title("Psychometric Function", fontsize = 11)
        ax_top1.tick_params(labelsize = 8, pad = 2)

        # Behavior Over Trials
        plot_behavior_over_trials(ax_top2, trial_events, smoothing_width=20)

        # Response time distributions (which will have two vertical subplots)
        #ax_top3.plot(...)
        plot_response_time_histograms(ax_top_high, ax_top_low, trial_events)
        ax_top_high.set_title('RT Dists', fontsize = 11)

        # Session-Level Decoding Plot
        ax_top4.plot(timestamps, accs_cat, color = 'k', label = 'Category', linewidth = 2)
        ax_top4.plot(timestamps, accs_choice, color = 'k', label = 'Choice', linewidth = 2, linestyle = ':')
        ax_top4.axhline(0.5, color = 'k', linestyle = '--')
        ax_top4.axvline(0, color = 'k', linestyle = '--')
        ax_top4.axvline(0.5, color = 'k', linestyle = '--')
        ax_top4.set_xlabel('Time From Stim Onset (s)', fontsize = 9)
        ax_top4.set_ylabel('SVM Prediction Accuracy', fontsize = 9)
        ax_top4.spines[['right','top']].set_visible(False)
        ax_top4.set_title("SVM Session-Level Decoding", fontsize = 11)
        ax_top4.tick_params(labelsize = 8, pad = 2)
        ax_top4.legend(loc = 'center left', bbox_to_anchor = (.98, 0.5), fontsize=6, frameon=False)

        # === Neuron plots (same as before) ===

        start = batch_idx * per_page
        end = min(start + per_page, len(neuron_ids))
        batch_ids = neuron_ids[start:end]

        legend_handles, legend_labels = None, None

        for idx, neuron_id in enumerate(batch_ids):
            row = idx // cols + 1  # +1 to account for top row
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Neuron {neuron_id}")

            handles, labels = plot_fn(spikes_df, neuron_id, trial_events, bin_size = 0.01, ax = ax, show_legend = (idx == 0))
            if idx == 0 and handles and labels:
                legend_handles, legend_labels = handles, labels

        # Hide unused batch cells
        for idx in range(len(batch_ids), rows * cols):
            row = idx // cols + 1
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')

        #if legend_handles and legend_labels:
        #    fig.legend(legend_handles, legend_labels,
        #               loc='center left', bbox_to_anchor=(0.01, 0.4),
        #               ncol=2, fontsize=9, frameon=False)

        fig.suptitle(f"{identifier} Neuron Group {batch_idx + 1}", fontsize=16, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.99])  # or try 0.985 to squeeze tighter
        fig.savefig(os.path.join(save_dir, f"{identifier}_neurons_{batch_idx+1}.png"), bbox_inches="tight")
        plt.close(fig)

def global_label_group(fig, ax, traces, colors, labels, x_offset=1.01):
    """
    Adds labels to the right of the plot, aligned with evenly spaced y positions.
    Uses axis-relative coordinates, so it works reliably inside batch figures.
    """
    valid = [(y, c, l) for y, c, l in zip(traces, colors, labels) if not np.all(np.isnan(y))]

    if not valid:
        return

    n = len(valid)
    y_coords = np.linspace(0.75, 0.25, n)

    for (ydata, color, label), y in zip(valid, y_coords):
        ax.annotate(
            label,
            xy=(x_offset, y),  # x just outside axis bounds
            xycoords='axes fraction',
            fontsize=6,
            color=color,
            ha='left',
            va='center',
            clip_on=False
        )

def complex_condition_plot(spikes_df, cID, trial_events, bin_size = 0.01, ax = None, show_legend = False):

# First, let's plot the stimulus-triggered functions:

    steps = np.arange(-0.1, 1, bin_size)
    xvals = (steps[0:-1] + steps[1:])/2

    tensor = gen_tensor(steps, (cID,), trial_events['stim_time'], spikes_df)
    tensor = np.squeeze(tensor)

    med = np.mean(tensor[trial_events['cat'] == 2,:],0)
    low = np.mean(tensor[trial_events['cat'] == 1,:],0)
    high = np.mean(tensor[trial_events['cat'] == 3,:],0)

    n_trials, n_timepoints = tensor.shape
    min_trials = 3

    def safe_condition_mean(cat_val, dir_val):
        # Align trial metadata with tensor
        cat_array = np.asarray(trial_events['cat'])[:n_trials]
        dir_array = np.asarray(trial_events['dir'])[:n_trials]

        # Find matching trials
        mask = np.logical_and(cat_array == cat_val, dir_array == dir_val)
        trial_idxs = np.where(mask)[0]

        if len(trial_idxs) >= min_trials:
            return np.mean(tensor[trial_idxs, :], axis=0)  # average across trials
        else:
            return np.full((n_timepoints,), np.nan)

    # Condition-specific traces
    low_tl  = safe_condition_mean(1, 1)
    low_tr  = safe_condition_mean(1, 2)
    low_nr  = safe_condition_mean(1, 0)

    high_tl = safe_condition_mean(3, 1)
    high_tr = safe_condition_mean(3, 2)
    high_nr = safe_condition_mean(3, 0)

    probe_tl = safe_condition_mean(2, 1)
    probe_tr = safe_condition_mean(2, 2)
    probe_nr = safe_condition_mean(2, 0)

    if ax is None:
        fig, ax = plt.subplots()

    # Remove the original Axes (passed in via 'ax' from the main batch plotter)
    fig = ax.figure
    ax.remove()

    # Settings
    n_subplots = 10 # (including gap)
    pad_row_idx = 5  # if you still want to pad below one panel
    spacing = 0.05  # this is now relative, not absolute
    extra_pad = 1.4  # add extra height to the lower panels, in units of 1

    height_ratios = [1] * (n_subplots)
    #height_ratios[pad_row_idx] += extra_pad  # e.g., boost row 4 (i.e., space appears after it)
    height_ratios[pad_row_idx] = 0.8
    
    # Create a nested GridSpec for the 9 vertical subplots within this GridSpec cell
    inner_gs = GridSpecFromSubplotSpec(
        n_subplots, 1,
        subplot_spec=ax.get_subplotspec(),  # key: links to the parent GridSpec slot
        height_ratios=height_ratios,
        hspace=spacing
    )

    axs = []
    for i in range(n_subplots):
        if i == pad_row_idx:
            axs.append(None)  # skip the spacer row
        else:
            axs.append(fig.add_subplot(inner_gs[i]))

    axs_clean = [ax for ax in axs if ax is not None]

    axs_clean[0].plot(xvals, smooth(med), color = '#984ea3', label = 'probe', linestyle = '--')
    axs_clean[0].plot(xvals, smooth(low), color = '#377eb8', label = 'low')
    axs_clean[0].plot(xvals, smooth(high), color = '#e41a1c', label = 'high')
    axs_clean[0].set_xticklabels([])

    global_label_group(
        fig, axs_clean[0],
        traces=[smooth(low), smooth(med), smooth(high) ],
        colors=['#377eb8', '#984ea3', '#e41a1c'],
        labels=['Low',  'Probe', 'High']
    )

    axs_clean[1].plot(xvals, smooth(low_nr), color = '#a6cee3', label = 'low (no lick)', linestyle = '--')
    axs_clean[1].plot(xvals, smooth(low_tl), color = '#6baed6', label = 'low (lick left)')
    axs_clean[1].plot(xvals, smooth(low_tr), color = '#08519c', label = 'low (lick right)')
    axs_clean[1].set_xticklabels([])

    global_label_group(
        fig, axs_clean[1],
        traces=[smooth(low_tl), smooth(low_tr), smooth(low_nr)],
        colors=['#6baed6', '#08519c', '#a6cee3'],
        labels=['Low (Lick Left)', 'Low (Lick Right)', 'Low (No Lick)']
    )

    axs_clean[2].plot(xvals, smooth(high_nr), color = '#fcae91', label = 'high (no lick)', linestyle = '--')    
    axs_clean[2].plot(xvals, smooth(high_tl), color = '#fb6a4a', label = 'high (lick left)')
    axs_clean[2].plot(xvals, smooth(high_tr), color = '#a50f15', label = 'high (lick right)')
    axs_clean[2].set_xticklabels([])

    global_label_group(
        fig, axs_clean[2],
        traces=[smooth(high_tl), smooth(high_tr), smooth(high_nr)],
        colors=['#fb6a4a', '#a50f15', '#fcae91'],
        labels=['High (Lick Left)', 'High (Lick Right)', 'High (No Lick)']
    )

    axs_clean[3].plot(xvals, smooth(probe_nr), color = '#cab2d6', label = 'probe (no lick)', linestyle = '--')
    axs_clean[3].plot(xvals, smooth(probe_tl), color = '#bc80bd', label = 'probe (lick left)', linestyle = '--')
    axs_clean[3].plot(xvals, smooth(probe_tr), color = '#6a3d9a', label = 'probe (lick right)', linestyle = '--')
    axs_clean[3].set_xticklabels([])

    global_label_group(
        fig, axs_clean[3],
        traces=[smooth(probe_tl), smooth(probe_tr), smooth(probe_nr)],
        colors=['#bc80bd', '#6a3d9a', '#cab2d6'],
        labels=['Probe (Lick Left)', 'Probe (Lick Right)', 'Probe (No Lick)']
    )

# Now for the d' based on left/right. We're going to use the 100ms before the response.

    low_arr = tensor[trial_events['cat'] == 1,:]
    high_arr = tensor[trial_events['cat'] == 3,:]

    s1 = np.var(low_arr, axis = 0, ddof=1)
    s2 = np.var(high_arr, axis = 0, ddof=1)
    s_pooled = np.sqrt((s1 + s2) / 2)
    s_pooled[s_pooled == 0] = np.nan

    dp = (np.mean(low_arr, axis = 0) - np.mean(high_arr, axis = 0)) / s_pooled

    axs_clean[4].plot(xvals, dp, 'k-', label = 'D-prime (Low - High)')
    axs_clean[4].axhline(0, color = 'k', linewidth = 0.5)
    axs_clean[4].set_xlabel('Time From Stim Onset (s)', fontsize = 9)

    global_label_group(
        fig, axs_clean[4],
        traces=[dp],
        colors=['k'],
        labels=['Cat D-Prime']
    )

    for sub_ax in axs_clean[0:5]:
        sub_ax.axvline(0, color = 'k', linestyle = '--')
        sub_ax.axvline(0.5, color = 'k', linestyle = '--')

        #sub_ax.text(0.01, 0.5, 'Spikes/s', transform=sub_ax.transAxes, fontsize=9, va='center', ha='left')
        sub_ax.set_ylabel('Spikes/s', fontsize = 9)

        sub_ax.spines[['right','top']].set_visible(False)
        sub_ax.set_xticks((-0.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1))
        sub_ax.tick_params(axis = 'x', labelrotation = -45)
        sub_ax.tick_params(labelsize = 6, pad = 2)
        #sub_ax.set_yticks([])

    for sub_ax in axs_clean[0:4]:
        sub_ax.set_ylim(bottom=0)

    axs_clean[4].set_ylabel('D (L-H)', fontsize = 9)

# Now plotting the response-triggered functions.

    steps = np.arange(-0.7, 0.7, bin_size)
    xvals = (steps[0:-1] + steps[1:])/2

    tensor = gen_tensor(steps, (cID,), trial_events['resp_time'], spikes_df)
    tensor = np.squeeze(tensor)

    left = np.mean(tensor[trial_events['dir'] == 1,:],0)
    right = np.mean(tensor[trial_events['dir'] == 2,:],0)

    n_trials, n_timepoints = tensor.shape
    min_trials = 3

    # Condition-specific traces
        
    left_sl  = safe_condition_mean(1, 1)
    left_sh  = safe_condition_mean(3, 1)
    left_sp  = safe_condition_mean(2, 1)

    right_sl = safe_condition_mean(1, 2)
    right_sh = safe_condition_mean(3, 2)
    right_sp = safe_condition_mean(2, 2)

    axs_clean[5].plot(xvals, smooth(left), color = '#7f7f7f', label = 'left') #, linewidth = 3)
    axs_clean[5].plot(xvals, smooth(right), color = '#4d4d4d', label = 'right') #, linewidth = 3)
    axs_clean[5].set_xticklabels([])

    global_label_group(
        fig, axs_clean[5],
        traces=[smooth(left), smooth(right)],
        colors=['#7f7f7f', '#4d4d4d'],
        labels=['Left', 'Right']
    )

    axs_clean[6].plot(xvals, smooth(left_sp), color = '#d17db7', label = 'left (stim probe)', linestyle = '--')
    axs_clean[6].plot(xvals, smooth(left_sl), color = '#377eb8', label = 'left (stim low)')
    axs_clean[6].plot(xvals, smooth(left_sh), color = '#e41a1c', label = 'left (stim high)')
    axs_clean[6].set_xticklabels([])

    global_label_group(
        fig, axs_clean[6],
        traces=[smooth(left_sl), smooth(left_sh), smooth(left_sp)],
        colors=['#377eb8', '#e41a1c', '#d17db7'],
        labels=['Left (Cat Low)', 'Left (Cat High)', 'Left (Cat Probe)']
    )

    axs_clean[7].plot(xvals, smooth(right_sp), color = '#7a4c94', label = 'right (stim probe)', linestyle = '--')
    axs_clean[7].plot(xvals, smooth(right_sl), color = '#08519c', label = 'right (stim low)')
    axs_clean[7].plot(xvals, smooth(right_sh), color = '#a50f15', label = 'right (stim high)')    
    axs_clean[7].set_xticklabels([])
    
    global_label_group(
        fig, axs_clean[7],
        traces=[smooth(right_sl), smooth(right_sh), smooth(right_sp)],
        colors=['#08519c', '#a50f15', '#7a4c94'],
        labels=['Right (Cat Low)', 'Right (Cat High)', 'Right (Cat Probe)']
    )

    # Now for the d' based on left/right. 

    left_arr = tensor[trial_events['dir'] == 1,:]
    right_arr = tensor[trial_events['dir'] == 2,:]

    s1 = np.var(left_arr, axis = 0, ddof=1)
    s2 = np.var(right_arr, axis = 0, ddof=1)
    s_pooled = np.sqrt((s1 + s2) / 2)
    s_pooled[s_pooled == 0] = np.nan

    dp = (np.mean(left_arr, axis = 0) - np.mean(right_arr, axis = 0)) / s_pooled

    axs_clean[8].plot(xvals, dp, 'k-', label = 'D-prime (Left - Right)')
    axs_clean[8].axhline(0, color = 'k', linewidth = 0.5)
    axs_clean[8].set_xlabel('Time From Response (s)', fontsize = 9)

    global_label_group(
        fig, axs_clean[8],
        traces=[dp],
        colors=['k'],
        labels=['Choice D-Prime']
    )

    for sub_ax in axs_clean[5:9]:
        sub_ax.axvline(0, color = 'k', linestyle = '--')

        #sub_ax.text(0.01, 0.5, 'Spikes/s', transform=sub_ax.transAxes, fontsize=9, va='center', ha='left')
        sub_ax.set_ylabel('Spikes/s', fontsize = 9)
        
        sub_ax.spines[['right','top']].set_visible(False)
        sub_ax.set_xticks((-.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7))
        sub_ax.tick_params(axis = 'x', labelrotation = -45)
        sub_ax.tick_params(labelsize = 6, pad = 2)
        #sub_ax.set_yticks([])

    for sub_ax in axs_clean[5:8]:
        sub_ax.set_ylim(bottom=0)

    axs_clean[8].set_ylabel('D (L-R)', fontsize = 9)

    axs_clean[0].text(0.45, 0.85, str(cID), transform=axs[0].transAxes,
         ha='center', va='bottom', fontsize=12, fontweight='bold')

    if show_legend:
        handles, labels = [], []
        for ax_i in axs_clean:
            h, l = ax_i.get_legend_handles_labels()
            handles += h
            labels += l
        return handles, labels
    else:
        return None, None

def make_effect_df(kept_clusters, trigger, spikes_df, trial_events = None, onset_window = [0, 0.1], onset_control_window = [-0.1, 0], 
                    offset_window = [0.5, 0.6], offset_control_window = [0.4, 0.5], pval_cut_off = 0.05, probe_stims = []):
    
    '''
    Creates data frame that summarizes whether there are onset and offset effects for each neuron, and whether those effecs are 
    categorical.

    Args:
    kept_clusters: list of cluster IDs to analyze. Each cluster ID refers to the "cluster" column of the spikes_df dataframe.
    trigger: event times, in seconds. Often stimulus onset times.
    spikes_df: pandas dataframe that has one row for each spike recorded. Columns must include 'cluster', 'time' and 'ch' (channel).
    category: Optional, array with same length as "trigger" that indicates category of trial stim. 1 = low, 2 = probe, 3 = high. 
    onset_window: time (in s) relative to trigger that defines the onset period (default is [0.0 - 0.1s])
    onset_control_window: time (in s) relative to trigger that is compared to onset window to determine onset response (default is [-0.1 - 0.0s])
    offset_window: time (in s) relative to trigger that defines the offset period (default is [0.5 - 0.6s])
    offset_control_window: time (in s) relative to trigger that is compared to offset window to determine offset response (default is [0.4 - 0.5s])
    pval_cut_off: p_value threshold to determine whether an effect is significant (default is 0.01).
    
    Returns:
    effect_df: pandas dataframe that has one row for each neuron. Columns:
        - cluster_id: matches kept_clusters, the cluster ID associated with each cluster.
        - channel: channel associated with the cluster ID
        - onset_effect: whether there is a significant neuronal response to the onset of the event. 
                -1 = suppressed, 0 = no effect, 1 = enhanced
        - offset_effect: whether there is a significant neuronal response to the offset of the event. 
                -1 = suppressed, 0 = no effect, 1 = enhanced
        The remaining columns are returned if "category" is inputted into the function.        
        - onset_categorical: whether there is a significant category effect during the onset period. 
                -1 = low category activity is higher, 0 = no effect, 1 = high category activity is higher      
        - onset_effect: whether there is a significant category effect during the offset period. 
                -1 = low category activity is higher, 0 = no effect, 1 = high category activity is higher   
        - auc: AUC of ROC for choice, measure of choice-modulation  
        - onset_categorical_t: the t-value associated with the categorical onset calculation
    pcnt_stim: percentage of clusters that are responsive to the onset *or* offset of stimulus
    pcnt_cat: percentage of clusters that are categorical either at the onset or offset of the stimulus
    probe_stims: optional list of stim values to use as probe stims -- default is the top half, unique stim values 6:12
    '''

    if len(probe_stims) < 1 and trial_events is not None:
        # By default, take the "top half" of unique stims as probe stims.
        probe_stims = np.unique(trial_events['stim'])[6:12]

    clusters = np.unique(kept_clusters)
    chs = np.array([spikes_df['ch'][spikes_df['cluster'] == c].values[0] for c in clusters])

    if 'interneuron_identity' in spikes_df.columns:
        int_id = np.array([spikes_df['interneuron_identity'][spikes_df['cluster'] == c].values[0] for c in clusters])

    control_response = gen_tensor(onset_control_window, kept_clusters, trigger, spikes_df)
    onset_response = gen_tensor(onset_window, kept_clusters, trigger, spikes_df)

    offset_control_response = gen_tensor(offset_control_window, kept_clusters, trigger, spikes_df)
    offset_response = gen_tensor(offset_window, kept_clusters, trigger, spikes_df)

    auc_tensor = gen_tensor([0, 0.5], kept_clusters, trigger, spikes_df)

    ##

    onset_effect_direction = np.zeros(len(clusters))
    cat_onset_effect_direction = np.zeros(len(clusters))

    cat_onset_d = np.zeros(len(clusters))

    offset_effect_direction = np.zeros(len(clusters))
    cat_offset_effect_direction = np.zeros(len(clusters))

    auc_roc = np.zeros(len(clusters))

    for ii in range(0,len(clusters)):
        
        control = control_response[:,ii]
        resp = onset_response[:,ii]
        val, p_value = ttest_rel(resp, control)
        if p_value < pval_cut_off:
            onset_effect_direction[ii] = np.sign(val)

        control = offset_control_response[:,ii]
        resp = offset_response[:,ii]
        val, p_value = ttest_rel(resp, control)
        if p_value < pval_cut_off:
            offset_effect_direction[ii] = np.sign(val)

        if trial_events is not None:

            category = trial_events['cat']

            low = onset_response[category == 1,ii]
            high = onset_response[category == 3,ii]
            val, p_value = ttest_ind(high, low)

            # Calculating d-prime
            s1 = np.var(low, ddof=1)
            s2 = np.var(high, ddof=1)
            s_pooled = np.sqrt((s1 + s2) / 2)

            if s_pooled == 0:
                cat_onset_d[ii] = np.nan
            else:
                cat_onset_d[ii] = (np.mean(low) - np.mean(high)) / s_pooled

            if p_value < pval_cut_off:
                cat_onset_effect_direction[ii] = np.sign(val)

            low = offset_response[category == 1,ii]
            high = offset_response[category == 3,ii]
            val, p_value = ttest_ind(high, low)
            if p_value < pval_cut_off:
                cat_offset_effect_direction[ii] = np.sign(val)

            if len(np.unique(category) == 3):
                temp = trial_events[np.logical_and(trial_events['resp'] == 1, trial_events['cat'] == 1)].head(1)
                if temp['acc'].values[0] == 1:
                    lowDir = temp['dir'].values[0]
                    highDir = 3 - temp['dir'].values[0]
                else:
                    lowDir = 3 - temp['dir'].values[0]
                    highDir = temp['dir'].values[0]

                auct = []
                for si, s in enumerate(probe_stims):
                    chose_low_probe = auc_tensor[np.logical_and(trial_events['stim'] == s, trial_events['dir'] == lowDir),ii]
                    chose_high_probe = auc_tensor[np.logical_and(trial_events['stim'] == s, trial_events['dir'] == highDir),ii]

                    if np.logical_and(len(chose_high_probe) > 0, len(chose_low_probe) > 0):
                        y_true = np.concatenate([np.zeros(len(chose_high_probe)),np.ones(len(chose_low_probe))])
                        y_scores = np.concatenate([chose_high_probe, chose_low_probe])

                        aucf = roc_auc_score(y_true, y_scores)
                        auct.append(aucf)
                temp = np.mean(auct)
                if temp < 0.5:
                    temp = 1 - temp
                auc_roc[ii] = temp
    temp = {
        'cluster_id': clusters,
        'channel': chs,
        'onset_effect': onset_effect_direction,
        'offset_effect': offset_effect_direction,
        'auc_roc': auc_roc,
    }
    effect_df = pd.DataFrame(temp)

    if 'interneuron_identity' in spikes_df.columns:
        effect_df['interneuron_identity'] =  int_id

    if category is not None:   
        effect_df['onset_categorical'] = cat_onset_effect_direction
        effect_df['offset_categorical'] = cat_offset_effect_direction
        effect_df['onset_categorical_d'] = cat_onset_d

    pcnt_stim = np.mean(np.logical_or(effect_df['onset_effect'] != 0, effect_df['offset_effect'] != 0))
    pcnt_cat = np.mean(np.logical_or(effect_df['onset_categorical'] != 0, effect_df['offset_categorical'] != 0))

    return effect_df, pcnt_stim, pcnt_cat

##

def get_CSI(steps, kept_clusters, trial_events, spikes_df, cat_bounds = (13.3, 14.05)):

    tensor = gen_tensor(steps, kept_clusters, trial_events['stim_time'], spikes_df)

    uf = np.unique(trial_events['stim'])
    freq = trial_events['stim']
    dft = pd.DataFrame(tensor)

    BCD = np.zeros(len(kept_clusters))
    WCD = np.zeros(len(kept_clusters))
    CSI = np.zeros(len(kept_clusters))

    for ni in range(0, len(kept_clusters)):
        BCDt = []
        WCDt = []
        for i, f1 in enumerate(uf):
            for j, f2 in enumerate(uf):
                if i != j:  
                    f1_resp = dft.loc[freq == f1, ni]
                    f2_resp = dft.loc[freq == f2, ni]
                    
                    diff = np.abs(np.mean(f1_resp) - np.mean(f2_resp))

                    if f1 < cat_bounds[0]: # f1 in low category
                        if f2 < cat_bounds[0]:
                            WCDt.append(diff)
                        elif f2 > cat_bounds[1]:
                            BCDt.append(diff)
                    elif f1 > cat_bounds[1]: # f1 in low category
                        if f2 < cat_bounds[0]:
                            BCDt.append(diff)
                        elif f2 > cat_bounds[1]:
                            WCDt.append(diff)

        BCD[ni] = np.nanmean(BCDt)
        WCD[ni] = np.nanmean(WCDt)

        if WCD[ni] > 0:
          CSI[ni] = BCD[ni]/WCD[ni]
        else:
          CSI[ni] = np.nan

    return BCD, WCD, CSI

    ##

def gaussian_cdf(x, mu, sigma, lower, upper):
    return lower + (upper - lower) * norm.cdf(x, loc=mu, scale=sigma)

def neurometric_curve(firing_rates, trial_events, ax = None, plot = True):

    stimulus_freqs = trial_events['stim']
    categories = trial_events['cat']
    direction = trial_events['dir']

    temp = trial_events[np.logical_and(trial_events['resp'] == 1, trial_events['cat'] == 1)].head(1)

    switch = False
    if temp['acc'].values[0] == 1:
        switch = True

    # Define category labels
    high_label = 3
    low_label = 1
    probe_label = 2
    
    # Convert categories to binary (1 = "high", 0 = "low"), ignore "probe" for training
    train_trials = (categories == high_label) | (categories == low_label)
    binary_labels = (categories[train_trials] == high_label).astype(int)
    
    # Standardize firing rates
    scaler = StandardScaler()
    firing_rates_scaled = scaler.fit_transform(firing_rates)
    
    # Train SVM with RBF kernel
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(firing_rates_scaled[train_trials], binary_labels)
    
    # Predict on all trials, including "probe"
    prob_high = svm.predict_proba(firing_rates_scaled)[:, 1]

    svm_dec = prob_high > 0.5
    
    # Get unique stimulus frequencies and compute average predicted probability
    unique_freqs = np.unique(stimulus_freqs)
    prob_high_avg = np.array([
        np.mean(prob_high[stimulus_freqs == f]) for f in unique_freqs
    ])
    dec_high_avg = np.array([
        np.mean(svm_dec[stimulus_freqs == f]) for f in unique_freqs
    ])

    # Compute psychometric curve from accuracy data
    psychometric_avg = np.array([
        np.mean(direction[np.logical_and(stimulus_freqs == f, direction > 0)]) for f in unique_freqs
    ])

    if switch:
        psychometric_avg = 2 - psychometric_avg

    # Fit psychometric curve to data
    x_data = trial_events['stim'][trial_events['dir'] > 0]
    p0 = [13.5, 1, 0, 1]
    bounds = ([min(x_data), 0.1, 0, 0.9],  # Lower bounds
            [max(x_data), 5.0, 0.1, 1.0])  # Upper bounds

    y_data = trial_events['dir'][trial_events['dir'] > 0]
    if switch:
        y_data = 2 - y_data
    params_psych, _ = curve_fit(gaussian_cdf, x_data, y_data, p0 = p0, bounds = bounds)
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit_psych = gaussian_cdf(x_fit, *params_psych)

    y_data = svm_dec[trial_events['dir'] > 0]
    params_neuro, _ = curve_fit(gaussian_cdf, x_data, y_data, p0 = p0, bounds = bounds)
    y_fit_neuro = gaussian_cdf(x_fit, *params_neuro)

    # Plot
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(unique_freqs, dec_high_avg, 'bo')
        ax.plot(unique_freqs, psychometric_avg, 'ro')
        ax.plot(x_fit, y_fit_psych, 'r-', label='Psychometric Curve')
        ax.plot(x_fit, y_fit_neuro, 'b-', label='Neurometric Curve')
        ax.axvline(np.log2(10000))
        ax.axvline(np.log2(17000))

        xts = np.array([6, 8, 12, 16, 24])

        ax.set_xticks(np.log2(xts*1000))
        ax.set_xticklabels(xts)

        ax.spines[['right', 'top']].set_visible(False)

        ax.set_xlabel('Stimulus Frequency (kHz)')
        ax.set_ylabel('P(High Response)')
        ax.set_title('Neurometric and Psychometric Curves')
        ax.legend()


    #filename = 'panels_for_aro_2025/' + 'Fig_Neurometric' + '.eps'
    #plt.savefig(filename, bbox_inches='tight', dpi=300, transparent=False)

    plt.show()
    
    return unique_freqs, prob_high_avg, params_psych[0], params_neuro[0]