from pathlib import Path

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import loadFns as lf
from create_cluster_info import create_cluster_info


experimenter = "BH"
subject = "AS20-demo"
date = "03112025"

# step: date format to help with get_row_df_from_public_sheet()
# step: also read gid mappings from an external JSON file
# step: also read subject info from a plain old JSON file

raw_data_root = "/home/ninjaben/codin/geffen-lab-data/raw_data"
raw_data_path = Path(raw_data_root, experimenter, subject, date)

analysis_root = "/home/ninjaben/codin/geffen-lab-data/anaysis"
analysis_path = Path(analysis_root, experimenter, subject, date)

processed_data_root = "/home/ninjaben/codin/geffen-lab-data/processed_data"
processed_data_path = Path(processed_data_root, experimenter, subject, date)

behavior_txt_matches = list(raw_data_path.glob("**/*.txt"))
behavior_txt_path = behavior_txt_matches[0]

behavior_mat_matches = list(raw_data_path.glob("**/*.mat"))
behavior_mat_path = behavior_mat_matches[0]

params_py_matches = list(processed_data_path.glob("**/params.py"))
params_py_path = params_py_matches[0]
phy_path = params_py_path.parent

# by "event times" I think we mean "stim event times".
event_times_matches = list(processed_data_path.glob("tprime/*/*nidq.xd_8_3_0.txt"))
event_times_path = event_times_matches[0]

spike_times_sec_matches = list(processed_data_path.glob("tprime/*/*/spike_times_sec_adj.npy"))
spike_times_sec_path = spike_times_sec_matches[0]

# I think the existing step takes care of finding and associating the files above.
# The stuff below could go in a script that's easy for lab folks to edit.

# This is new from Anjali -- save some of the Phy arrays in a "raw data" pickle.
# I might call this the "sorting info pickle"
# cluster_peak_channel seems not to be present unlsess you save it from Phy.
# Step could take lists of .npy, .tsv/.csv, and catgt .txt to add to the pickle by name.
# Step could do them all dynamically and warn when one is not found.
# Step should save with same subdir structure we've been using.
def raw_data(phy_path):
    spike_times = sorted(lf.find_files(".npy", "spike_times_sec_adj", phy_path))
    spike_times = np.load(spike_times[0])

    cluster_group = sorted(lf.find_files(".tsv", "cluster_group", phy_path))
    cluster_group = pd.read_csv(cluster_group[0], sep='\t')

    cluster_info = sorted(lf.find_files(".tsv", "cluster_info", phy_path))
    cluster_info = pd.read_csv(cluster_info[0], sep='\t')

    cluster_KSLabel = sorted(lf.find_files(".tsv", "cluster_KSLabel", phy_path))
    cluster_KSLabel = pd.read_csv(cluster_KSLabel[0], sep='\t')

    cluster_peak_channel = sorted(lf.find_files(".tsv", "cluster_peak_channel", phy_path))
    if cluster_peak_channel:
        cluster_peak_channel = pd.read_csv(cluster_peak_channel[0], sep='\t')
    else:
        cluster_peak_channel = None
        print(f"didn't find 'cluster_peak_channel'")

    # Why on earth would we not take all the events we have?
    event_times = sorted(lf.find_files(".csv", "events", phy_path))
    if event_times:
        event_times = pd.read_csv(event_times[0], delimiter=',', header=None)
        print(event_times.shape)
        print(event_times.columns)
    else:
        event_times = None
        print(f"didn't find 'events'")

    print('raw_data_events:', event_times)

    # Step should also be able to grab an aligned signal, eg. treadmill.

    spike_clusters = sorted(lf.find_files(".npy", "spike_clusters", phy_path))
    spike_clusters = np.load(spike_clusters[0])

    channel_map = sorted(lf.find_files(".npy", "channel_map", phy_path))
    channel_map = np.load(channel_map[0])

    raw_data_df = {
        'cluster_group': cluster_group,
        'cluster_info': cluster_info,
        'cluster_KSLabel': cluster_KSLabel,
        'cluster_peak_channel': cluster_peak_channel,
        'event_times': event_times,
        'spike_clusters': spike_clusters,
        'spike_times': spike_times,
        'channel_map': channel_map
    }

    # print (cluster_info['depth'])
    # raw_data_pd = pd.DataFrame(raw_data_df)

    # print(cluster_info.columns)
    # print(cluster_info.head())

    # pkl_name = f'{tag}_{sess_date}_raw_data_imec1.pkl'
    # with open(pkl_name, 'wb') as f:
    #     pickle.dump(raw_data_df, f)

    return raw_data_df


raw_data_df = raw_data(phy_path)

session_info = lf.get_row_df_from_public_sheet(subject, date)

# I might call this the "neural-plus-behavioral pickle"
# We could make it optional, in case there's no behavior data.

def make_pickle(trial_events, spikes_df, cluster_info, kept_clusters, raw_data_df, nb_times, session_info):

    # This seems redundant, but we can add it for now.
    uf = np.unique(trial_events['stim'])
    if len(uf) > 8:
        session_type = 'testing_session'
    else:
        session_type = 'training_session'

    # existing step is already taking [start stop step] for the stim and response edges.
    all_clusters = np.unique(spikes_df['cluster'])
    edges_stim = np.arange(-1.0, 1.0, 0.005)
    tensor_stim = lf.gen_tensor(edges_stim, all_clusters, trial_events['stim_time'], spikes_df)
    edges_resp = np.arange(-1.0, 1.0, 0.005)
    tensor_resp = lf.gen_tensor(edges_resp, all_clusters, trial_events['resp_time'], spikes_df)

    event_times = trial_events['stim_time']
    print('event_times:', event_times)

    # We're including the channel_map in both pickles: "raw" and whatever this is
    channel_map = raw_data_df['channel_map']

    # The first several of these come from gen_dataframe_local, above.
    df_dict = {
        "all_session_info": session_info,
        "trial_events": trial_events,
        "spikes_df": spikes_df,
        "cluster_info": cluster_info,
        "kept_clusters": kept_clusters,
        "nb_times": nb_times,
        "tensor_stim": tensor_stim,
        "edges_stim": edges_stim,
        "tensor_resp": tensor_resp,
        "edges_resp": edges_resp,
        "session_type": session_type,
        "uf": uf,
        "channel_map": channel_map
    }

    return df_dict


# This still seems to work, and we'll use the results below.
# by "events" I think we mean "stim events"
trial_events, spikes_df, cluster_info, kept_clusters, nb_times = lf.gen_dataframe_local(
    behavior_txt_path,
    behavior_mat_path,
    phy_path,
    event_times_path,
    spike_times_sec_path,
    interneuron_search=False
)

df_dict = make_pickle(trial_events, spikes_df, cluster_info, kept_clusters, raw_data_df, nb_times, session_info)
print(df_dict)
