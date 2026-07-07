from pathlib import Path
import os
from datetime import datetime
import logging

import numpy as np
import scipy.io as spio
from scipy import stats
import pandas as pd


def gen_tensor(time_edges, neurons, events, spikes_df, nSpikes=False):
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

    mat = np.zeros((len(events), len(neurons), len(stepT)))

    for isp, spi in enumerate(neurons):
        sp = spikes_df.loc[spikes_df['cluster'] == spi, 'time'].values
        for ie, ev in enumerate(events):
            for s, step in enumerate(stepT):
                winTemp = (time_edges[s], time_edges[s+1])
                mat[ie, isp, s] = np.sum(np.logical_and(sp > (ev + winTemp[0]), sp < (ev + winTemp[1])))

    if nSpikes == False:
        mat /= (winTemp[1] - winTemp[0])

    if len(stepT) == 1:
        mat = np.squeeze(mat)

    return mat


def load_behavior(matfile: Path):
    '''
    Loads behavior file, returns dataframe with trial-by-trial information

    Args:
    matfile: file name of .mat file

    Returns:
    behavior_df: pandas dataframe that has one row for each behavior trial. there are columns associated
        the stimulus frequency, stimulus category, trial accuracy, response direction, and whether the 
        mouse responded at all. 
    '''

    mat_contents = loadmat(matfile)

    stim = np.log2(mat_contents['tt'][:, 4])
    cat = mat_contents['tt'][:, 0]
    acc = np.array(mat_contents['resp']).squeeze()
    dirT = np.array(mat_contents['lickSide']).squeeze()
    dirT[np.isnan(dirT)] = 0
    resp = np.array([1 if t > 0 else 0 for t in dirT])

    behavior_df = {
        'stim': stim,
        'cat': cat,
        'acc': acc,
        'dir': dirT,
        'resp': resp,
    }
    behavior_df = pd.DataFrame(behavior_df)

    return behavior_df


def load_behavior_habit(taskfile):  # , stim_events_df):
    '''
    Loads text file, returns the stimulus frequencies associated with each trial of habituation

    Args:
    taskfile: file name of .txt file, habituation
    '''

    with open(taskfile, 'r') as f:
        mystr = f.read()

    lines = [line for line in mystr.split('\n') if line.strip() != '']

    for ii, tx in enumerate(lines):
        if tx.split(' ')[0] == '0001':
            startPt = ii
            break
    else:
        startPt = None

    lines = lines[startPt:]

    # All lines above this point find the text file, load the text file, and then determine how many lines into
    # the text file that the behavior session information begins (which is when the line starts with "0001"). Now,
    # we go through the rest of the lines and separate them out- each line will look like "XXXX YYYYYYY ZZZZ" where
    # XXXX is the trial number, YYYYYYY is the time in microseconds since the arduino code started running, and
    # ZZZZ is the tag that explains what happened at this time point. We extract these values separately for each
    # trial. The timestamps actually reset back to zero if the clock has been running for too long, so we also
    # deploy a fix for that potential case in this section.

    trialN = []
    timestampT = []
    tag = []

    for x in lines:
        tagT = x.split(' ')[2]

        if np.logical_and('HOLDSTART' not in tagT, 'WHEELTURN' not in tagT):
            trialN.append(x.split(' ')[0])
            timestampT.append(x.split(' ')[1])
            tag.append(tagT)

    lst = [eval(i) for i in timestampT]
    timestamp = [float(i)/1000000 for i in lst]

    if np.max(timestamp) > 4200:
        timestamp = np.array([ts if ts > 1000 else ts + 4294.967295 for ts in timestamp])

    response_tags = np.array([tag[idx + 1] for idx, x in enumerate(tag) if 'STIMON' == x and idx + 2 < len(tag)])
    dir = response_tags == 'RIGHT_LICK'
    dir = 1 + dir.astype(int)

    temp = np.array(["TT" in t for t in tag])
    tag = np.array(tag)
    tts = tag[temp == True]
    cat = tts.copy()

    # This is hard-coded because the sessions this is meant to analyze use "Extremes-only", so only 6kHz and 28kHz.
    tts[tts == "TT1"] = 6000
    tts[tts == "TT2"] = 28000
    tts = tts.astype(int)

    cat[cat == "TT1"] = 1
    cat[cat == "TT2"] = 3
    cat = cat.astype(int)

    # This is where I'm going to try to figure out whether H = left or H = right for the mouse.

    rewarded_licks = np.array([tag[idx - 1] for idx, x in enumerate(tag) if 'CORRECT' == x])
    first_rewarded_lick = rewarded_licks[0]

    if first_rewarded_lick == 'LEFT_LICK':
        if cat[0] == 1:
            high_dir = 'RIGHT'
        elif cat[0] == 2:
            high_dir = 'LEFT'
    elif first_rewarded_lick == 'RIGHT_LICK':
        if cat[0] == 1:
            high_dir = 'LEFT'
        elif cat[0] == 2:
            high_dir = 'RIGHT'

    # Now, need to use that information to determine the accuracy of the "licks"

    if high_dir == 'RIGHT':
        cat_t = 1 + (cat - 1)/2
        acc = dir == cat_t
        acc = acc.astype(int)
    elif high_dir == 'LEFT':
        cat_t = 3 - (1 + (cat - 1)/2)
        acc = dir == cat_t
        acc = acc.astype(int)

    resp = np.ones(len(cat))
    resp = resp.astype(int)

    behavior_df = {
        'stim': np.log2(tts),
        'cat': cat,
        'acc': acc,
        'dir': dir[0:len(cat)],
        'resp': resp,
    }

    return pd.DataFrame(behavior_df)


def load_response_events(taskfile, stim_events_df):
    '''
    Loads text file, returns dataframe with stimulus event times and response event times, adjusted to
        align with the stimulus event times from the NIDAQ card.

    Args:
    taskfile: file name of .txt file
    stim_events_df: pandas dataframe that contains column "time", referring to stimulus event times
        measured on the same clock as the spike data

    Returns:
    events_df: pandas dataframe that has one row for each behavior trial. There are columns for the
        stimulus event time associated with that trial and the adjusted response time associated with 
        that trial. If the trial had no response, the response time is "NaN".
    noise_burst_times: numpy array that contains all (adjusted) times that the arduino reported "INCORRECT",
        which we are using as an estimate for when the noise burst is presented.  
    '''

    with open(taskfile, 'r') as f:
        mystr = f.read()

    lines = [line for line in mystr.split('\n') if line.strip() != '']

    for ii, tx in enumerate(lines):
        if tx.split(' ')[0] == '0001':
            startPt = ii
            break
    else:
        startPt = None

    lines = lines[startPt:]

    # All lines above this point find the text file, load the text file, and then determine how many lines into
    # the text file that the behavior session information begins (which is when the line starts with "0001"). Now,
    # we go through the rest of the lines and separate them out- each line will look like "XXXX YYYYYYY ZZZZ" where
    # XXXX is the trial number, YYYYYYY is the time in microseconds since the arduino code started running, and
    # ZZZZ is the tag that explains what happened at this time point. We extract these values separately for each
    # trial. The timestamps actually reset back to zero if the clock has been running for too long, so we also
    # deploy a fix for that potential case in this section.

    trialN = []
    timestampT = []
    tag = []

    for x in lines:
        tagT = x.split(' ')[2]

        if np.logical_and('HOLDSTART' not in tagT, 'WHEELTURN' not in tagT):
            trialN.append(x.split(' ')[0])
            timestampT.append(x.split(' ')[1])
            tag.append(tagT)

    lst = [eval(i) for i in timestampT]
    timestamp = [float(i)/1000000 for i in lst]

    if np.max(timestamp) > 4200:
        timestamp = np.array([ts if ts > 1000 else ts + 4294.967295 for ts in timestamp])

    # The method behind this madness really only makes sense if you know what the text file looks like. For every trial,
    # there is a "RESPON" tag that says when the response window opens. The next tag to appear will either be "RESPONSE_CW",
    # "RESPONSE_CCW" or "RESPOFF_MISS", which indicates the response window has closed without any response recorded. Thus,
    # we calculate the response time by finding the time associated with the tag following each "RESPON". Often, the behavioral
    # session ends during the response window, and therefore we don't want to include trials where there isn't a response only
    # because the session was ended- so we check to make sure that there are future tags. Finally, we replace the time with "NaN"
    # if the associated tag indicates a trial with no response.

    stim_times = np.array([timestamp[idx] for idx, x in enumerate(tag) if 'STIMON' == x])

    if "Habituation_2" in taskfile:
        # In the habituation 2 task, there's no "resp_on" after the stimulus onset.
        response_times = np.array([timestamp[idx + 1]
                                  for idx, x in enumerate(tag) if 'STIMON' == x and idx + 2 < len(tag)])
        response_tags = np.array([tag[idx + 1] for idx, x in enumerate(tag) if 'STIMON' == x and idx + 2 < len(tag)])
    else:
        response_times = np.array([timestamp[idx + 1]
                                  for idx, x in enumerate(tag) if 'RESPON' == x and idx + 2 < len(tag)])
        response_tags = np.array([tag[idx + 1] for idx, x in enumerate(tag) if 'RESPON' == x and idx + 2 < len(tag)])

    response_times[response_tags == 'RESPOFF_MISS'] = 'NaN'

    stim_times = stim_times[0:len(response_times)]

    # To adjust the arduino time stamps to match the recording timestamps, we will perform a linear regression to find
    # the proper offset (which captures differences in recording start time) and slope (which captures if there's any
    # lag that increases or decreases over time). We calculate the residuals for the stimulus event times, which capture
    # trial-to-trial temporal mismatch between the timestamps, and apply those residuals to other timestamps (such as
    # response time) from the same trial in order to do our best to transform the arduino timestamps to be used on the
    # same clock as the spike data.

    X = stim_times
    Y = stim_events_df['time'][0:len(X)]

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    transformed_txt_times = X * slope + intercept
    resid = transformed_txt_times - Y
    adj_response_times = (response_times * slope + intercept) - resid

    events_df = {
        'stim_time': Y,
        'resp_time': adj_response_times
    }

    # Here, we simply take the times that are associated with the arduino reporting an incorrect decision. In the behavioral
    # code, when this information is read by the MATLAB loop, it triggers a noise burst being played through the soundcard.
    # Although there are delays associated with both the information being transmitted to MATLAB and the sound being presented
    # through the speaker, these times serve as estimates for the noise burst timings. The arduino timestamps are adjusted to
    # the spike time clock using the slope and intercept of the linear regression performed earlier.

    if "Habituation_2" not in taskfile:
        noise_burst_times = np.array([timestamp[idx] for idx, x in enumerate(tag) if 'INCORRECT' == x])
        noise_burst_times = noise_burst_times * slope + intercept
    else:
        noise_burst_times = np.array(())

    return pd.DataFrame(events_df), noise_burst_times

# Here we're playing around with getting google sheets data


def get_row_dict_from_public_sheet(
    date_obj: datetime,
    sheet_id: str,
    tab_gid: str,
    sheet_date_format: str = '%Y-%m-%d',
    sheet_date_column_name: str = 'DATE',
) -> dict[str, str]:
    """
    Read subject and session metadata from a Google Sheets doc on the web.

    We have two ways to query Google sheets for subject medadata:

    The "/export" url returns well-formed CSV data with headers and data cells filled in with the text we expect.

        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_tab_gid}"

    But, this requires us to know the "gid" of the tab within the sheet, for each subject.
    The "gid" is an arbitrary id, not a convenient tab/subject name.

    The "gviz/tq" url allows SQL-like queries and allows us to look up a tab by known subject name rather than obscure gid.

        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tq=SELECT%20*&tqx=out:csv&sheet={sheet_tab_name}"

    But, this returns CSV data with missing values.
    This seems to be caused by incorrect guessing of datatypes for columns that contain mixed types.

    Here we opt for the complete and well-formed data, at the expense of needing to keep track of gids ourselves.
    We can always learn the gid for a sheet tab by scraping it out of the browser url, for example:

        https://docs.google.com/spreadsheets/d/1_hiEZ6xfpQNN-XLbrtfjkTdmALU4zI21aTrUDhsZxHo/edit?gid=1564640587#gid=1564640587

    This url from the browser ends with "gid=1564640587"
    """

    # Construct a URL for CSV export of one tab within a public sheet.
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={tab_gid}"
    logging.info(f"Looking up session info Sheet from: {url}")

    try:
        # Read raw data and expect a header row.
        data = pd.read_csv(url, header=1)

        # Find the row whose date matches the give date string.
        date_string = date_obj.strftime(sheet_date_format)
        last_index = data[sheet_date_column_name].where(data[sheet_date_column_name] == date_string).last_valid_index()
        if last_index:
            logging.info(f"Found date {date_string} at row at index {last_index}.")
            return data.iloc[last_index].to_dict()
        else:
            logging.warning(f"No row found for date {date_string}.")
            return {}

    except Exception as e:
        logging.warning(f"Unable to get session info from Sheet: {e}")
        return {}

##


def load_neuronal(
    stim_times_path: Path,
    spike_times_sec_path: Path,
    phy_path: Path
):
    '''
    Loads neuronal file, returns dataframe with trial-by-trial information.

    Args:
    stim_times_path: path to a .txt file with eg stim event times, to align on
    spike_times_sec_path: path to a .npy file with spike times in seconds
    phy_path: path to directory of Phy outputs

    Returns:
    spikes_df: pandas dataframe that has one row for each spike recorded. The first column is the ID
        associated with the cluster that spiked, while the second column is the time (in seconds) when
        the spike was recorded
    stim_events_df: pandas dataframe that has one row for each stimulus event trigger recorded. The
        first column is the event ID associate with the stimulus, which is "1". The second column is 
        the time (in seconds) when the event was recorded
    kept_clusters: numpy array that contains the IDs associated with the clusters that should be
        analyzed. This is based on the choice of whether to "allow_mua".
    '''

    spike_times = np.load(spike_times_sec_path)
    spike_clusters = sorted(find_files(".npy", "spike_clusters", phy_path))
    spike_clusters = np.load(spike_clusters[0])

    spikes_df = {
        'cluster': np.squeeze(spike_clusters),
        'time': np.squeeze(spike_times)
    }
    spikes_df = pd.DataFrame(spikes_df)

    # If manual clustering has occurred (I think) then the cluster_info file will exist. This file has all
    # the info that we need, and links each neuron to the appropriate channel. However, if this file does not
    # exist, we can still get most of the information from the raw kilosort labels in the cluster_group file.

    cluster_info_path = sorted(find_files(".tsv", "cluster_info", phy_path))
    if len(cluster_info_path) == 0:
        cluster_group = sorted(find_files(".tsv", "cluster_group", phy_path))
        cluster_group = pd.read_csv(cluster_group[0], sep='\t')
        kept_clusters = cluster_group.loc[cluster_group['KSLabel'] == 'good', 'cluster_id'].values
        logging.warning('cluster_info file not found, using KSLabels. No channel info available.')

    else:
        cluster_info = pd.read_csv(cluster_info_path[0], sep='\t')
        is_good, cluster_label = pick_good_clusters(cluster_info)
        kept_clusters = cluster_info.loc[is_good, 'cluster_id'].values

        temp = cluster_info[['cluster_id', 'ch']].copy()
        spikes_df = pd.merge(spikes_df, temp, 'left', left_on='cluster', right_on='cluster_id')
        spikes_df.drop('cluster_id', axis=1, inplace=True)

    if stim_times_path is None:
        event_times_matches = sorted(find_files(".csv", "events", phy_path))
        stim_times_path = event_times_matches[0]
    event_times = np.genfromtxt(stim_times_path, delimiter=',')

    stim_events_df = {
        'event': 1 + 0*event_times,
        'time': event_times
    }
    stim_events_df = pd.DataFrame(stim_events_df)

    if len(cluster_info_path) == 0:
        return spikes_df, stim_events_df, kept_clusters, cluster_group
    else:
        return spikes_df, stim_events_df, kept_clusters, cluster_info


def pick_good_clusters(cluster_info):
    """Pick 'good' clusters based on Phy curation 'group' column, fall back on Kilosort 'KSLabel' column."""

    # Prefer to use Phy curation 'group' column.
    if 'group' in cluster_info:
        logging.info("Picking good units from info column 'group'.")
        cluster_label = cluster_info['group'].astype(str)
    else:
        logging.warning("Info column 'group' not found, picking good units from info column 'KSLabel'.")
        cluster_label = cluster_info['KSLabel'].astype(str)

    is_good = cluster_label == 'good'
    if not np.any(is_good):
        raise ValueError("No 'good' units found.")

    return is_good, cluster_label

# Here, we have general functions to locate the files we're interested in.


def find_files(fileend, keyword, folder):
    '''
    Locates desired files

    Args:
    fileend: type of file required (for example, '.csv')
    keyword: string to look for in file name
    folder: folder to start in (but will walk through subfolders)

    Returns:
    result: list of files that match query
    '''
    result = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(fileend) and keyword in file:
                result.append(os.path.normpath(os.path.join(root, file)))
    return result


def _todict(matobj):
    '''
    Thanks to 'mergen', from StackOverflow: a recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _check_keys(dict):
    '''
    Thanks to 'mergen', from StackOverflow: changes mat-objects to nested dicts
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def loadmat(filename):
    '''
    Thanks to 'mergen', from StackOverflow: cures all entries from spio.loadmat that aren't changed
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
