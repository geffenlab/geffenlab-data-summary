from pathlib import Path
from shutil import copytree, copy2
import os
from datetime import datetime

import numpy as np
import scipy.io as spio
from scipy import stats
import pandas as pd

from scipy.interpolate import interp1d


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


def gen_dataframe_local(
    behavior_txt_path: Path,
    behavior_mat_path: Path,
    phy_path: Path,
    event_times_path: Path,
    spike_times_sec_path: Path,
    interneuron_search: bool = True,
):
    '''
    This is similar to the original population-analysis gen_dataframe_git(), and should have the same result.
    The difference is this one looks for all data, behavioral and neuronal, on the file system and not via the GitHub API.
    '''

    if "Habituation" in behavior_txt_path.as_posix():
        behavior_df = load_behavior_habit(None, behavior_txt_path)
    else:
        behavior_df = load_behavior(None, behavior_mat_path)

    print(f"Loading neuronal data from {phy_path}")
    spikes_df, stim_events_temp, kept_clusters, cluster_info = load_neuronal(
        None,
        None,
        event_times_path,
        spike_times_sec_path,
        interneuron_search,
        neuronal_loc=phy_path.as_posix()
    )
    events_df, nb_times = load_response_events(None, behavior_txt_path.as_posix(), stim_events_temp)

    if len(events_df) == len(behavior_df):
        trial_events = pd.concat([events_df, behavior_df], axis = 1)
    else:
        if len(events_df) > len(behavior_df):
            events_df = events_df[0:len(behavior_df)]
        else:
            behavior_df = behavior_df[0:len(events_df)]

        trial_events = pd.concat([events_df, behavior_df], axis = 1)

    return trial_events, spikes_df, cluster_info, kept_clusters, nb_times


def load_behavior(ID, matfile: Path):
    '''
    Loads behavior file, returns dataframe with trial-by-trial information

    Args:
    ID: Mouse ID
    matfile: file name of .mat file

    Returns:
    behavior_df: pandas dataframe that has one row for each behavior trial. there are columns associated
        the stimulus frequency, stimulus category, trial accuracy, response direction, and whether the 
        mouse responded at all. 
    '''

    mat_contents = loadmat(matfile)

    stim = np.log2(mat_contents['tt'][:,4])
    cat = mat_contents['tt'][:,0]
    acc = np.array(mat_contents['resp']).squeeze()
    dirT = np.array(mat_contents['lickSide']).squeeze()
    dirT[np.isnan(dirT)] = 0
    resp = np.array([1 if t > 0 else 0 for t in dirT])

    behavior_df = {
        'stim' : stim,
        'cat' : cat,
        'acc' : acc,
        'dir' : dirT,
        'resp': resp,
    }
    behavior_df = pd.DataFrame(behavior_df)

    return behavior_df

def load_behavior_habit(ID, taskfile): #, stim_events_df):
   
    '''
    Loads text file, returns the stimulus frequencies associated with each trial of habituation

    Args:
    ID: Mouse ID
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

    tts[tts == "TT1"] = 6000        # This is hard-coded because the sessions this is meant to analyze use "Extremes-only", so only 6kHz and 28kHz.
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
        'stim' : np.log2(tts),
        'cat' : cat,
        'acc' : acc,
        'dir' : dir[0:len(cat)],
        'resp': resp,
    }

    return pd.DataFrame(behavior_df)


def load_response_events(ID, taskfile, stim_events_df):
    '''
    Loads text file, returns dataframe with stimulus event times and response event times, adjusted to
        align with the stimulus event times from the NIDAQ card.

    Args:
    ID: Mouse ID
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
        response_times = np.array([timestamp[idx + 1] for idx, x in enumerate(tag) if 'STIMON' == x and idx + 2 < len(tag)])        # In the habituation 2 task, there's no "resp_on" after the stimulus onset.
        response_tags = np.array([tag[idx + 1] for idx, x in enumerate(tag) if 'STIMON' == x and idx + 2 < len(tag)])
    else:
        response_times = np.array([timestamp[idx + 1] for idx, x in enumerate(tag) if 'RESPON' == x and idx + 2 < len(tag)])
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

    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
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

## Here we're playing around with getting google sheets data

def get_row_df_from_public_sheet(
        subject,
        date_string,
        date_format_in = '%m%d%Y',
        date_format_sheet = '%Y-%m-%d',
        date_column_name = 'DATE',
        sheet_id = '1_hiEZ6xfpQNN-XLbrtfjkTdmALU4zI21aTrUDhsZxHo',
        gid_mapping_csv = "sheet_gids_per_subject.csv"
):
    """
    Read subject and session metadata from a Google Sheets doc on the web.

    We have two ways to query Google sheets for subject medadata.
    
    The "/export" url returns well-formed CSV data with headers and data cells filled in with the text we expect.

        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_tab_gid}"

    But, this requires us to know the "gid" of the tab within the sheet, that corresponds to a give subject.

    The "gviz/tq" url allows sheet queries and allows us to look up a tab by known subject name rather than obscure gid.

        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tq=SELECT%20*&tqx=out:csv&sheet={sheet_tab_name}"

    But, this returns CSV data with missing values.
    This seems to be caused by incorrect guessing of datatypes for columns that contain mixed types.

    Here we opt for the complete and well-formed data, at the expense of needing to keep track of gids ourselves.
    We can always learn the gid for a sheet tab by scraping it out of the browser url, for example:

        https://docs.google.com/spreadsheets/d/1_hiEZ6xfpQNN-XLbrtfjkTdmALU4zI21aTrUDhsZxHo/edit?gid=1564640587#gid=1564640587

    This url from the browser ends with "gid=1564640587"
    """

    # Look up our known subjects and gids.
    sheet_gids = pd.read_csv(gid_mapping_csv)

    gids = sheet_gids.loc[sheet_gids['subject'] == subject, 'gid']
    if gids.empty:
        print(f"Could not find a gid for subject {subject} in document {gid_mapping_csv}.")
        return None
    else:
        gid = gids.values[0]
        print(f"Found worksheet gid {gid} for subject {subject}.")

    # Construct URL for CSV export of the specific tab
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    try:
        date_obj = datetime.strptime(date_string, date_format_in)
        date_string = date_obj.strftime(date_format_sheet)

        # Read raw data with no header to get full control
        data = pd.read_csv(url, header=1)

        last_index = data[date_column_name].where(data[date_column_name] == date_string).last_valid_index()
        if last_index:
            print(f"Found date {date_string} for subject '{subject}' at index {last_index}.")
            return data.iloc[last_index].to_dict()
        else:
            print(f"No row found for date {date_string} for subject '{subject}'.")
            return None
    
    except Exception as e:
        print(f"Error fetching or parsing sheet: {e}")
        return None

##

def load_neuronal(
    tag,
    sess_date,
    event_times_path = None,
    spike_times_sec_path = None,
    interneuron_search = True,
    neuronal_loc = 'G:' + os.sep + 'Anjali_sorted' + os.sep + 'Preprocessed_data' + os.sep
):
    '''
    Loads neuronal file, returns dataframe with trial-by-trial information

    Args:
    tag: often mouse ID + session type, string to look for in folder
    sess_id: date of session to analyze, in "YYYY-MM-DD_HR-MN-SC" format 
    neuronal_loc: relative path to folder that contains neuronal files
    allow_mua: should clusters identified as multi-unit be included in the "kept-clusters" array?

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

    nl_og = neuronal_loc
    if tag is not None and sess_date is not None:
        # First, we transform the date to the format that KS uses, and pull out the relevant folder. The code
        # currently fails if there are multiple recordings from the same day, this could be fixed in a few
        # different ways.

        date_obj = datetime.strptime(sess_date, '%y%m%d')
        date_temp = date_obj.strftime('%m%d%Y')

        # Now, we find the folder that contains the neural data. First we narrow down the list of potential
        # folders to the ones that contain the right date and tag (often mouse ID). Then, if that returns
        # multiple folders, we ask the user to choose which folder to use.

        res = find_folders(date_temp, neuronal_loc)

        res = [r for r in res if tag in r]
        #res = [r for r in res if 'Single6Tone' in r]

        res = sorted(res, key = len)
        temp = []
        for folder in res:
            if not any(os.path.commonpath([folder, top]) == top for top in temp):
                temp.append(folder)
        res = temp

        if not res:
            print(f"No matching KS folders found for mouse {tag} on {sess_date}, trying different date format:")
            date_temp_2 = date_obj.strftime('%m%d%y')
            res = find_folders(date_temp_2, neuronal_loc)

            res = [r for r in res if tag in r]
            #res = [r for r in res if 'Single6Tone' in r]

            res = sorted(res, key = len)
            temp = []
            for folder in res:
                if not any(os.path.commonpath([folder, top]) == top for top in temp):
                    temp.append(folder)
            res = temp

            if len(res) > 0:
                print("Folder(s) found with new date format.")
            
            if len(res) == 0:
                raise ValueError('No folders found for neuronal analysis.')

        if len(res) == 1:
            neuronal_loc = res[0]

        if len(res) > 1:
            print("Multiple folders found, please select one:")
            for idx, file in enumerate(res,1):
                print(f"{idx}. {os.path.basename(file)}")

            a = True
            while a == True:
                try:
                    choice = int(input("Enter the number of the file you want to select: "))
                    if 1 <= choice <= len(res):
                        neuronal_loc = res[choice - 1]
                        a = False
                    else:
                        print("Invalid selection. Please enter a number from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

    if spike_times_sec_path is None:
        spike_times = sorted(find_files(".npy", "spike_times_sec_adj", neuronal_loc))
        neuronal_loc = os.path.abspath(os.path.join(os.path.dirname(spike_times[0]), '.'))
        spike_times = np.load(spike_times[0])
    else:
        spike_times = np.load(spike_times_sec_path)

    spike_clusters = sorted(find_files(".npy", "spike_clusters", neuronal_loc))
    spike_clusters = np.load(spike_clusters[0])

    spikes_df = {
        'cluster' : np.squeeze(spike_clusters),
        'time' : np.squeeze(spike_times)
    }
    spikes_df = pd.DataFrame(spikes_df)

    # If manual clustering has occurred (I think) then the cluster_info file will exist. This file has all
    # the info that we need, and links each neuron to the appropriate channel. However, if this file does not
    # exist, we can still get most of the information from the raw kilosort labels in the cluster_group file.

    cluster_info_1 = sorted(find_files(".tsv", "cluster_info", neuronal_loc))
    
    if len(cluster_info_1) == 0:
        cluster_group = sorted(find_files(".tsv", "cluster_group", neuronal_loc))
        cluster_group = pd.read_csv(cluster_group[0], sep='\t')
        cluster_group['interneuron_identity'] = np.nan
        kept_clusters = cluster_group.loc[cluster_group['KSLabel'] == 'good', 'cluster_id'].values
        print('Warning: cluster_info file not found, using KSLabels. No channel info available.')

    else:
        cluster_info = pd.read_csv(cluster_info_1[0], sep='\t')
        is_good, cluster_label = pick_good_clusters(cluster_info)
        kept_clusters = cluster_info.loc[is_good, 'cluster_id'].values

        temp = cluster_info[['cluster_id','ch']].copy()
        spikes_df = pd.merge(spikes_df, temp, 'left', left_on = 'cluster', right_on = 'cluster_id')
        spikes_df.drop('cluster_id', axis=1, inplace = True)

        if interneuron_search:
            df = identify_interneurons(tag, sess_date, neuronal_loc = nl_og, pt_cutoff = 13.5) ## This is where we choose the cutoffs for the interneuron identification
            
            temp = df[['Cluster_ID', 'interneuron_identity']]
            spikes_df = pd.merge(spikes_df, temp, 'left', left_on = 'cluster', right_on = 'Cluster_ID')
            spikes_df.drop('Cluster_ID', axis=1, inplace = True)

            cluster_info = pd.merge(cluster_info, temp, 'left', left_on = 'cluster_id', right_on = 'Cluster_ID')
            cluster_info.drop('Cluster_ID', axis=1, inplace = True)

    if event_times_path is None:
        event_times_matches = sorted(find_files(".csv", "events", neuronal_loc))
        event_times_path = event_times_matches[0]
    event_times = np.genfromtxt(event_times_path, delimiter=',')

    stim_events_df = {
        'event' : 1 + 0*event_times,
        'time' : event_times
    }
    stim_events_df = pd.DataFrame(stim_events_df)

    if len(cluster_info_1) == 0:
        return spikes_df, stim_events_df, kept_clusters, cluster_group
    else:
        return spikes_df, stim_events_df, kept_clusters, cluster_info


def pick_good_clusters(cluster_info):
    """Pick 'good' clusters based on Phy curation 'group' column, fall back on Kilosort 'KSLabel' column."""

    # Prefer to use Phy curation 'group' column.
    if 'group' in cluster_info:
        print("Picking good units from info column 'group'.")
        cluster_label = cluster_info['group'].astype(str)
    else:
        print("Info column 'group' not found, picking good units from info column 'KSLabel'.")
        cluster_label = cluster_info['KSLabel'].astype(str)

    is_good = cluster_label == 'good'
    if not np.any(is_good):
        raise ValueError("No 'good' units found.")

    return is_good, cluster_label


def identify_interneurons(
    tag,
    sess_date,
    hw_cutoff = None,
    pt_cutoff = None,
    fr_cutoff = None,
    neuronal_loc = 'G:' + os.sep + 'Anjali_sorted' + os.sep + 'Preprocessed_data' + os.sep,
):
    '''
    This code is from Anjali, it identifies interneurons
    '''
    if sess_date is not None:
        date_obj = datetime.strptime(sess_date, '%y%m%d')
        date_temp = date_obj.strftime('%m%d%Y')

        # Now, we find the folder that contains the neural data. First we narrow down the list of potential
        # folders to the ones that contain the right date and tag (often mouse ID). Then, if that returns
        # multiple folders, we ask the user to choose which folder to use. 

        res = find_folders(date_temp, neuronal_loc)

        res = [r for r in res if tag in r]
        # res = [r for r in res if 'Single6Tone' in r]

        res = sorted(res, key = len)
        temp = []
        for folder in res:
            if not any(os.path.commonpath([folder, top]) == top for top in temp):
                temp.append(folder)
        res = temp

        if not res:
            print(f"No matching KS folders found for interneuron identification for mouse {tag} on {sess_date}, trying different date format:")
            date_temp_2 = date_obj.strftime('%m%d%y')
            res = find_folders(date_temp_2, neuronal_loc)

            res = [r for r in res if tag in r]
            #res = [r for r in res if 'Single6Tone' in r]

            res = sorted(res, key = len)
            temp = []
            for folder in res:
                if not any(os.path.commonpath([folder, top]) == top for top in temp):
                    temp.append(folder)
            res = temp

            if len(res) > 0:
                print("Folder(s) found with new date format.")
            
            if len(res) == 0:
                #print("Still no folders found.")
                raise ValueError('No folders found for interneuron identification.')

        if len(res) == 1:
            neuronal_loc = res[0]

        if len(res) > 1:
            print("Multiple folders found, please select one:")
            for idx, file in enumerate(res,1):
                print(f"{idx}. {os.path.basename(file)}")

            a = True
            while a == True:
                try:
                    choice = int(input("Enter the number of the file you want to select: "))
                    if 1 <= choice <= len(res):
                        neuronal_loc = res[choice - 1]
                        a = False
                    else:
                        print("Invalid selection. Please enter a number from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")     

    mean_waveforms = sorted(find_files(".npy", "mean_waveforms", neuronal_loc))
    if mean_waveforms:
        mean_waveforms = mean_waveforms[0]
        mean_waveforms_np = np.load(mean_waveforms)
    else:
        mean_waveforms_np = load_templates(neuronal_loc)

    print(f"Mean waveforms shape: {mean_waveforms_np.shape}")

    cluster_info = sorted(find_files(".tsv", "cluster_info", neuronal_loc))

    if len(cluster_info) > 0:

        clust_info = pd.read_csv(cluster_info[0], sep='\t')
        is_good, clust_label = pick_good_clusters(clust_info)

        clust_ID_wav = clust_info['cluster_id']
        clust_ch_wav = clust_info['ch']

        if 'firing_rate' in clust_info:
            Firing_rate = clust_info['firing_rate']
        elif 'fr' in clust_info:
            Firing_rate = clust_info['fr']
        else:
            # Fall back to "firing range": https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/firing_range.html
            Firing_rate = clust_info['firing_range']

    # Now, going to just use the "good" ones for everything from this point further:

        clust_ID_wav = clust_ID_wav[is_good]
        Firing_rate = Firing_rate[is_good]
        clust_ch_wav = clust_ch_wav[is_good]
        clust_label = clust_label[is_good]

        peak_trough_val_uV = np.nan + np.zeros(len(clust_ID_wav))
        peak_trough_val_ms = np.nan + np.zeros(len(clust_ID_wav))
        halfwidth_val = np.nan + np.zeros(len(clust_ID_wav))

        for index, i in enumerate(range(len(clust_ID_wav))):

            temp_ID = clust_ID_wav.iloc[i]
            temp_ch = clust_ch_wav.iloc[i]

            waveform_temp = mean_waveforms_np[temp_ID, temp_ch,:]
            baseline_start_temp = np.mean(mean_waveforms_np[temp_ID, temp_ch, 0:5])

            # Pick global minima for the waveform: first get the index then plot the value
            peak_value_temp = np.min(waveform_temp)  # Returns the index of the minimum value
            peak_index_temp = np.where(waveform_temp == peak_value_temp)[0] # This is an array of indices

            if len(peak_index_temp) > 1:
                temp_5 = 100 # just doing something
                #print('Noise Crept In... (re-sort, maybe)')
            else:
                peak_index_temp = peak_index_temp.item()

                # find the max value, which will correspond to trough, after the peak value

                trough_val_temp = np.max(waveform_temp[peak_index_temp + 1:])

                #find index of trough
                trough_index_temp = np.where(waveform_temp == trough_val_temp)[0]
                trough_index_temp  = trough_index_temp[0]

                # peak to trough length of waveform
                peak_trough_val_uV_temp = abs((trough_val_temp) - (peak_value_temp))

                # 1.time between peak and trough
                peak_trough_val_ms_temp = trough_index_temp - peak_index_temp
                
                #Append values
                peak_trough_val_uV[index] = peak_trough_val_uV_temp
                peak_trough_val_ms[index] = peak_trough_val_ms_temp

                # Get Halfwidth now

                val_halfwidth_temp = ((abs(peak_value_temp)) + (abs(baseline_start_temp)))/2
                val_halfwidth_temp = val_halfwidth_temp *(-1)
                #print (val_halfwidth_temp)
                
                waveform_len = len(waveform_temp)
                dt=1 #use dt to convert the x-axis to time form

                time_val= np.arange(0, waveform_len, 1)

                y_constant = val_halfwidth_temp

                # Step 2: Detect the crossings (sign changes) of halfwidth line on the waveform
                # np.sign(waveform - y_constant) gives +1 if waveform > y_constant, -1 if waveform < y_constant, and 0 if equal.
                crossings = np.where(np.diff(np.sign(waveform_temp - y_constant)))[0]  # Indices of sign changes
            
                # Step 3: Interpolate to find the exact x values at the crossings
                crossing_time_values = []
                for idx in crossings:
                    # Linear interpolation between the points surrounding the crossing
                    x0, x1 = time_val[idx], time_val[idx + 1]
                    y0, y1 = waveform_temp[idx], waveform_temp[idx + 1]
                    
                    # Perform linear interpolation to find the exact x value at which the waveform crosses y_constant
                    interpolator = interp1d([y0, y1], [x0, x1], kind='linear')
                    crossing_time = interpolator(y_constant)
                    #print(crossing_time)
                    crossing_time_values.append(crossing_time)
                    #print('/n',crossing_time_values )
                    
                if len(crossing_time_values) > 1:
                    Halfwidth_ms_temp = crossing_time_values[1] - crossing_time_values[0]
                    halfwidth_val[index] = Halfwidth_ms_temp
                else: 
                    #print("Not enough crossing points detected.")
                    Halfwidth_ms_temp = np.nan
                    halfwidth_val[index] = Halfwidth_ms_temp
        
        compINs = False

        interneuron_identity = np.zeros(len(halfwidth_val)) + 1

        if hw_cutoff is not None:
            interneuron_identity = np.logical_and(interneuron_identity, halfwidth_val < hw_cutoff)
            compINs = True
        if pt_cutoff is not None:
            interneuron_identity = np.logical_and(interneuron_identity, peak_trough_val_ms < pt_cutoff)
            compINs = True
        if fr_cutoff is not None:
            interneuron_identity = np.logical_and(interneuron_identity, Firing_rate > fr_cutoff)
            compINs = True

        Clust_waveform_details={
                'Cluster_ID':clust_ID_wav,
                'Cluster_channel': clust_ch_wav,
                'Cluster_label': clust_label,
                'Halfwidth_value': halfwidth_val,
                'peak_to_trough_ms': peak_trough_val_ms,
                'peak_to_trough_uV': peak_trough_val_uV,
                'Firing_rate': Firing_rate,
                }

        Clust_waveform_details = pd.DataFrame(Clust_waveform_details)

        if compINs:
            Clust_waveform_details['interneuron_identity'] = interneuron_identity

        return Clust_waveform_details

    else:
        print('No Cluster Info File Found, No IN Search')
        Clust_waveform_details={
                'interneuron_identity': np.nan + np.zeros(len(spike_times)),
                }
        return pd.DataFrame(Clust_waveform_details)


def load_templates(
    neuronal_loc: str
):
    """Load a matrix of mean waveforms per sorted cluster, with shape (n_clusters, n_channels, n_samples).

    This is similar to loading a mean_waveforms.npy as produced by the Jennifer Colonell ecephys_spike_sorting pipeline.
    Some tools, like SpikeInterface, may provide the same in formation in a sparse form, as templates.npy, and possibly template_ind.npy.
    This reconstitutes a full, non-sparse matrix of shape (n_clusters, n_channels, n_samples), from those files.

    It should be possible to use either mean_waveforms.npy or (templates.npy plus template_ind.npy) interchangeably.
    """
    # Look for mean waveforms per cluster as templates.npy, from eg Spike Interface.
    # Don't confuse this with similar_templates.npy or spike_templates.npy!
    templates_npy = Path(neuronal_loc, "templates.npy")
    print(f"Loading waveform templates from {templates_npy}")
    templates = np.load(templates_npy)

    # templates has shape (num_units, num_samples, max_num_channels), which does not match our desired output shape.
    # In particular, the max_num_channels dimension is sparse, with raw channel indices stored separately in template_ind.npy.
    template_ind_npy = Path(neuronal_loc, "template_ind.npy")
    if template_ind_npy.exists():
        print(f"Loading template channel inds from {template_ind_npy}")
        template_ind = np.load(template_ind_npy)

        # Reconstruct a full matrix with shape (num_units, num_channels, num_samples).
        num_units = templates.shape[0]
        num_samples = templates.shape[1]
        num_channels = template_ind.max() + 1
        full_templates = np.zeros((num_units, num_channels, num_samples))
        for unit_index in range(num_units):
            channel_indices = template_ind[unit_index]
            channel_is_present = np.nonzero(channel_indices >= 0)[0]
            unit_template = templates[unit_index, :, channel_is_present]
            full_templates[unit_index, channel_indices[channel_is_present], :] = unit_template

    else:
        print(f"Transposing dense templates to have shape (n_clusters, n_channels, n_samples)")
        full_templates = np.transpose(templates, (0, 2, 1))

    return full_templates

## Here, we have general functions to locate the files we're interested in.

def combine_neural_data(
    processed_data_path: Path,
    params_py_pattern: str = "exported/phy/**/params.py",
    cluster_info_pattern: str = "curated/**/cluster_info.tsv",
    spike_times_sec_adj_pattern: str = "exported/tprime/**/spike_times_sec_adj.npy",
    event_times_pattern: str = "exported/tprime/**/*nidq.xd_8_3_0.txt"
):
    """Combine data from a few pipeline steps into one "neuronal location" directory."""

    print(f"Gathering session neuronal data within {processed_data_path}")

    neuronal_path = Path(processed_data_path, "neuronal")
    neuronal_path.mkdir(exist_ok=True, parents=True)
    print(f"Gathering session neuronal data to {neuronal_path}")

    # Start with the Phy dir exported from Spike Interface.
    params_py = find_glob(processed_data_path, params_py_pattern)
    print(f"Copying exported Phy files from {params_py.parent}")
    copytree(params_py.parent, neuronal_path, dirs_exist_ok=True)

    # Overwrite any Phy files that were changed during manual curation (optional).
    curated_cluster_info_tsv = find_glob(processed_data_path, cluster_info_pattern, missing_ok=True)
    if curated_cluster_info_tsv is not None:
        print(f"Copying curated Phy files from {curated_cluster_info_tsv.parent}")
        copytree(curated_cluster_info_tsv.parent, neuronal_path, dirs_exist_ok=True)

    # Replace and suppliment with files adjusted by TPrime.
    adjusted_spike_times = find_glob(processed_data_path, spike_times_sec_adj_pattern, missing_ok=True)
    if adjusted_spike_times is not None:
        print(f"Copying adjusted files from {adjusted_spike_times.parent}")
        copytree(adjusted_spike_times.parent, neuronal_path, dirs_exist_ok=True)

    # Copy a list of event times as "events.csv".
    adjusted_events = find_glob(processed_data_path, event_times_pattern)
    events_csv = Path(neuronal_path, "events.csv")
    print(f"Copying adjusted events from {adjusted_events} to {events_csv}")
    copy2(adjusted_events, events_csv)

    return neuronal_path


def find_glob(parent_path: Path, pattern, missing_ok: bool = False) -> Path:
    print(f"Searching: {parent_path}")
    print(f"Using glob pattern: {pattern}")
    matches = list(parent_path.glob(pattern))
    print(f"Found matches: {matches}")
    if matches:
        return matches[0]
    else:
        if missing_ok:
            return None
        else:
            raise ValueError(f"No files matching pattern '{pattern}' found within parent path: {parent_path}")


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

def find_folders(keyword, base_folder):
    '''
    Locates desired folders

    Args:
    keyword: string to look for in folder name
    base_folder: folder to start in (but will walk through subfolders)

    Returns:
    result: list of folders that match query
    '''
    result = []
    for root, dirs, files in os.walk(base_folder):
        # Filter and add only matching directories
        result.extend(
            os.path.normpath(os.path.join(root, dir)) for dir in dirs if keyword in dir
        )
        del dirs[:]  
        break
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

