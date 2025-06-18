import numpy as np
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

def preprocess_data(nb_subjects=10, run_type=2, band='both', crop=(1.0, 2.0), verbose=True):
    """
    Preprocess EEGBCI motor imagery data.

    Parameters
    ----------
    nb_subjects : int
        Number of subjects to load (1-109).
    run_type : int
        Task type: 
        1 = real left/right fist,
        2 = imagined left/right fist,
        3 = real hands/feet,
        4 = imagined hands/feet
    band : str
        Frequency band: 'mu', 'beta', or 'both'.
    crop : tuple of float
        Time window (in seconds) to crop after epoching.
    verbose : bool
        If True, print progress messages.

    Returns
    -------
    labels : np.ndarray
        Array of labels (0 or 1).
    data : dict
        Dictionary with keys 'mu' and/or 'beta', values are 3D EEG arrays (epochs x channels x time).
    info : dict
        Metadata: channel names, sfreq, band info, etc.
    """

    tmin, tmax = -1.0, 4.0
    subjects = list(range(1, nb_subjects + 1))

    run_map = {
        1: ([3, 7, 11], {'T1': 'left', 'T2': 'right'}),
        2: ([4, 8, 12], {'T1': 'left', 'T2': 'right'}),
        3: ([5, 9, 13], {'T1': 'hands', 'T2': 'feet'}),
        4: ([6, 10, 14], {'T1': 'hands', 'T2': 'feet'}),
    }

    if run_type not in run_map:
        raise ValueError("Invalid run_type. Choose 1, 2, 3, or 4.")

    runs, event_name_map = run_map[run_type]

    # Load raw data
    raw_list = []
    for subj in subjects:
        try:
            fnames = eegbci.load_data(subj, runs)
            raws = [read_raw_edf(f, preload=True, verbose='error') for f in fnames]
            raw_list.extend(raws)
            if verbose:
                print(f"Loaded subject {subj}")
        except Exception as e:
            if verbose:
                print(f"Skipping subject {subj} due to error: {e}")

    if not raw_list:
        raise RuntimeError("No data could be loaded.")

    raw = concatenate_raws(raw_list)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.set_eeg_reference(projection=True)

    # Rename annotations for consistent labeling
    raw.annotations.rename(event_name_map)
    events, event_id = events_from_annotations(raw)
    # Remove non-task markers like 'T0' if present
    event_id = {k: v for k, v in event_id.items() if k in event_name_map.values()}


    # Pick EEG channels
    picks = pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads")

    data = {}
    bands = {
        'mu': (8.0, 12.0),
        'beta': (13.0, 30.0)
    }

    for b in bands:
        if band in [b, 'both']:
            raw_filt = raw.copy().filter(*bands[b], fir_design='firwin', skip_by_annotation='edge', verbose='error')
            epochs = Epochs(raw_filt, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                            proj=True, picks=picks, baseline=None, preload=True, verbose='error')
            epochs = epochs.crop(tmin=crop[0], tmax=crop[1])
            data[b] = epochs.get_data()

    # Get labels (already aligned due to Epochs structure)
    labels = epochs.events[:, -1]
    unique_ids = np.unique(labels)
    label_map = {eid: idx for idx, eid in enumerate(unique_ids)}
    labels = np.array([label_map[e] for e in labels])


    info = {
        'channel_names': [raw.ch_names[i] for i in picks],
        'sfreq': raw.info['sfreq'],
        'band': band,
        'event_id': event_id
    }

    return labels, data, info


"""
Example of use:
labels, data, info = preprocess_data(nb_subjects=5, run_type=2, band='both', crop=(1.0, 2.0))
print("Shape of Mu data:", data['mu'].shape)
print("Labels:", np.unique(labels))

"""