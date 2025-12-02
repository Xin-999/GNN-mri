#%%
import argparse
from sklearn.covariance import ledoit_wolf
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
import math
import os
import pickle
import dill
from pathlib import Path

#%%
# Load data from .npy files
def _load_cognitive_scores(
    score_path: Path,
    key_values,
    key_name: str = 'run_id',
    score_column: str = 'cognitive_score',
    create_template: bool = False,
):
    """
    Loads cognitive regression scores from a CSV file.
    If the file does not exist, a template is created for the user to populate.
    """
    score_path = Path(score_path)
    if not score_path.exists():
        if create_template:
            unique_keys = pd.unique(pd.Series(key_values).astype(str))
            template = pd.DataFrame({key_name: unique_keys, score_column: [np.nan] * len(unique_keys)})
            template.to_csv(score_path, index=False)
            raise FileNotFoundError(
                f"Cognitive score file not found at {score_path}. "
                f"A template with columns '{key_name}' and '{score_column}' has been created—please fill it in."
            )
        raise FileNotFoundError(f"Cognitive score file not found at {score_path}.")

    df = pd.read_csv(score_path)
    required_cols = {key_name, score_column}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Cognitive score file must contain columns: {required_cols}")

    df[key_name] = df[key_name].astype(str)
    score_map = df.set_index(key_name)[score_column].to_dict()
    key_values = [str(value) for value in key_values]
    missing = [rid for rid in pd.unique(pd.Series(key_values)) if rid not in score_map]
    if missing:
        raise ValueError(
            f"Cognitive score file at {score_path} is missing scores for: {missing}"
        )

    scores = []
    for rid in key_values:
        val = score_map[rid]
        if pd.isna(val):
            raise ValueError(f"Cognitive score for {key_name} '{rid}' is NaN. Please provide a numeric value.")
        scores.append(float(val))
    return np.array(scores, dtype=float)


def load_data(
    data_dir='./data/all_shen_roi_ts',
    expected_nrois=268,
    # score_csv='./data/cogn_pc_scores.csv',
    score_csv='./data/ListSort_AgeAdj.csv',
    score_column='ListSort_AgeAdj',
    score_key='subject',
    create_score_template=False,
):
    """
    Loads Shen ROI time-series (tab-delimited .txt) files and user-provided cognitive scores.
    Performs initial data cleaning by removing runs with missing ROIs or all-zero time points.

    Args:
        data_dir (str): Directory containing the Shen ROI time-series files.
        expected_nrois (int): Expected number of ROIs per time point.
        score_csv (str): Path to CSV containing cognitive scores.
        score_column (str): Column in the CSV to use as the cognitive target.
        score_key (str): Identifier type to match scores ('run_id' or 'subject').
        create_score_template (bool): If True and the CSV is missing, create a template to fill in.

    Returns:
        tuple: (data, cognitive_scores)
            data (list[np.ndarray]): Cleaned list of time-series arrays per run.
            cognitive_scores (np.ndarray): Continuous regression targets provided by the user.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Time-series directory not found: {data_path}")

    txt_files = sorted(data_path.glob('*.txt'))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")

    data = []
    subjects = []
    runs = []
    shapes = []

    run_ids = []

    for path in txt_files:
        ts = np.loadtxt(path)
        if ts.ndim == 1:
            ts = ts[:, np.newaxis]

        parts = path.stem.split('_')
        subject_id = parts[0] if len(parts) > 0 else 'UNKNOWN'
        run_name = parts[1] if len(parts) > 1 else 'UNKNOWN'

        data.append(ts)
        subjects.append(subject_id)
        runs.append(run_name)
        shapes.append(ts.shape)
        run_ids.append(path.stem)

    to_remove = []
    for i, ts in enumerate(data):
        if ts.shape[1] == expected_nrois:
            zero_cols = np.all((ts == 0), axis=0)
            if np.any(zero_cols):
                to_remove.append(i)
                print(f"Removing {txt_files[i].name} due to zero-value ROI columns.")
        else:
            zero_rows = np.all((ts == 0), axis=1)
            if np.any(zero_rows):
                to_remove.append(i)
                print(f"Removing {txt_files[i].name} due to zero-value time points (shape mismatch).")

    if to_remove:
        keep_idx = [idx for idx in range(len(data)) if idx not in to_remove]
        data = [data[idx] for idx in keep_idx]
        subjects = [subjects[idx] for idx in keep_idx]
        runs = [runs[idx] for idx in keep_idx]
        shapes = [shapes[idx] for idx in keep_idx]
        run_ids = [run_ids[idx] for idx in keep_idx]

    if score_key not in {'run_id', 'subject'}:
        raise ValueError("score_key must be either 'run_id' or 'subject'.")

    key_values = run_ids if score_key == 'run_id' else subjects

    cognitive_scores = _load_cognitive_scores(
        Path(score_csv),
        key_values,
        key_name='run_id' if score_key == 'run_id' else 'Subject',
        score_column=score_column,
        create_template=create_score_template,
    )

    tp_lengths = [shape[0] for shape in shapes]
    if tp_lengths:
        print(f"Loaded {len(data)} runs from {data_path}.")
        print(f"ROI count per run: {shapes[0][1]} (expected {expected_nrois}).")
        print(f"Time points range from {min(tp_lengths)} to {max(tp_lengths)}.")
        print(f"Cognitive score range: {cognitive_scores.min():.4f} to {cognitive_scores.max():.4f}.")

    return data, cognitive_scores

# Compute the correlation & do thresholding
## Function for Converting Covariance to Correlation
def cov2corr(covariance):
    """
    Converts a covariance matrix to a correlation matrix.
    """
    v = np.sqrt(np.diag(covariance)) # Standard deviations
    outer_v = np.outer(v, v)        # Outer product of standard deviations
    correlation = covariance / outer_v # Correlation formula
    correlation[covariance == 0] = 0 # Handle cases where covariance is zero
    return correlation

## Function for Proportional Thresholding
def threshold_proportional(W, p, copy=True):
    """
    Thresholds the connectivity matrix by preserving a proportion 'p' of the strongest weights.
    All other weights and diagonal elements are set to 0.
    
    Args:
        W (np.ndarray): Weighted or binary connectivity matrix.
        p (float): Proportion of weights to preserve (0 < p < 1).
        copy (bool): If True, a copy of W is made to avoid modifying in place.
    
    Returns:
        np.ndarray: Thresholded connectivity matrix.
    """
    assert 0 < p < 1, "Proportion p must be between 0 and 1."
    if copy:
        W = W.copy()
    n = len(W)                        # number of nodes
    np.fill_diagonal(W, 0)            # clear diagonal (self-connections)
    
    # Determine if matrix is symmetric to handle upper/lower triangle efficiently
    if np.all(W == W.T):                # if symmetric matrix
        W[np.tril_indices(n)] = 0        # set lower triangle to 0 to avoid double counting
        ud = 2                        # factor for symmetric matrix (links counted twice)
    else:
        ud = 1
    
    ind = np.where(W)                    # find all non-zero link indices
    I = np.argsort(W[ind])[::-1]        # sort indices by magnitude in descending order
    
    # Number of links to be preserved
    en = round((n * n - n) * p / ud)
    
    # Set weights of weaker links to 0
    W[(ind[0][I][en:], ind[1][I][en:])] = 0    # apply threshold
    
    if ud == 2:                        # if symmetric matrix
        W[:, :] = W + W.T                        # reconstruct symmetry
    
    # Ensure the highest correlation coefficient is 1 (or close to it)
    # This line seems to be a specific heuristic, might need review based on data characteristics.
    W[W > 0.9999] = 1                          
    return W

def extract_ldw_corr(data, wSize, shift):
    """
    Extracts Ledoit-Wolf optimal shrinkage covariance, converts to correlation,
    and applies proportional thresholding using a sliding window approach.
    
    Args:
        data (list): List of subject time-series data (each element is a np.ndarray).
        wSize (int): Sliding window size.
        shift (int): Shift (step size) for the sliding window.
    
    Returns:
        tuple: (node_feats, LDW_adj_mat, nWin)
            node_feats (list): List of lists, where each inner list contains
                               correlation matrices (node features) for each window of a subject.
            LDW_adj_mat (list): List of lists, where each inner list contains
                               thresholded adjacency matrices for each window of a subject.
            nWin (list): List of number of windows for each subject.
    """
    nSub = len(data)
    nROI = data[0].shape[1] # Number of ROIs
    tpLen = [item.shape[0] for item in data] # Time points length for each subject
    
    overlap = wSize - shift # Overlap between consecutive windows
    # Calculate number of windows for each subject
    nWin = [int((l - overlap) / (wSize - overlap)) for l in tpLen]
    
    node_feats = [] # Container for node features (correlation matrices)
    LDW_adj_mat = [] # Container for adjacency matrices

    for sub in tqdm(range(len(data)), desc="Processing subjects"):    # For each subject
        corr_mat_subject = [] # Correlation matrices for current subject
        adj_mat_subject = [] # Adjacency matrices for current subject
        
        for wi in range(nWin[sub]): # Iterate through windows for the current subject
            st = wi * (wSize - overlap) # Start index of the window
            en = st + wSize             # End index of the window
            w_data = data[sub][st:en, :] # Extract data for the current window
            
            # Apply Ledoit-Wolf covariance estimation
            covariance_matrix, _ = ledoit_wolf(w_data, assume_centered=False)
            
            # Convert covariance to correlation
            corr_neg = cov2corr(covariance_matrix)
            corr = np.abs(corr_neg) # Take absolute value for thresholding
            corr_mat_subject.append(corr_neg) # Store original correlation matrix as node features

            # Apply proportional thresholding to create adjacency matrix
            th_corr = threshold_proportional(corr, 0.40) # Keep top 40% coefficients
            
            # Fill diagonal with ones to avoid zero-degree nodes (common in graph analysis)
            np.fill_diagonal(th_corr, 1)
            adj_mat_subject.append(th_corr) # Store thresholded adjacency matrix

            # Assertions for data integrity (optional, but good for debugging)
            assert not np.all(np.all((th_corr == 0), axis=1)), 'adjacency matrix contains rows of all zeros'
            assert not np.all(np.all((th_corr == 0), axis=0)), 'adjacency matrix contains columns of all zeros'
            assert np.all(th_corr >= 0), 'adjacency matrix contains negative values'
        
        node_feats.append(corr_mat_subject)
        LDW_adj_mat.append(adj_mat_subject)
        
    return node_feats, LDW_adj_mat, nWin

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Ledoit-Wolf graphs for Movie HCP time-series.")
    parser.add_argument(
        '--score-csv',
        default='./data/cogn_pc_scores.csv',
        help="Path to CSV containing cognitive targets (default: ./data/cogn_pc_scores.csv)."
    )
    parser.add_argument(
        '--score-column',
        default='cogn_PC1',
        help="Column in the score CSV to use as the cognitive target (default: cogn_PC1)."
    )
    parser.add_argument(
        '--score-key',
        choices=['run_id', 'subject'],
        default='subject',
        help="Identifier used to match runs with cognitive scores (default: subject)."
    )
    parser.add_argument(
        '--score-template',
        action='store_true',
        help="If set, create a template CSV with the requested identifiers when the score file is missing."
    )
    args = parser.parse_args()

    data_timeseries_path = './data/all_shen_roi_ts'
    if not os.path.isdir(data_timeseries_path):
        raise FileNotFoundError(f"Expected directory with Shen ROI time-series: {data_timeseries_path}")

    print("Loading raw fMRI time-series data...")
    data, cognitive_scores = load_data(
        data_timeseries_path,
        score_csv=args.score_csv,
        score_column=args.score_column,
        score_key=args.score_key,
        create_score_template=args.score_template,
    )
    # Ensure data is a list of numpy arrays, as expected by extract_ldw_corr
    data = [np.array(item) for item in data]
    print(f"Loaded data from {len(data)} runs.")

    # Sliding window parameters
    wSize = 20  # Window size (number of time points in each window)
    shift = 10  # Shift size (number of time points to move for the next window)

    print(f"Extracting Ledoit-Wolf correlations and adjacency matrices with window size {wSize} and shift {shift}...")
    node_feats, adj_mats, nWin = extract_ldw_corr(data, wSize, shift)
    print("Extraction complete.")

    # Prepare data for saving
    LDW_data = {}
    LDW_data['adj_mat'] = adj_mats
    LDW_data['node_feat'] = node_feats
    LDW_data['cognitive_scores'] = cognitive_scores

    win_info = {}
    win_info['wSize'] = wSize
    win_info['shift'] = shift
    win_info['nWin'] = nWin

    # Define path to save processed data
    saveTo = './data/ldw_data/'
    os.makedirs(saveTo, exist_ok=True) # Create directory if it doesn't exist
    
    print(f"Saving processed data to {saveTo}...")
    # Save the processed data using pickle
    with open(os.path.join(saveTo, 'LDW_movie-hcp_data.pkl'), 'wb') as f:
        pickle.dump(LDW_data, f, protocol=4) # protocol=4 for compatibility
        
    # Save window information
    with open(os.path.join(saveTo, 'win_info.pkl'), 'wb') as f:
        pickle.dump(win_info, f, protocol=4)
    print("Processed data saved successfully.")
