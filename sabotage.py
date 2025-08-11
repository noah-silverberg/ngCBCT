import os
import numpy as np
import mat73
import scipy.io as sio  # for saving .mat files

n = 3                          # every n-th ORIGINAL cycle
offset = 0                     # 0..n-1 to shift which original cycle counts as first
modifier = 'SABOTAGE_third'    # name for output folder
# hf_scans = [('08', '01'), ('10', '01'), ('14', '01'), ('15', '01'), ('20', '01'), ('24', '01')]
hf_scans = [('01', '01')]
ff_scans = None
# ff_scans = [('06', '01'), ('16', '01'), ('18', '01'), ('22', '01'), ('26', '01'), ('27', '01')]

mat_projections_dir = os.path.join(r"H:\Public\Noah", "mat")
mat_projections_dir_SABOTAGE = os.path.join(r"H:\Public\Noah", f"mat_{modifier}")
os.makedirs(mat_projections_dir_SABOTAGE, exist_ok=True)

def group_contiguous(idx_1d: np.ndarray):
    """Split a sorted, unique 1D array of integers into contiguous clumps (one clump per cycle that was kept)."""
    if idx_1d.size == 0:
        return []
    if not np.all(idx_1d == np.unique(idx_1d)):
        raise ValueError("Input to group_contiguous must be sorted and unique")
    arr = idx_1d.astype(np.int64)
    split_points = np.where(np.diff(arr) != 1)[0] + 1
    return np.split(arr, split_points)

def reconstruct_all_cycles_from_odd_index(odd_idx: np.ndarray, angles: np.ndarray):
    """
    Reconstruct the full original sequence of cycles by interleaving:
      kept cycles  = contiguous clumps present in odd_idx
      removed ones = the integer gaps between consecutive kept clumps
    Also handles a possible leading gap before the first kept clump and a trailing gap after the last.
    """
    clumps = group_contiguous(odd_idx)
    if not clumps:
        return []

    cycles = []
    # Optional leading gap (if recording doesn't start at clumps[0][0])
    first_start = int(clumps[0][0])
    if first_start > 1:
        leading = np.arange(1, first_start, dtype=np.int64)
        if leading.size > 0:
            cycles.append(leading)

    # Interleave clumps and gaps
    for i in range(len(clumps)):
        cycles.append(clumps[i])  # kept cycle
        if i < len(clumps) - 1:
            gap_start = int(clumps[i][-1]) + 1
            gap_end = int(clumps[i + 1][0])     # exclusive
            gap = np.arange(gap_start, gap_end, dtype=np.int64)
            if gap.size > 0:
                cycles.append(gap)

    # trailing gap (if there are frames after the last kept clump)
    total_last_frame = angles.size
    last_end = int(clumps[-1][-1])
    if last_end < total_last_frame:
        print(f"Adding trailing gap from {last_end + 1} to {total_last_frame}")
        trailing = np.arange(last_end + 1, total_last_frame + 1, dtype=np.int64)
        if trailing.size > 0:
            cycles.append(trailing)

    return cycles

def resample_every_n_from_original(odd_idx: np.ndarray, every_n: int, angles: np.ndarray, start_offset: int = 0):
    """
    Select every_n-th ORIGINAL cycle:
      1) reconstruct the full cycle list (kept clumps + gaps = original cycles),
      2) keep cycles whose (index - start_offset) % every_n == 0,
      3) concatenate selected cycles.
    """
    all_cycles = reconstruct_all_cycles_from_odd_index(odd_idx, angles)
    if not all_cycles:
        return np.array([], dtype=np.int64)

    selected = [cy for i, cy in enumerate(all_cycles) if ((i - start_offset) % every_n) == 0]
    if not selected:
        return np.array([], dtype=np.int64)

    return np.concatenate(selected).astype(np.int64)

hf_scan_files = [f"p{scan[0]}.HF{scan[1]}.HF.mat" for scan in hf_scans] if hf_scans is not None else []
ff_scan_files = [f"p{scan[0]}.FF{scan[1]}.FF.mat" for scan in ff_scans] if ff_scans is not None else []
scan_files = hf_scan_files + ff_scan_files

for file in os.listdir(mat_projections_dir):
    if not file.endswith('.mat'):
        raise ValueError(f"File {file} does not end with .mat")

    # Skip files that are not in either list
    if scan_files and (file not in scan_files):
        print(f"Skipping {file} as it is not in the specified scan lists.")
        continue

    path = os.path.join(mat_projections_dir, file)
    path_sabotage = os.path.join(mat_projections_dir_SABOTAGE, file)

    # read
    mat = mat73.loadmat(path)
    if 'odd_index' not in mat:
        raise KeyError(f"'odd_index' not found in {file}")

    odd_index = mat['odd_index']
    if not isinstance(odd_index, np.ndarray) or odd_index.ndim != 1:
        raise ValueError(f"'odd_index' in {file} is not a 1D numpy array")

    odd_index = np.unique(odd_index.astype(np.int64))  # sorted & unique

    # compute new sampling: every n-th ORIGINAL cycle
    new_index = resample_every_n_from_original(odd_index, n, mat['angles'], offset)

    # write: try to preserve original dict but with updated odd_index; fall back to minimal
    to_save = dict(mat)
    to_save['odd_index'] = new_index
    sio.savemat(path_sabotage, to_save, do_compression=True)
    print(f"Saved {path_sabotage} with {len(new_index)} angles (from {len(odd_index)})")