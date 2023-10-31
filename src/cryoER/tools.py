import torch
import MDAnalysis as mda

def mdau_to_pos_arr(u, frame_cluster = None):
    protein_CA = u.select_atoms("protein and name CA")
    if frame_cluster is None:
        n_frame = len(u.trajectory)
    else:
        n_frame = len(frame_cluster)
    pos = torch.zeros((n_frame, len(protein_CA), 3), dtype=float)
    if frame_cluster is None:
        for i, ts in enumerate(u.trajectory):
            pos[i] = torch.from_numpy(protein_CA.positions)
    else:
        for i, ts in enumerate(u.trajectory[frame_cluster]):
            pos[i] = torch.from_numpy(protein_CA.positions)
    pos -= pos.mean(1).unsqueeze(1)
    return pos