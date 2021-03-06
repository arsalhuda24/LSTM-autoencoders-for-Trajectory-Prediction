from torch.utils.data import DataLoader
from traj import TrajectoryDataset, seq_collate


# def data_loader(path):
#     dset = TrajectoryDataset(
#         path,
#         obs_len=8,
#         pred_len=12,
#         skip=1,
#         delim='\t')
#
#     loader = DataLoader(
#         dset,
#         batch_size=64,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=seq_collate)
#     return dset, loader

from torch.utils.data import DataLoader

# from sgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader