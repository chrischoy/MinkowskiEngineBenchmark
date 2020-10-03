import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from models.resunet import ResUNetBN2C
from models.common import Timer

import torch
import MinkowskiEngine as ME

if not os.path.isfile("redkitchen-20.ply"):
    print("Downloading a mesh...")
    urlretrieve(
        "https://node1.chrischoy.org/data/publications/fcgf/redkitchen-20.ply",
        "redkitchen-20.ply",
    )


def benchmark(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
    model.eval()
    model = model.to(device)

    pcd = o3d.io.read_point_cloud(config.input)

    if ME.__version__.split(".")[1] == "5":
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
          vox_coords = torch.from_numpy(np.array(pcd.points)) / config.voxel_size
          coords = ME.utils.batched_coordinates(
              [vox_coords for i in range(batch_size)]
          )
          feats = torch.from_numpy(np.ones((len(coords), 1))).float()

          with torch.no_grad():
              t = Timer()
              for i in range(10):
                  # initialization time includes copy to GPU
                  t.tic()
                  sinput = ME.SparseTensor(
                      feats,
                      coords,
                      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                      device=device,
                  )
                  model(sinput)
                  t.toc()
              print(f"{batch_size}\t{len(sinput)}\t{t.min_time}")

    elif ME.__version__.split(".")[1] == "4":
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
          vox_coords = torch.from_numpy(np.array(pcd.points)) / config.voxel_size
          coords = ME.utils.batched_coordinates(
              [vox_coords for i in range(batch_size)]
          )
          feats = torch.from_numpy(np.ones((len(coords), 1))).float()

          with torch.no_grad():
              t = Timer()
              for i in range(10):
                  # initialization time includes copy to GPU
                  t.tic()
                  sinput = ME.SparseTensor(feats, coords,).to(device)
                  model(sinput)
                  t.toc()
              print(f"{batch_size}\t{len(sinput)}\t{t.min_time}")

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="redkitchen-20.ply",
        type=str,
        help="path to a pointcloud file",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.05,
        type=float,
        help="voxel size to preprocess point cloud",
    )

    config = parser.parse_args()
    benchmark(config)
