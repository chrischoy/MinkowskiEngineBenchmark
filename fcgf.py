# Based on https://github.com/chrischoy/FCGF/blob/master/demo.py
#
# MIT License
#
# Copyright (c) 2019 Chris Choy (chrischoy@ai.stanford.edu), Jaesik Park (jaesik.park@postech.ac.kr)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from models.resunet import ResUNetBN2C
from models.common import Timer
from collections import defaultdict

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

    num_conv_layers = defaultdict(int)
    for l in model.modules():
        if isinstance(l, ME.MinkowskiConvolution) or isinstance(
            l, ME.MinkowskiConvolutionTranspose
        ):
            num_conv_layers[l.kernel_generator.kernel_size[0]] += 1
    print(num_conv_layers)

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
