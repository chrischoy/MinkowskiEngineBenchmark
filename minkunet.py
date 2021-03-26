# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
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
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import numpy as np
from urllib.request import urlretrieve
from collections import defaultdict

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import MinkowskiEngine as ME
from models.minkunet import MinkUNet14, MinkUNet18, MinkUNet34, MinkUNet50
from models.common import Timer

# Check if the weights and file exist and download
if not os.path.isfile("1.ply"):
    print("Downloading the room ply file...")
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="1.ply")
parser.add_argument("--use_cpu", action="store_true")


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd


if __name__ == "__main__":
    config = parser.parse_args()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not config.use_cpu) else "cpu"
    )
    print(f"Using {device}")
    # Define a model and load the weights
    model = MinkUNet18(3, 20).to(device)
    model.eval()
    print(model)

    num_conv_layers = defaultdict(int)
    for l in model.modules():
        if isinstance(l, ME.MinkowskiConvolution) or isinstance(
            l, ME.MinkowskiConvolutionTranspose
        ):
            num_conv_layers[l.kernel_generator.kernel_size[0]] += 1
    print(num_conv_layers)

    voxel_size = 0.02
    timer = Timer()
    coords, colors, pcd = load_file(config.file_name)
    batch_sizes = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20]
    if ME.__version__.split(".")[1] == "5":
        # Measure time
        for batch_size in  batch_sizes:
            timer = Timer()
            coordinates = ME.utils.batched_coordinates(
                [coords / voxel_size for i in range(batch_size)], dtype=torch.float32
            )
            features = torch.rand(len(coordinates), 3).float()
            with torch.no_grad():
                for i in range(10):
                    timer.tic()
                    # Feed-forward pass and get the prediction
                    in_field = ME.TensorField(
                        features=features,
                        coordinates=coordinates,
                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                        # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
                        allocator_type=ME.GPUMemoryAllocatorType.PYTORCH,
                        device=device,
                    )
                    # Convert to a sparse tensor
                    sinput = in_field.sparse()
                    # Output sparse tensor
                    soutput = model(sinput)
                    # get the prediction on the input tensor field
                    out_field = soutput.slice(in_field)
                    timer.toc()
            print(batch_size, soutput.shape, timer.min_time)

    elif ME.__version__.split(".")[1] == "4":
        # Measure time
        for batch_size in batch_sizes:
            timer = Timer()
            coordinates = ME.utils.batched_coordinates(
                [coords / voxel_size for i in range(batch_size)]
            )
            features = torch.rand(len(coordinates), 3).float()
            with torch.no_grad():
                for i in range(10):
                    timer.tic()
                    # Feed-forward pass and get the prediction
                    sinput = ME.SparseTensor(features.to(device), coords=coordinates,)
                    # Output sparse tensor
                    soutput = model(sinput)
                    # get the prediction on the input tensor field
                    out_field = soutput.slice(sinput)
                    timer.toc()
            print(batch_size, timer.min_time)
    else:
        raise NotImplementedError
