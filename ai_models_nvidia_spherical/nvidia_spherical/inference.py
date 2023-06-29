import pathlib
import urllib
from typing import Optional, Tuple
import sys
import datetime

import torch
import numpy as np
import contextlib


class Inference(torch.nn.Module):
    n_history = 0

    def __init__(
        self,
        model,
        channels,
        center: np.array,
        scale: np.array,
        time_step=datetime.timedelta(hours=6),
    ):
        super().__init__()
        self.time_dependent = False
        self.model = model
        self.channels = channels
        self.graph = None  # Cuda graph
        self.iteration = 0
        self.out = None
        self.time_step = time_step

        center = torch.from_numpy(np.squeeze(center)).float()
        scale = torch.from_numpy(np.squeeze(scale)).float()

        self.register_buffer("scale_org", scale)
        self.register_buffer("center_org", center)

        self.register_buffer("scale", scale[self.channels, None, None])
        self.register_buffer("center", center[self.channels, None, None])

    def normalize(self, x):
        return (x - self.center_org[None, :, None, None]) / self.scale_org[
            None, :, None, None
        ]

    def run_steps(
        self, x, n, normalize=True, cuda_graphs=False, autocast_fp16=False, time=None
    ):
        for _, data, _ in self.run_steps_with_restart(
            x, n, normalize, cuda_graphs, autocast_fp16, time
        ):
            yield data

    def run_steps_with_restart(
        self, x, n, normalize=True, cuda_graphs=False, autocast_fp16=False, time=None
    ):
        """Yield (time, unnormalized data, restart) tuples

        restart = (time, unnormalized data)
        """
        if self.time_dependent and not time:
            raise ValueError("Time dependent models require ``time``.")

        if self.time_dependent and cuda_graphs:
            raise NotImplementedError(
                "Time-dependent models will give incorrect results with cuda-graphs."
            )

        if self.n_history > 0:
            raise NotImplementedError(f"{self.n_history=}. History is not supported.")

        time = time or datetime.datetime(1900, 1, 1)

        with torch.no_grad():
            # drop all but the last time point
            # remove channels

            _, n_time_levels, n_channels, _, _ = x.shape
            assert n_time_levels == self.n_history + 1

            if normalize:
                x = self.normalize(x)

            x = x[:, -1, self.channels].clone()

            # yield initial time for convenience
            restart = dict(
                x=x[:, None],
                normalize=False,
                cuda_graphs=cuda_graphs,
                autocast_fp16=autocast_fp16,
                time=time,
            )
            yield time, self.scale * x + self.center, restart

            for i in range(n):
                restart = dict(
                    x=x[:, None],
                    normalize=False,
                    cuda_graphs=cuda_graphs,
                    autocast_fp16=autocast_fp16,
                    time=time,
                )
                if not cuda_graphs:
                    with (
                        torch.cuda.amp.autocast()
                        if autocast_fp16
                        else contextlib.nullcontext()
                    ):
                        if self.time_dependent:
                            # TODO address this proposal: https://gitlab-master.nvidia.com/earth-2/fcn-mip/-/issues/25
                            y = self.model(x, time)
                        else:
                            y = self.model(x)

                    self.out = self.scale * y + self.center
                    x = y
                # CUDA graphs
                else:
                    if self.iteration < 11:  # For DDP if needed (idk)
                        warmup_stream = torch.cuda.Stream()
                        with torch.cuda.stream(warmup_stream):
                            with (
                                torch.cuda.amp.autocast()
                                if autocast_fp16
                                else contextlib.nullcontext()
                            ):
                                y = self.model(x)
                            self.out = self.scale * y + self.center
                            x.copy_(y)
                    elif self.iteration == 11:
                        self.graph = torch.cuda.CUDAGraph()
                        x = x.detach().clone()
                        with torch.cuda.graph(self.graph):
                            print("Recording graph!")
                            with (
                                torch.cuda.amp.autocast()
                                if autocast_fp16
                                else contextlib.nullcontext()
                            ):
                                y = self.model(x)
                            self.out = self.scale * y + self.center
                            x.copy_(y)
                    else:
                        self.graph.replay()

                self.iteration += 1
                out = self.out
                time = time + self.time_step

                # create args and kwargs for future use
                restart = dict(
                    x=x[:, None],
                    normalize=False,
                    cuda_graphs=cuda_graphs,
                    autocast_fp16=autocast_fp16,
                    time=time,
                )
                yield time, out, restart
