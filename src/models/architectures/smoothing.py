import torch
import torch.nn as nn


class SegmentedSmoothing(nn.Module):
    def __init__(self, kernel_size: int, boundary_width: int):
        """
        Smooths last 2 dimensions of the input tensor with average
        pooling, but the boundary and interior regions are smoothed
        separately.
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.half_kernel = kernel_size // 2
        self.boundary_width = boundary_width

    def _in_boundary(self, i: int, j: int, x_w: int, x_h: int) -> bool:
        return (
            i < self.boundary_width
            or i >= x_w - self.boundary_width
            or j < self.boundary_width
            or j >= x_h - self.boundary_width
        )

    def _smooth(self, x: torch.Tensor, w_idx: int, h_idx: int) -> torch.Tensor:
        x_w = x.shape[-2]
        x_h = x.shape[-1]
        x_elem_out = torch.zeros(x.shape[:-2], device=x.device, dtype=x.dtype)

        in_h_0, in_h_1 = max(0, w_idx - self.half_kernel), min(
            x_w, w_idx + self.half_kernel + 1
        )
        in_w_0, in_w_1 = max(0, h_idx - self.half_kernel), min(
            x_h, h_idx + self.half_kernel + 1
        )
        divide_factor = 0
        for i in range(in_h_0, in_h_1):
            for j in range(in_w_0, in_w_1):
                if self._in_boundary(i, j, x_w, x_h) == self._in_boundary(
                    w_idx, h_idx, x_w, x_h
                ):
                    x_elem_out += x[..., i, j]
                    divide_factor += 1
        x_elem_out /= divide_factor
        return x_elem_out

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_w, x_h = x_in.shape[-2:]
        x_out = torch.zeros_like(x_in)
        for i in range(x_w):
            for j in range(x_h):
                x_out[..., i, j] = self._smooth(x_in, i, j)
        return x_out


class SegmentedSmoothingV2(nn.Module):
    def __init__(self, kernel_size: int, boundary_width: int):
        """
        Smooths last 2 dimensions of the input tensor with average
        pooling, but the boundary and interior regions are smoothed
        separately.
        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.boundary_width = boundary_width

        self.avg_pool = nn.AvgPool3d(
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, 1, 1),
            padding=(0, kernel_size // 2, kernel_size // 2),
            count_include_pad=False,
        )
        self.sum_pool = nn.AvgPool3d(
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, 1, 1),
            padding=(0, kernel_size // 2, kernel_size // 2),
            divisor_override=1,
        )

        interior_slice = slice(boundary_width, -boundary_width)
        self.interior_idx = (..., interior_slice, interior_slice)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        interior_mask = torch.zeros_like(x_in, device=x_in.device, dtype=torch.bool)
        interior_mask[self.interior_idx] = True

        x_out = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        # Smooth interior
        x_out[self.interior_idx] = self.avg_pool(x_in[self.interior_idx])
        # Smooth boundary
        x_boundary_sum = self.sum_pool(x_in * (~interior_mask).float())
        x_boundary_count = self.sum_pool((~interior_mask).float())
        x_out = torch.where(~interior_mask, x_boundary_sum / x_boundary_count, x_out)

        return x_out
