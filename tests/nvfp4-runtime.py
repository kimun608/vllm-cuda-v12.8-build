"""Check exact FP4 packing and dirty padding on an SM120 CUDA 12.8 build."""
import json

import torch
from vllm import _custom_ops as ops


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def linear_scales(scales, rows, columns):
    rows = round_up(rows, 128)
    blocks = round_up(columns // 16, 4)
    return (
        scales.view(torch.float8_e4m3fn)
        .reshape(1, rows // 128, blocks // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(rows, blocks)
        .float()
    )


@torch.inference_mode()
def main():
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_capability()[0] >= 10, "NVFP4 requires Blackwell"
    values = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
    codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15],
                         dtype=torch.uint8, device="cuda")
    packed = codes[::2] | (codes[1::2] << 4)
    global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    cases = 0
    for dtype in (torch.float16, torch.bfloat16):
        for rows, columns, physical_columns in ((128, 64, 64), (90, 80, 80), (150, 48, 64)):
            for factor in (0, 1, 2):
                x = torch.tensor(values, dtype=dtype, device="cuda").repeat(rows, columns // 16) * factor
                if factor == 0:
                    x.zero_()  # Use positive zero; negative zero has a distinct FP4 encoding.
                expected = packed.repeat(rows, columns // 16) if factor else torch.zeros((rows, columns // 2), dtype=torch.uint8, device="cuda")
                for swizzled in (False, True):
                    out, scales = ops.scaled_fp4_quant(
                        x, global_scale, is_sf_swizzled_layout=swizzled,
                        padded_n=physical_columns if physical_columns != columns else None,
                    )
                    torch.testing.assert_close(out[:, :columns // 2], expected, atol=0, rtol=0)
                    assert torch.count_nonzero(out[:, columns // 2:]) == 0
                    decoded = linear_scales(scales, rows, physical_columns) if swizzled else scales.float()
                    torch.testing.assert_close(decoded[:rows, :columns // 16], torch.full((rows, columns // 16), factor, dtype=torch.float32, device="cuda"), atol=0, rtol=0)
                    assert torch.count_nonzero(decoded[:rows, columns // 16:]) == 0
                    if swizzled:
                        assert torch.count_nonzero(decoded[rows:]) == 0
                        # Dirty the entire allocation, then ensure the kernel overwrites padding.
                        dirty_out = torch.full_like(out, 0xff)
                        dirty_scales = torch.full((round_up(rows, 128), round_up(physical_columns // 16, 4) // 4), 0x7f7f7f7f, dtype=torch.int32, device="cuda")
                        torch.ops._C.scaled_fp4_quant.out(x, global_scale, True, output=dirty_out, output_scale=dirty_scales)
                        torch.testing.assert_close(dirty_out, out, atol=0, rtol=0)
                        torch.testing.assert_close(linear_scales(dirty_scales, rows, physical_columns), decoded, atol=0, rtol=0)
                    cases += 1
    torch.cuda.synchronize()
    print(json.dumps({"status": "passed", "cases": cases, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "checks": ["exact FP4 bytes", "both scale layouts", "zero and scaled input", "padded output", "dirty scale padding"]}, indent=2))


if __name__ == "__main__":
    main()
