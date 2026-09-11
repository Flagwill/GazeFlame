"""ROCm/GPU 设备支持辅助（AMD 独显/核显走 PyTorch HIP 后端）。

本机 Radeon 680M 是 gfx1035，不在 ROCm 官方支持列表里：必须设置
``HSA_OVERRIDE_GFX_VERSION=10.3.0`` 让运行时加载 gfx1030 的 kernel，
否则 ``torch.cuda.is_available()`` 仍返回 True，但任何 kernel 一执行就报
``HIP error: invalid device function``（这是最大的坑，勿删这个探测）。

PyTorch 的 ROCm build 完全复用 ``cuda`` 这套 API 名字：设备字符串就是
``"cuda"``，不需要写 ``"hip"``。

用法（所有入口脚本都从这里拿设备）::

    from common.gpu import setup_rocm_env, resolve_device
    setup_rocm_env()                      # import torch 后、首次 CUDA 调用前
    device = resolve_device(args.device)  # "auto"/"cuda"/"cpu" -> "cuda"/"cpu"
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

# gfx1035 (Radeon 680M) 等 RDNA2 非官方支持芯片的 ISA 覆盖值。
GFX_OVERRIDE_DEFAULT = "10.3.0"

# gfx1035 核显安全占空比（秒）。实测：背靠背满载 ~300-400 次 forward 必现
# GPU Hang / Memory access fault（reset 后仍复现）；实测 >=100ms 空闲窗口可
# 连续运行数千帧。稳定独显不会启用该护栏；也可用环境变量微调本机余量。
ROCM_SAFE_INTERVAL = 0.10

# ROCm torch wheel 在 torch/lib/ 里捆绑了旧版 libdrm/libdrm_amdgpu，
# import torch 会把它们带进进程全局符号表，随后 pyrender/pyglet 的 GLX
# (Mesa radeonsi) 绑定到旧符号 → "NoSuchConfigException / Failed to
# initialize Pyglet window"。对策：在 import torch 之前先用 RTLD_GLOBAL
# 抢注系统的 libdrm.so.2 / libdrm_amdgpu.so.1（ld.so 按 SONAME 去重，
# torch 的捆绑副本不会被加载）。所有 GUI+ROCm 混合入口必须先调用本函数。
_SYSTEM_LIBDRM_CANDIDATES = (
    "/usr/lib/libdrm.so.2",  # Arch/others
    "/usr/lib/x86_64-linux-gnu/libdrm.so.2",  # Debian/Ubuntu
    "/usr/lib64/libdrm.so.2",
)
_SYSTEM_LIBDRM_AMDGPU_CANDIDATES = (
    "/usr/lib/libdrm_amdgpu.so.1",
    "/usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1",
    "/usr/lib64/libdrm_amdgpu.so.1",
)


def preload_system_libdrm() -> None:
    """（内部）把系统 libdrm(-amdgpu) 抢注进全局符号表（须早于 import torch）。"""
    import ctypes
    import platform

    if platform.system() != "Linux":
        return

    def _first_existing(candidates):
        return next((p for p in candidates if os.path.exists(p)), None)

    for candidates in (_SYSTEM_LIBDRM_CANDIDATES, _SYSTEM_LIBDRM_AMDGPU_CANDIDATES):
        path = _first_existing(candidates)
        if path is None:
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:  # 无权限/依赖缺失: 不致命, 交给后续 GLX 报错
            pass


def prepare_amd_gpu_env() -> None:
    """ROCm 混合环境（GPU 推理 + pyrender）必须在 ``import torch`` 之前调用。

    做三件事：
      1. ``os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")`` —
         gfx1035 等非官方芯片否则任何 kernel 都报 invalid device function；
      2. 抢注系统 libdrm — 否则 torch/lib 的旧 libdrm 破坏 pyrender 的 GLX。
      3. 关闭 MIOpen GCN 汇编卷积 — gfx1035 借助 ISA override 运行汇编 kernel
         会在长时间重复 forward 后触发 amdgpu page fault；通用 HIP kernel
         仍然运行在 GPU 上。

    纯 CPU / NVIDIA-CUDA build 下两个动作都无副作用，可以无条件调用。
    """
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", GFX_OVERRIDE_DEFAULT)
    preload_system_libdrm()
    if os.environ.get("GAZE_ROCM_SAFE_KERNELS", "1") != "0":
        os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")


def is_rocm_build() -> bool:
    """当前 torch 是否为 ROCm/HIP build（torch.version.hip 非空）。"""
    import torch

    return getattr(torch.version, "hip", None) is not None


@lru_cache(maxsize=1)
def is_rocm_igpu() -> bool:
    """Return whether the active ROCm device is an integrated GPU.

    The duty-cycle workaround is required for the shared-memory Radeon 680M,
    but applying it to a discrete AMD card needlessly caps throughput.  An
    explicit ``GAZE_ROCM_IGPU`` override is provided for unusual systems and
    test environments; unknown hardware remains conservative.
    """
    override = os.environ.get("GAZE_ROCM_IGPU")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}
    try:
        import torch

        if not is_rocm_build() or not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        name = str(getattr(props, "name", "")).lower()
        # Common AMD integrated-device names, including the Radeon 680M in
        # the documented deployment.  If the name is not recognizable, use
        # the conservative path so stability is preferred over peak speed.
        integrated_markers = (
            "680m", "660m", "780m", "760m", "740m", "radeon vega",
            "integrated graphics", "radeon graphics",
        )
        if any(marker in name for marker in integrated_markers):
            return True
        # A very small reported aperture is characteristic of an iGPU.  Do
        # not classify larger devices this way because low-end dGPUs exist.
        total_memory = int(getattr(props, "total_memory", 0) or 0)
        return 0 < total_memory <= 2 * 1024**3
    except Exception:  # noqa: BLE001 - hardware probing must never break startup
        return True


def setup_rocm_env() -> None:
    """HIP build 下补上 HSA_OVERRIDE_GFX_VERSION（尊重用户已设置的值）。"""
    if is_rocm_build():
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", GFX_OVERRIDE_DEFAULT)


def _probe_cuda() -> tuple[bool, str | None]:
    """真正跑一个小 kernel 验证 GPU 可用（is_available 不够可靠）。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() 为 False"
        x = torch.rand(64, 64, device="cuda")
        _ = (x @ x).sum()
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001 - 任何 HIP/CUDA 错误都视为不可用
        return False, str(exc)


def resolve_device(requested: str = "auto") -> str:
    """把 ``auto``/``cuda`` 解析成实际可用的设备字符串（``cuda``/``cpu``）。"""
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    setup_rocm_env()  # 覆盖值需在首次 CUDA/HIP 调用之前设置
    gpu_ok, probe_error = _probe_cuda()
    if gpu_ok:
        return "cuda"
    if requested != "auto":
        print(f"警告: 请求了 GPU 但 kernel 探测失败 ({probe_error})，回退到 cpu")
    elif probe_error:
        print(f"提示: auto 的 GPU kernel 探测失败 ({probe_error})，当前回退到 cpu")
    return "cpu"


def rocm_safe_interval(default: float = ROCM_SAFE_INTERVAL) -> float:
    """Return the validated inter-forward delay for the ROCm duty-cycle guard."""
    try:
        value = float(os.environ.get("GAZE_ROCM_SAFE_INTERVAL", default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value < 0.0:
        return default
    return value


def synchronize_device(device: str) -> None:
    """Wait until work queued on a CUDA/ROCm device has actually completed."""
    if str(device) != "cuda":
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_rocm_display_env(device: str, headless: bool,
                             flame_enabled: bool) -> bool:
    """Keep HighGUI rendering off an ROCm iGPU while inference stays on GPU.

    On the Radeon 680M, the desktop compositor can contend with HIP even when
    the application itself does not create a pyrender scene.  HighGUI uses Qt
    for its window; force its client-side path through XCB/software rasterizer
    for this narrow live-GUI case.  ``GAZE_ROCM_SOFTWARE_DISPLAY=0`` restores
    the old display path for systems where this workaround is unnecessary.
    """
    if headless or flame_enabled or str(device) != "cuda":
        return False
    if not is_rocm_build() or os.environ.get("GAZE_ROCM_SOFTWARE_DISPLAY", "1") == "0":
        return False
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    return True


def rocm_igpu_guard(device: str, elapsed: float,
                    interval: float = ROCM_SAFE_INTERVAL) -> None:
    """ROCm 核显占空比护栏：把相邻两次 GPU forward 的间隔补足到 ``interval``。

    gfx1035 实测：背靠背满载 ~300-400 次 forward 必现 GPU Hang（详见
    AGENTS.md 评估结论）；默认在每次 forward 后留 >=100ms 的安全间隔。
    CPU / NVIDIA-CUDA 调用下为无操作；真实摄像头天然帧间隔 >100ms 时也是
    无操作（不会拖慢实时流水线）。
    稳定的 AMD 独显若不需要护栏，可设环境变量 ``GAZE_ROCM_SAFE_MODE=0``。
    """
    if str(device) != "cuda" or not is_rocm_build() or not is_rocm_igpu():
        return
    if os.environ.get("GAZE_ROCM_SAFE_MODE", "1") == "0":
        return
    import time

    interval = rocm_safe_interval(interval)
    remaining = interval - elapsed
    if remaining > 0:
        time.sleep(remaining)


def flame_render_device(device: str) -> str:
    """FLAME 形变(供 pyrender 渲染)应使用的 torch 设备。

    pyrender 的离屏渲染本身走 OpenGL(CPU 侧驱动光栅化), FLAME 顶点形变计算量
    极小(batch=1, CPU 上 5ms 级), 放 GPU 对速度无帮助。但把形变丢到这块**与显示
    共享内存的 ROCm 核显**上, 会叠加在 GazeTR 之上让 iGPU 持续满载 → 渲染几十秒
    即 GPU Hang(实测 --flame 跑一会就 Memory access fault)。所以 ROCm 核显下
    FLAME 形变使用 cpu；离散 AMD/NVIDIA CUDA 沿用推理设备。
    """
    if str(device) == "cuda" and is_rocm_build() and is_rocm_igpu():
        return "cpu"
    return device


def flame_realtime_device(requested: str, flame_enabled: bool) -> str:
    """实时循环里 FLAME 面板开启时的推理设备（gfx1035 核显安全策略）。

    实测：只要 pyrender 的 OpenGL 渲染与 HIP kernel 在同一块核显上**同时持续
    运行**（--flame 且有人脸，~55-160 帧内必现 Memory access fault / GPU Hang），
    与 FLAME 形变放在哪个设备无关；把 GL 切到 llvmpipe 也救不了（X 显示合成
    仍占 GPU）。而**无 GL 负载**时同样的 HIP 节奏可稳定跑数千帧。
    对策：核显 + FLAME 实时场景下推理一律回退 cpu —— GazeTR CPU 前向仅 ~12ms，
    本来就满足 10fps 实时，GPU 在此场景没有任何速度优势；GL 则回到与改动前
    数月一致的"纯图形"使用方式（稳定）。
    NVIDIA/CPU 构建、稳定 AMD 独显（GAZE_ROCM_SAFE_MODE=0）不受影响。
    """
    requested = (requested or "auto").lower()
    safe_mode = os.environ.get("GAZE_ROCM_SAFE_MODE", "1") != "0"

    resolved = resolve_device(requested)
    if (flame_enabled and resolved == "cuda" and is_rocm_build()
            and is_rocm_igpu() and safe_mode):
        print("提示: ROCm 核显与 pyrender 的 GPU 渲染互斥（持续共载会 GPU Hang），"
              "FLAME 实时模式改用 cpu 推理（10fps 实时足够；独显可设 "
              "GAZE_ROCM_SAFE_MODE=0 解除）")
        return "cpu"
    if (resolved == "cuda" and is_rocm_build() and is_rocm_igpu()
            and safe_mode and requested == "auto"):
        print(
            "提示: ROCm 实时 auto 使用 GPU，已启用占空比护栏 "
            f"(相邻 kernel 间隔 >= {rocm_safe_interval():.2f}s；可用 "
            "GAZE_ROCM_SAFE_INTERVAL 调整)"
        )
    if resolved == "cuda" and is_rocm_build() \
            and is_rocm_igpu() \
            and os.environ.get("MIOPEN_DEBUG_GCN_ASM_KERNELS") == "0":
        print("提示: 已启用 MIOpen 兼容 kernel 模式（仍使用 GPU）；可用 "
              "GAZE_ROCM_SAFE_KERNELS=0 恢复汇编 kernel")
    return resolved
