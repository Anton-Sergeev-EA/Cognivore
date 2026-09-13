"""Build script for the ``cognivore._native`` C++ extension.

Packaging metadata itself lives in ``pyproject.toml``; this file exists only
because building a compiled pybind11 extension still needs an imperative
hook to describe the sources and compiler flags. If the extension fails to
build (no C++ toolchain available), ``cognivore`` still installs and runs
correctly, falling back to the pure-Python/NumPy index implementation in
``cognivore.index.python_index`` -- see ``cognivore/index/__init__.py``.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _host_supports_avx2() -> bool:
    """Best-effort check that *this* machine's CPU supports AVX2/FMA.

    Because ``pip install`` (without a prebuilt wheel) compiles the
    extension on the same machine that will run it, it's safe to bake in
    AVX2 codegen only when we can confirm the host CPU actually has it --
    otherwise the resulting binary would crash with "illegal instruction"
    the first time ``dot_product`` runs. When we can't tell (non-Linux, or
    the check fails for any reason), we conservatively skip the flag and
    fall back to the portable scalar kernel in vector_index.cpp.
    """
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            flags_line = next((line for line in fh if line.startswith("flags")), "")
        return "avx2" in flags_line and "fma" in flags_line
    except OSError:
        return False


def _supports_openmp(compile_flag: str, link_flag: str) -> bool:
    """Probes the actual compiler (not just the platform) for OpenMP
    support by compiling and linking a throwaway program. This matters
    because the answer varies a lot in practice: it "just works" with GCC
    on Linux (libgomp ships with the compiler), but Apple's default Clang
    has no OpenMP runtime unless the user has separately installed
    `libomp` -- silently requiring `-fopenmp` there would break the build
    for anyone without it, which is worse than shipping the (still
    correct, just single-threaded) fallback.
    """
    compiler = sysconfig.get_config_var("CXX") or ("cl" if sys.platform == "win32" else "c++")
    compiler = compiler.split()[0]
    source = "#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n"
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "probe.cpp"
        bin_path = Path(tmp) / "probe.out"
        src_path.write_text(source)
        try:
            subprocess.run(
                [compiler, compile_flag, link_flag, str(src_path), "-o", str(bin_path)],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            return False


extra_compile_args: list[str] = []
extra_link_args: list[str] = []
if sys.platform != "win32":
    extra_compile_args += ["-O3", "-std=c++17"]
    if _host_supports_avx2():
        extra_compile_args += ["-mavx2", "-mfma"]
    if _supports_openmp("-fopenmp", "-fopenmp"):
        extra_compile_args += ["-fopenmp"]
        extra_link_args += ["-fopenmp"]
else:
    extra_compile_args += ["/std:c++17", "/O2"]
    if _supports_openmp("/openmp", "/openmp"):
        extra_compile_args += ["/openmp"]

ext_modules = [
    Pybind11Extension(
        "cognivore._native",
        ["native/bindings.cpp", "native/vector_index.cpp"],
        include_dirs=["native"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
