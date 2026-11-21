#!/usr/bin/env bash
# User-space toolchain + KenLM build in WSL (no sudo). Idempotent.
set -euo pipefail
MM=$HOME/micromamba
mkdir -p "$MM/bin" && cd "$MM"
if [ ! -x "$MM/bin/micromamba" ]; then
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$MM/mm.tar.bz2"
  python3 -c "import tarfile,sys; tarfile.open(sys.argv[1],'r:bz2').extract('bin/micromamba', sys.argv[2])" "$MM/mm.tar.bz2" "$MM"
  chmod +x "$MM/bin/micromamba"
fi
export MAMBA_ROOT_PREFIX="$MM/root"
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/kenlm" ]; then
  "$MM/bin/micromamba" create -y -n kenlm -c conda-forge python=3.10 gxx_linux-64 cmake make libboost-devel eigen zlib bzip2 xz pip git
fi
ENVP="$MAMBA_ROOT_PREFIX/envs/kenlm"
cd "$HOME"
[ -d kenlm ] || git clone --depth 1 https://github.com/kpu/kenlm.git
cd kenlm && mkdir -p build && cd build
export PATH="$ENVP/bin:$PATH" CMAKE_PREFIX_PATH="$ENVP"
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$ENVP/bin/x86_64-conda-linux-gnu-gcc" -DCMAKE_CXX_COMPILER="$ENVP/bin/x86_64-conda-linux-gnu-g++" -DBOOST_ROOT="$ENVP" -DBoost_NO_SYSTEM_PATHS=ON
make -j8 lmplz build_binary
"$ENVP/bin/pip" install pyctcdecode https://github.com/kpu/kenlm/archive/master.zip numpy
echo "KENLM_OK $(ls $HOME/kenlm/build/bin)"
