#!/usr/bin/env bash

set -euo pipefail

OPENVINO_TAG="2025.4.2"
NPU_DRIVER_TAG="v1.28.0"
NPU_DRIVER_ARCHIVE="linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz"
LEVEL_ZERO_DEB="level-zero_1.24.2+u24.04_amd64.deb"
WORK_DIR="${HOME}/npu_install/${NPU_DRIVER_TAG}"
OPENVINO_DIR="${HOME}/openvino"
VENV_DIR="${HOME}/ov_env_src"
BACKUP_DIR="/usr/local/lib/disabled-npu-stack"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

require_sudo() {
  sudo -v
}

cleanup_old_stack() {
  log "Removing old packages and environments"
  sudo dpkg --purge --force-remove-reinstreq \
    intel-driver-compiler-npu \
    intel-fw-npu \
    intel-level-zero-npu \
    intel-level-zero-npu-dbgsym || true

  sudo dpkg --purge --force-remove-reinstreq \
    level-zero \
    level-zero-devel \
    libze1 || true

  python3 -m pip uninstall -y openvino openvino-dev 2>/dev/null || true
  rm -rf "${HOME}/ov_env" "${VENV_DIR}"
  sudo rm -rf /opt/openvino

  sudo mkdir -p "${BACKUP_DIR}"
  sudo mv /usr/local/lib/libze_* "${BACKUP_DIR}/" 2>/dev/null || true
  sudo mv /usr/local/lib/libze_intel_npu.so* "${BACKUP_DIR}/" 2>/dev/null || true
  sudo ldconfig

  rm -rf "${HOME}/linux-npu-driver/build" "${OPENVINO_DIR}/build"
}

install_npu_driver() {
  log "Installing linux-npu-driver ${NPU_DRIVER_TAG}"
  mkdir -p "${WORK_DIR}"
  cd "${WORK_DIR}"

  rm -f "${NPU_DRIVER_ARCHIVE}"
  wget "https://github.com/intel/linux-npu-driver/releases/download/${NPU_DRIVER_TAG}/${NPU_DRIVER_ARCHIVE}"
  tar -xf "${NPU_DRIVER_ARCHIVE}"

  sudo apt update
  sudo apt install -y libtbb12
  sudo dpkg -i ./*.deb || sudo apt install -f -y
}

install_level_zero() {
  log "Installing Level Zero 1.24.2"
  cd "${WORK_DIR}"
  rm -f "${LEVEL_ZERO_DEB}"
  wget "https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/${LEVEL_ZERO_DEB}"
  sudo dpkg -i "${LEVEL_ZERO_DEB}" || sudo apt install -f -y
  sudo ldconfig
}

configure_groups() {
  log "Adding current user to render and video groups"
  sudo gpasswd -a "${USER}" render
  sudo gpasswd -a "${USER}" video
}

build_openvino() {
  log "Building OpenVINO ${OPENVINO_TAG} from source"
  cd "${HOME}"

  if [[ ! -d "${OPENVINO_DIR}/.git" ]]; then
    git clone https://github.com/openvinotoolkit/openvino.git "${OPENVINO_DIR}"
  fi

  cd "${OPENVINO_DIR}"
  git fetch --tags
  git checkout "${OPENVINO_TAG}"
  git submodule update --init --recursive

  mkdir -p build
  cd build

  cmake .. \
    -DENABLE_INTEL_NPU=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/openvino

  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
}

create_venv() {
  log "Creating clean Python virtual environment"
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install numpy
}

print_next_steps() {
  cat <<EOF

Installation finished.

Next steps:
1. Log out and log back in, or reboot, so the render/video group change takes effect.
2. Activate the environment:
   source "${VENV_DIR}/bin/activate"
3. Load OpenVINO environment:
   source /opt/openvino/setupvars.sh
4. Verify:
   python -c "import openvino; print(openvino.__file__); print(openvino.__version__)"
   ldconfig -p | grep ze_loader
   dpkg -l | egrep 'intel.*npu|level-zero|libze1'
5. Run probe:
   cd "${HOME}/myproject/test_NPU"
   python TestNpu.py --probe

If probe still fails, collect the outputs above before changing versions again.
EOF
}

main() {
  require_sudo
  cleanup_old_stack
  install_npu_driver
  install_level_zero
  configure_groups
  build_openvino
  create_venv
  print_next_steps
}

main "$@"
