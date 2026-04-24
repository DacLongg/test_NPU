Bước cài lại từ đầu
Theo bộ khớp mà ta đã xác định:

OpenVINO 2025.4.2
linux-npu-driver v1.28.0
Level Zero 1.24.2
1. Cài driver package chuẩn

mkdir -p ~/npu_install/v1.28.0
cd ~/npu_install/v1.28.0

wget https://github.com/intel/linux-npu-driver/releases/download/v1.28.0/linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz

sudo apt update
sudo apt install -y libtbb12
sudo dpkg -i *.deb
2. Cài Level Zero đúng version

wget https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/level-zero_1.24.2+u24.04_amd64.deb
sudo dpkg -i level-zero_1.24.2+u24.04_amd64.deb
sudo ldconfig
3. Thêm quyền

sudo gpasswd -a $USER render
sudo gpasswd -a $USER video
newgrp render
4. Build lại OpenVINO

cd ~
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git checkout 2025.4.2
git submodule update --init --recursive

mkdir build
cd build
cmake .. \
  -DENABLE_INTEL_NPU=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/openvino
make -j"$(nproc)"
sudo make install
sudo ldconfig
5. Tạo env sạch

python3 -m venv ~/ov_env_src
source ~/ov_env_src/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
source /opt/openvino/setupvars.sh
6. Xác minh

python -c "import openvino; print(openvino.__file__); print(openvino.__version__)"
ldconfig -p | grep ze_loader
dpkg -l | egrep 'intel.*npu|level-zero|libze1'