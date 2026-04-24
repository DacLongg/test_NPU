# Huong Dan Cai Dat NPU Tren Ubuntu 24.04

Tai lieu nay mo ta quy trinh cai dat de OpenVINO co the nhin thay va chay duoc `NPU` tren may Intel Core Ultra, dong thoi giai thich y nghia cua tung buoc.

Phan huong dan nay duoc rut ra tu qua trinh debug thuc te trong du an nay. Bo version da chay duoc la:

- `Ubuntu 24.04`
- `OpenVINO 2025.4.2` build tu source
- `linux-npu-driver v1.28.0`
- `Level Zero 1.24.2`

## 1. Muc tieu cua qua trinh cai dat

De benchmark `CPU` va `NPU`, chi phan Python la chua du. Stack hoan chinh gom 4 lop:

1. `Kernel driver`:
   Driver trong kernel giao tiep voi phan cung NPU.
2. `Firmware`:
   Firmware duoc nap cho thiet bi khi khoi dong.
3. `Userspace runtime`:
   Cac thu vien nhu `libze_loader.so`, `libze_intel_npu.so` cho plugin NPU su dung.
4. `OpenVINO`:
   Plugin `intel_npu` trong OpenVINO dung runtime ben duoi de compile va chay graph.

Neu 4 lop nay lech version, hien tuong thuong gap la:

- `NPU` xuat hien trong `available_devices`
- nhung `compile_model(..., "NPU")` bi loi `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`

## 2. Kiem tra phan cung va he thong truoc khi cai

### 2.1 Kiem tra kernel driver

```bash
lsmod | grep intel_vpu
```

Y nghia:

- Neu thay `intel_vpu`, kernel da nap driver cho NPU.
- Neu khong thay, cac buoc phia tren Python/OpenVINO se khong co y nghia.

### 2.2 Kiem tra firmware

```bash
sudo dmesg | grep -i vpu
```

Y nghia:

- Xac nhan firmware duoc nap thanh cong.
- Neu firmware loi, NPU co the van xuat hien mot phan nhung khong chay inference on dinh.

### 2.3 Kiem tra node thiet bi

```bash
ls /dev/accel/
```

Y nghia:

- Neu co `accel0`, he dieu hanh da tao device node cho NPU.

## 3. Vi sao can phan quyen `render` va `video`

```bash
sudo usermod -aG render $USER
sudo usermod -aG video $USER
```

Y nghia:

- NPU va cac accelerator cua Intel thuong duoc quan ly qua cac device node can quyen `render` hoac `video`.
- Neu thieu group nay, OpenVINO co the detect duoc thiet bi nhung user khong du quyen de dung.

Sau do can `logout/login` hoac reboot de group co hieu luc day du.

## 4. Don dep stack cu truoc khi cai lai

Day la buoc quan trong nhat neu may da tung cai nhieu lan.

### 4.1 Go moi ban OpenVINO pip cu

```bash
python3 -m pip uninstall -y openvino openvino-dev
rm -rf ~/ov_env ~/ov_env_src
```

Y nghia:

- Loai bo ban `openvino` trong pip de tranh trung voi ban build tu source.

### 4.2 Xoa OpenVINO cai cu trong `/opt/openvino`

```bash
sudo rm -rf /opt/openvino
```

Y nghia:

- Dam bao khong con mot ban install cu nao bi `setupvars.sh` nap vao moi truong.

### 4.3 Day bo `libze_*` cu ra khoi `/usr/local/lib`

```bash
sudo mkdir -p /usr/local/lib/disabled-npu-stack
sudo mv /usr/local/lib/libze_* /usr/local/lib/disabled-npu-stack/ 2>/dev/null || true
sudo mv /usr/local/lib/libze_intel_npu.so* /usr/local/lib/disabled-npu-stack/ 2>/dev/null || true
sudo ldconfig
```

Y nghia:

- Day la buoc tranh `userspace runtime` cu chiem uu tien trong linker.
- Trong qua trinh debug, day la nguyen nhan chinh gay mismatch.
- Khong xoa han ngay, ma backup sang cho khac de co the doi chieu neu can.

### 4.4 Kiem tra lai da sach chua

```bash
python3 -c "import openvino; print(openvino.__file__)"
ldconfig -p | grep ze_loader
```

Ket qua mong muon:

- Python bao `ModuleNotFoundError`
- `ldconfig` khong con tro den `ze_loader` cu trong `/usr/local/lib`

## 5. Cai `linux-npu-driver v1.28.0`

```bash
mkdir -p ~/npu_install/v1.28.0
cd ~/npu_install/v1.28.0

wget https://github.com/intel/linux-npu-driver/releases/download/v1.28.0/linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.28.0.20251218-20347000698-ubuntu2404.tar.gz

sudo apt update
sudo apt install -y libtbb12
sudo dpkg -i *.deb
```

Y nghia:

- Cai `userspace runtime` da duoc release san, thay vi build ngau nhien tu `main`.
- Ban `v1.28.0` la ban da duoc xac minh tuong thich voi OpenVINO `2025.4`.

## 6. Cai `Level Zero 1.24.2`

```bash
cd ~/npu_install/v1.28.0
wget https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/level-zero_1.24.2+u24.04_amd64.deb
sudo dpkg -i level-zero_1.24.2+u24.04_amd64.deb
sudo ldconfig
```

Y nghia:

- `Level Zero` la lop API/runtime ma plugin NPU su dung de tao va chay graph.
- Neu version `Level Zero` va `linux-npu-driver` khong khop, NPU thuong detect duoc nhung compile graph that bai.

## 7. Build `OpenVINO 2025.4.2` tu source

```bash
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
```

Y nghia:

- Build OpenVINO dung tag ma ban muon dung.
- Bat plugin `intel_npu`.
- Cai vao `/opt/openvino` de phan biet ro voi pip package.

Luu y:

- Truoc day ban tung dung `OpenVINO 2025.4.2` voi `linux-npu-driver v1.32.1`, ket qua la toan bo graph fail.
- Nguyen nhan la lech version, khong phai do code benchmark.

## 8. Tao virtual environment sach

```bash
python3 -m venv ~/ov_env_src
source ~/ov_env_src/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
source /opt/openvino/setupvars.sh
```

Y nghia:

- `venv` chi dung de chua `numpy` va moi truong Python sach.
- `openvino` se duoc import tu `/opt/openvino`, khong phai tu pip.

## 9. Kiem tra sau cai dat

### 9.1 Kiem tra Python dang import dung OpenVINO

```bash
python -c "import openvino; print(openvino.__file__); print(openvino.__version__)"
```

Ket qua mong muon:

- duong dan duoi `/opt/openvino/...`
- version `2025.4.2`

### 9.2 Kiem tra linker

```bash
ldconfig -p | grep ze_loader
```

Ket qua mong muon:

- `libze_loader.so` tro den `/usr/lib/x86_64-linux-gnu/...`
- khong con tro den `/usr/local/lib/libze_loader.so`

### 9.3 Kiem tra package da duoc cai

```bash
dpkg -l | egrep 'intel.*npu|level-zero|libze1'
```

Y nghia:

- Xac nhan stack userspace hien tai den tu package manager.

## 10. Kiem tra NPU bang probe

```bash
cd ~/myproject/test_NPU
python TestNpu.py --probe
```

Neu stack dung, ban se thay:

- `Detected devices: ['CPU', 'NPU']`
- cac `NPU properties`
- `Compile probe: ... Compile: OK` voi cac graph `mlp_tiny`, `cnn_tiny`, `cnn_default`

Day la dau hieu quan trong nhat cho thay NPU co the compile graph that su.

## 11. Nhung loi da gap trong qua trinh debug

### 11.1 NPU detect duoc nhung khong benchmark duoc

Loi:

```text
ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
```

Nguyen nhan:

- OpenVINO va `linux-npu-driver` lech version
- `libze_loader.so` va `libze_intel_npu.so` build tay trong `/usr/local/lib` ghi de runtime dung

### 11.2 Dung OpenVINO pip thay vi source build

Ban `openvino` tu pip co the khong nhin thay hoac khong khop plugin NPU trong moi truong cua ban. Vi vay du an nay chot huong:

- `OpenVINO` build tu source
- runtime NPU dung package release phu hop

## 12. File script tu dong hoa

Neu muon tu dong hoa toan bo quy trinh, co the tham khao file [reinstall_npu_stack.sh](/home/ddragon/myproject/test_NPU/reinstall_npu_stack.sh).

Script nay gom:

- don dep stack cu
- cai `linux-npu-driver v1.28.0`
- cai `Level Zero 1.24.2`
- build `OpenVINO 2025.4.2`
- tao `venv` sach

## 13. Ket luan

De chay duoc NPU, dieu quan trong nhat khong phai la chi cai them Python package, ma la phai dong bo:

- kernel driver
- firmware
- Level Zero
- Intel NPU userspace runtime
- OpenVINO plugin

Khi bo version khop nhau, `TestNpu.py --probe` se compile OK tren NPU. Khi do benchmark CPU va NPU moi co y nghia.
