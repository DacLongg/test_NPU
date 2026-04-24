# Giai Thich Code Benchmark CPU Va NPU

Tai lieu nay giai thich:

1. `TestNpu.py` dang lam gi
2. `CPU` va `NPU` khac nhau o dau
3. vi sao trong mot so bai test `CPU` nhanh hon `NPU`
4. cach doc ket qua benchmark dung

## 1. Muc dich cua `TestNpu.py`

File [TestNpu.py](/home/ddragon/myproject/test_NPU/TestNpu.py:1) co 2 vai tro:

1. `Probe`:
   Kiem tra xem stack NPU da san sang hay chua.
2. `Benchmark`:
   Do toc do inference tren `CPU` va `NPU` voi cung mot model synthetic.

No khong phai la model AI thuc te cua du an. No la cong cu de:

- xac minh NPU compile duoc graph
- so sanh xu huong hieu nang giua CPU va NPU

## 1.1 `Synthetic model` la gi

`Synthetic model` la model duoc tao ra de benchmark hoac test he thong, chu khong phai model duoc huan luyen de giai mot bai toan that nhu phan loai anh, nhan dien vat the, hay chatbot.

Noi ngan gon:

- model that:
  co du lieu huan luyen, co muc tieu ung dung, co trong so co y nghia nghiep vu
- model synthetic:
  chi tao ra de sinh workload tinh toan cho engine inference

Trong file `TestNpu.py`, cac trong so:

- duoc tao ngau nhien bang `numpy`
- khong phai tham so da huan luyen
- khong dung de dua ra du doan co y nghia

Y nghia cua model synthetic trong bai test:

- giup tao mot graph on dinh, lap lai duoc
- de dang dieu chinh muc do nang cua workload
- phu hop de kiem tra `CPU` va `NPU` co compile, chay, va xu ly duoc graph hay khong

Han che cua model synthetic:

- khong dai dien day du cho model AI that
- khong phan anh day du chi phi tien xu ly, hau xu ly, quantization, memory layout that
- khong dam bao ket qua benchmark se giong khi chay model san pham

Vi vay, benchmark synthetic dung de:

- smoke test
- debug stack
- so sanh xu huong

va khong nen dung de ket luan tuyet doi rang:

- `NPU nhanh hon CPU trong moi truong hop`
- hoac `CPU luon nhanh hon NPU`

## 2. Tong quan cau truc code

### 2.1 `parse_args()`

Ham [parse_args()](/home/ddragon/myproject/test_NPU/TestNpu.py:9) doc tham so command line.

Nhung tham so quan trong:

- `--model-type cnn|mlp`
- `--batch`
- `--height --width`
- `--features --layers`
- `--hint LATENCY|THROUGHPUT`
- `--num-requests`
- `--repeats`
- `--probe`

Y nghia:

- Cho phep thay doi kieu workload de xem CPU/NPU phan ung ra sao.

### 2.2 `import_dependencies()`

Ham [import_dependencies()](/home/ddragon/myproject/test_NPU/TestNpu.py:110) nap:

- `numpy`
- `openvino`
- `openvino.runtime`

Neu thieu dependency, chuong trinh dung ngay.

### 2.3 `print_diagnostics()`

Ham [print_diagnostics()](/home/ddragon/myproject/test_NPU/TestNpu.py:150) in:

- `LD_LIBRARY_PATH`
- `libze_loader.so`
- driver `xe`
- danh sach OpenVINO devices

Y nghia:

- Day la tang chan doan moi truong.
- No giup biet duoc minh dang dung runtime nao truoc khi benchmark.

## 3. Hai loai model synthetic trong script

### 3.1 MLP

Ham [create_mlp_model()](/home/ddragon/myproject/test_NPU/TestNpu.py:192) tao model gom:

- `MatMul`
- `Add`
- `ReLU`

lap lai theo so `layers`.

Y nghia:

- Mo phong workload dense tensor.
- Thuong phu hop de test kha nang compile graph co nhieu phep nhan ma tran.

### 3.2 CNN

Ham [create_cnn_model()](/home/ddragon/myproject/test_NPU/TestNpu.py:213) tao model gom:

- `Convolution`
- `Add`
- `ReLU`
- `MaxPool`

Y nghia:

- Gan voi workload thi giac may tinh hon.
- Thuong la bai smoke test hop ly hon cho NPU so voi MLP.

## 4. Du lieu input duoc tao nhu the nao

Ham [build_input_data()](/home/ddragon/myproject/test_NPU/TestNpu.py:255) sinh random tensor bang `numpy`.

Y nghia:

- Dam bao `CPU` va `NPU` nhan cung mot input shape
- Giup benchmark tap trung vao toc do inference, khong phu thuoc du lieu that

Han che:

- Vi du lieu va model deu synthetic, ket qua khong the thay the benchmark tren model AI that.

## 5. Dinh nghia `latency` va `throughput`

Day la hai chi so rat hay bi nham lan khi benchmark AI.

### 5.1 `Latency` la gi

`Latency` la thoi gian de hoan thanh mot lan inference.

Co the hieu don gian:

- dua 1 input vao
- cho model xu ly xong
- nhan output
- tong thoi gian do la `latency`

Neu mot inference mat `2 ms`, ta noi latency la `2 ms`.

Y nghia:

- phu hop voi bai toan can phan hoi nhanh cho tung request
- vi du:
  - mot camera gui tung frame de xu ly ngay
  - mot tro ly ao tra loi tung yeu cau
  - mot he thong can response time thap

`Latency` thap co nghia la:

- nguoi dung doi it hon
- tung request xong nhanh hon

### 5.2 `Throughput` la gi

`Throughput` la so luong inference hoan thanh trong mot don vi thoi gian.

Thuong duoc bieu dien bang:

- `FPS`
- hoac `samples per second`

Neu trong 1 giay xu ly xong 500 inference, throughput la `500 FPS`.

Y nghia:

- phu hop voi bai toan can xu ly duoc nhieu viec nhat co the
- vi du:
  - server xu ly nhieu request cung luc
  - pipeline xu ly hang loat
  - bat ky he thong nao uu tien tong nang suat

`Throughput` cao co nghia la:

- trong cung mot khoang thoi gian, he thong xu ly duoc nhieu mau hon

### 5.3 Khac nhau giua latency va throughput

Hai chi so nay lien quan nhung khong giong nhau.

Mot he thong co the:

- latency rat tot nhung throughput khong cao
- throughput rat cao nhung latency tung request khong phai tot nhat

Vi du:

- CPU thuong manh o latency cua request don le
- NPU co the manh hon o throughput khi co nhieu request song song

Do do, khi hoi:

- `CPU nhanh hon NPU khong?`

can hoi tiep:

- nhanh hon theo `latency` hay `throughput`?
- voi `batch` nao?
- voi bao nhieu request song song?

## 6. Script do latency va throughput ra sao

### 5.1 Chay sync

Ham [run_sync_requests()](/home/ddragon/myproject/test_NPU/TestNpu.py:263) goi:

```python
compiled_model.infer_new_request(...)
```

lap lai tung lan.

Y nghia:

- Day la kieu do `latency` co ban.
- Moi request chay xong roi request sau moi bat dau.

### 5.2 Chay async

Ham [run_async_requests()](/home/ddragon/myproject/test_NPU/TestNpu.py:270) tao nhieu infer request va `start_async()`.

Y nghia:

- Day la cach do than thien hon voi `throughput`.
- Co the phan anh tot hon kha nang cua NPU khi co nhieu request song song.

### 6.3 Cong thuc script dang dung

Trong `benchmark_device()`, script do thoi gian tong cho `iterations` lan chay, sau do tinh:

```text
avg_latency_ms = tong_thoi_gian / so_lan_chay
throughput_fps = so_lan_chay / tong_thoi_gian
```

Sau do script lap lai nhieu lan va lay `median`.

Y nghia:

- `latency` trong script la latency trung binh moi request cua mot dot benchmark
- `throughput` trong script la tong request moi giay cua dot benchmark do
- `median` giup giam anh huong cua lan do bi nhieu

## 7. Nguyen ly hoat dong cua model benchmark

Phan nay quan trong, vi no giai thich "model dang tinh gi" thay vi chi liet ke operator.

### 7.1 Nguyen ly cua `MLP`

`MLP` trong script duoc tao boi [create_mlp_model()](/home/ddragon/myproject/test_NPU/TestNpu.py:192).

Mot lop `MLP` trong bai test thuc hien chuoi tinh toan:

1. `MatMul`
2. `Add`
3. `ReLU`

Neu viet gon theo toan hoc:

```text
Y = ReLU(XW + b)
```

Trong do:

- `X`: input tensor
- `W`: ma tran trong so
- `b`: bias
- `ReLU`: giu gia tri duong, dua gia tri am ve 0

Script lap chuoi nay nhieu lan theo `--layers`.

Y nghia tinh toan:

- `MatMul` la phep tinh nang nhat
- no mo phong workload dense linear algebra
- rat phu hop de tao tai tinh toan lien tuc tren tensor

Y nghia benchmark:

- giup xem thiet bi xu ly cac phep nhan ma tran va activation ra sao
- phu hop voi cac model co nhieu fully connected layer

### 7.2 Nguyen ly cua `CNN`

`CNN` trong script duoc tao boi [create_cnn_model()](/home/ddragon/myproject/test_NPU/TestNpu.py:213).

Moi block trong `CNN` gom:

1. `Convolution`
2. `Add`
3. `ReLU`
4. `MaxPool`

Neu mo ta bang y nghia:

- `Convolution`:
  truot mot bo loc nho qua anh hoac feature map de trich xuat dac trung
- `Add`:
  cong them bias
- `ReLU`:
  tao phi tuyen
- `MaxPool`:
  giam kich thuoc khong gian, giu lai thong tin noi bat hon

Script tao 3 block lien tiep voi so kenh tang dan:

- `16`
- `32`
- `64`

Y nghia benchmark:

- tao workload gan voi thi giac may tinh hon MLP
- phu hop hon de kiem tra kha nang NPU compile va chay cac op pho bien

### 7.3 Tai sao trong so ngau nhien van benchmark duoc

Trong so trong model nay la ngau nhien, nhung benchmark van co gia tri vi muc tieu o day la:

- do chi phi tinh toan
- do chi phi compile va execute graph

khong phai:

- danh gia do chinh xac
- danh gia chat luong du doan

Noi cach khac:

- benchmark nay do "toc do xu ly graph"
- khong do "model co du doan dung hay khong"

## 8. Tai sao them `hint` va `num_requests`

Ham [benchmark_device()](/home/ddragon/myproject/test_NPU/TestNpu.py:290) compile model voi:

- `PERFORMANCE_HINT = LATENCY` hoac `THROUGHPUT`

Va co the chay voi:

- `num_requests = 1`
- hoac nhieu request song song

Y nghia:

- `LATENCY`: toi uu cho tung request don le
- `THROUGHPUT`: toi uu cho tong so mau xu ly trong mot khoang thoi gian

NPU thuong co co hoi tot hon o:

- `THROUGHPUT`
- `num_requests > 1`
- batch lon hon

## 9. Vai tro cua `batch`

`Batch` la so mau duoc dua vao model trong cung mot lan inference.

Vi du:

- `batch = 1`: moi lan chi xu ly 1 mau
- `batch = 8`: moi lan xu ly 8 mau cung luc

Y nghia:

- batch lon hon thuong lam tong workload lon hon
- accelerator nhu NPU co the co co hoi tan dung tai nguyen tot hon

Nhung doi lai:

- latency cua mot lan infer co the tang
- memory dung nhieu hon

Vi vay:

- batch nho hay hop cho latency
- batch lon thuong hop hon cho throughput

## 10. Tai sao script lay median thay vi mot lan do

Trong [benchmark_device()](/home/ddragon/myproject/test_NPU/TestNpu.py:323), benchmark duoc lap lai nhieu lan qua tham so `--repeats`, sau do lay `median`.

Y nghia:

- Giam anh huong cua nhieu nen:
- scheduler
- cache warm-up
- jitter he thong
- Ket qua on dinh hon so voi chi do mot lan

## 11. CPU va NPU khac nhau the nao

### 8.1 CPU

CPU la bo xu ly da muc dich.

Diem manh:

- linh hoat
- latency thap cho workload nho
- xu ly tot graph nho, request don le, model synthetic don gian

Diem yeu:

- ve lau dai co the ton dien hon
- voi mot so model AI dac thu, throughput co the kem hon accelerator

### 8.2 NPU

NPU la bo xu ly chuyen dung cho neural network.

Diem manh:

- toi uu cho mot so phep tinh AI
- thuong phu hop hon voi throughput, efficiency, va workload duoc compiler ho tro tot

Diem yeu:

- khong phai model nao cung hop
- khong phai luc nao cung nhanh hon CPU
- rat nhay cam voi version runtime, driver, compiler

## 12. Tai sao CPU co the nhanh hon NPU

Ban da thay nhieu ket qua trong do `CPU` nhanh hon `NPU`. Dieu nay khong bat thuong.

Ly do thuong gap:

1. Workload qua nho
   Batch 1, input nho, request sync tung lan.
2. Model synthetic don gian
   CPU vectorized xu ly rat tot.
3. Dang do `LATENCY`
   CPU thuong manh hon o latency tung request.
4. Chua dung nhieu request song song
   NPU thuong de the hien hon trong throughput mode.
5. Khong phai model that
   Model AI that co the co hanh vi rat khac.

Noi cach khac, cau hoi "CPU hay NPU nhanh hon?" khong co dap an chung. Dap an dung phai la:

- nhanh hon tren model nao
- batch nao
- latency hay throughput
- request dong bo hay song song

## 13. Cach doc ket qua benchmark

Trong phan output `BENCHMARK RESULTS`, script in:

- `Median total time`
- `Median latency`
- `Median throughput`
- tung `Run`

Y nghia:

- `Median latency`: thoi gian trung vi cho moi request. Cang thap cang tot.
- `Median throughput`: so request moi giay. Cang cao cang tot.

Neu output in:

```text
CPU latency / NPU latency: 0.29x
```

thi nghia la:

- latency CPU bang 0.29 lan latency NPU
- tuc CPU nhanh hon ve latency

Neu output in:

```text
NPU throughput / CPU throughput: 0.29x
```

thi nghia la:

- throughput NPU chi bang 29% CPU
- tuc NPU chua co loi the tren cau hinh do

### 13.1 Doc ket qua mot cach dung

Neu thay:

- `CPU` co latency thap hon

thi chi co nghia:

- trong cau hinh do, request don le hoac median request cua CPU nhanh hon

Neu thay:

- `NPU` co throughput cao hon

thi co nghia:

- trong cau hinh do, NPU xu ly duoc nhieu request hon moi giay

Hai ket luan nay co the cung dung trong hai mode benchmark khac nhau.

## 14. Cach test de tim xem NPU co nhanh hon CPU khong

Dung mot ma tran benchmark thay vi mot lenh duy nhat.

### 11.1 Test co ban

```bash
python TestNpu.py --model-type cnn
```

Y nghia:

- Day la diem bat dau.
- Thuong CPU se manh hon neu workload con nho.

### 11.2 Test than thien hon voi NPU

```bash
python TestNpu.py --model-type cnn --hint THROUGHPUT --num-requests 4 --batch 4 --iterations 200 --warmup 50
```

Y nghia:

- Tang song song
- Tang tai
- Chuyen sang huong throughput

### 11.3 Test nang hon nua

```bash
python TestNpu.py --model-type cnn --hint THROUGHPUT --num-requests 4 --batch 8 --height 320 --width 320 --iterations 200 --warmup 50
```

Y nghia:

- Neu NPU co co hoi the hien, day la kieu test hop ly hon.

## 15. Cach ket luan dung tu ket qua

Khong nen viet:

- `NPU nhanh hon CPU`
- `CPU nhanh hon NPU`

neu moi chi thu mot cau hinh.

Nen viet theo dang:

- Trong benchmark `cnn`, `batch=1`, `224x224`, `LATENCY`, `CPU` nhanh hon `NPU`.
- Trong benchmark `cnn`, `THROUGHPUT`, `num_requests=4`, `batch=8`, can do tiep de xem NPU co vuot CPU hay khong.

Day la cach ket luan ky thuat dung va khong suy dien qua muc.

## 16. Gioi han cua bai test hien tai

Script nay huu ich de debug va benchmark so bo, nhung van co gioi han:

- model la synthetic
- khong dung model AI that
- chua do dien nang
- chua do do tre dau-cuoi trong ung dung that

Neu muc tieu sau cung la danh gia cho san pham, can benchmark them tren:

- model that
- input that
- batch that
- deployment setting that

## 17. Ket luan

`TestNpu.py` la cong cu dung de:

- xac minh stack NPU co chay duoc hay khong
- so sanh CPU va NPU tren cung mot graph synthetic
- tim xem workload nao co the phu hop hon voi NPU

Gia tri lon nhat cua no khong phai la dua ra "chan ly chung" rang CPU hay NPU nhanh hon, ma la:

- chi ra dieu kien nao CPU manh hon
- chi ra dieu kien nao NPU bat dau co co hoi vuot len
- giup ban dat cau hoi benchmark dung hon cho model that sau nay
