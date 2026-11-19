# BÁO CÁO NỘI DUNG ĐỒ ÁN KHAI PHÁ DỮ LIỆU
## ĐỀ TÀI: XÂY DỰNG ỨNG DỤNG GỢI Ý SẢN PHẨM MUA KÈM
**Phương pháp:** Khai phá luật kết hợp bằng thuật toán Apriori và FP-Growth  
**Bộ dữ liệu thực nghiệm:** Online Retail (UCI Machine Learning Repository, ID: 352)

---

## 1. TỔNG QUAN VỀ BỘ DỮ LIỆU THỰC NGHIỆM

### 1.1. Nguồn gốc và thông tin xác thực
* **Tên bộ dữ liệu:** Online Retail
* **Đơn vị công bố:** UCI Machine Learning Repository (Mã định danh: 352)
* **Tác giả:** Daqing Chen (2015)
* **Thời gian giao dịch:** Từ ngày 01/12/2010 đến ngày 09/12/2011 (toàn bộ 1 năm hoạt động bán lẻ)
* **Đặc thù dữ liệu:** Dữ liệu hóa đơn bán hàng thực tế của một doanh nghiệp bán lẻ trực tuyến có trụ sở tại Vương quốc Anh, chuyên kinh doanh đồ quà tặng thủ công, gốm sứ và đồ trang trí nội thất gia đình. Khách hàng bao gồm cả người mua lẻ cá nhân và các đại lý mua sỉ.
* **Bản quyền sử dụng:** Creative Commons Attribution 4.0 (CC BY 4.0)
* **Trích dẫn chuẩn:**  
  *Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33*

### 1.2. Cấu trúc các trường dữ liệu ban đầu
File dữ liệu gốc `Online Retail.xlsx` bao gồm **541.909 dòng giao dịch** với **8 trường thông tin**:

| Tên trường | Kiểu dữ liệu | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- |
| `InvoiceNo` | Chuỗi ký tự | Số hóa đơn gồm 6 chữ số. Nếu bắt đầu bằng chữ 'C' là hóa đơn hủy giao dịch |
| `StockCode` | Chuỗi ký tự | Mã định danh sản phẩm hàng hóa |
| `Description` | Chuỗi ký tự | Tên và mô tả chi tiết của sản phẩm |
| `Quantity` | Số nguyên | Số lượng sản phẩm mua trong từng dòng hóa đơn |
| `InvoiceDate` | Ngày giờ | Thời gian tạo hóa đơn bán hàng |
| `UnitPrice` | Số thực | Đơn giá sản phẩm (tính bằng Bảng Anh - GBP) |
| `CustomerID` | Số nguyên | Mã định danh khách hàng (có thể bị khuyết thiếu) |
| `Country` | Chuỗi ký tự | Tên quốc gia nơi khách hàng sinh sống |

---

## 2. QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU VÀ THỐNG KÊ BIẾN ĐỘNG

Trong khai phá giỏ hàng, việc làm sạch dữ liệu là điều kiện tiên quyết để thuật toán chạy chính xác và tránh sinh ra các luật rác. Quy trình xử lý gồm hai giai đoạn chính:

### 2.1. Các bước làm sạch dữ liệu
1. **Loại bỏ dữ liệu thiếu sót:** Loại bỏ tất cả các dòng bị rỗng ở các trường bắt buộc: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `UnitPrice`.
2. **Loại bỏ giao dịch bất thường:**
   * Loại bỏ các hóa đơn có mã `InvoiceNo` bắt đầu bằng ký tự `C` (Cancelled - đơn hoàn trả hoặc đơn bị hủy).
   * Loại bỏ các dòng có số lượng `Quantity ≤ 0` hoặc đơn giá `UnitPrice ≤ 0` (hàng mẫu 0 đồng, dòng kiểm kê tồn kho bị lỗi).
3. **Loại bỏ mã phi hàng hóa:** 
   * Bộ dữ liệu chứa một số mã tính cước dịch vụ như: `POST` (cước bưu điện), `DOT` (phí giao hàng), `M` (điều chỉnh thủ công), `BANK CHARGES` (phí ngân hàng), `AMAZONFEE` (phí sàn), `C2`. 
   * Các mã này là chi phí phát sinh, không phải sản phẩm tiêu dùng nên phải loại bỏ để không làm sai lệch luật mua kèm.
4. **Phân vùng thị trường:** Giữ lại các giao dịch tại thị trường chính là Vương quốc Anh (`Country = United Kingdom`) chiếm hơn 90% dữ liệu, nhằm đảm bảo tính đồng nhất về hành vi tiêu dùng của khách hàng nội địa.

### 2.2. Biến đổi cấu trúc sang ma trận giỏ hàng
Thuật toán Apriori và FP-Growth không nhận đầu vào dạng bảng giao dịch chi tiết mà yêu cầu dạng **ma trận giỏ hàng nhị phân**:
* **Lọc Top 150 sản phẩm bán chạy nhất:** Chọn ra 150 mã sản phẩm có số lượng hóa đơn xuất hiện nhiều nhất trong tập dữ liệu. Điều này giúp tập trung vào các mặt hàng chủ lực, đồng thời ngăn chặn hiện tượng bùng nổ tổ hợp làm quá tải bộ nhớ khi chạy Apriori.
* **Xây dựng One-Hot Matrix:** Gom nhóm theo mã hóa đơn, chuyển đổi thành bảng nhị phân:
  * Mỗi hàng là một mã hóa đơn.
  * Mỗi cột là một mã sản phẩm trong Top 150.
  * Giá trị là `1` nếu hóa đơn có chứa sản phẩm đó, và `0` nếu không mua.
* **Lọc giỏ hàng hợp lệ:** Chỉ giữ lại những hóa đơn chứa **từ 2 sản phẩm trở lên** thuộc danh mục 150 sản phẩm trên (vì hóa đơn chỉ có 1 món thì không thể khai phá được hành vi mua kèm).

### 2.3. Bảng thống kê biến động dữ liệu qua các bước

| Giai đoạn xử lý | Số dòng / Số giỏ | Tỷ lệ giữ lại | Ý nghĩa thực tế |
| :--- | :---: | :---: | :--- |
| **Dữ liệu thô ban đầu** | 541.909 dòng | 100.0% | Toàn bộ giao dịch trong file gốc từ UCI |
| **Sau khi làm sạch** | 484.041 dòng | 89.32% | Đã loại bỏ 57.868 dòng lỗi, đơn hủy và mã cước phí |
| **Danh mục sản phẩm chọn lọc** | 150 mặt hàng | - | Nhóm sản phẩm chủ lực có tần suất xuất hiện cao nhất |
| **Ma trận giỏ hàng hợp lệ** | **12.851 giỏ hàng** | - | Số hóa đơn thực tế đưa vào huấn luyện mô hình |

---

## 3. CƠ SỞ LÝ THUYẾT VÀ CÁC CHỈ SỐ ĐÁNH GIÁ LUẬT KẾT HỢP

Một luật kết hợp có dạng: **Tiền đề (Vế trước A) $\rightarrow$ Kết luận (Vế sau B)**  
Ý nghĩa: *"Khách hàng đã mua tập sản phẩm A thì có xu hướng mua kèm thêm sản phẩm B"*.

### 3.1. Chỉ số Support (Độ hỗ trợ)
* **Khái niệm:** Đo lường mức độ phổ biến hoặc tần suất xuất hiện đồng thời của cả A và B trong toàn bộ các giỏ hàng của cửa hàng.
* **Công thức tính:**
  $$\text{Support}(A \rightarrow B) = \frac{\text{Số giỏ hàng chứa cả A và B}}{\text{Tổng số giỏ hàng}}$$
* **Ý nghĩa thực tế:** Dùng để lọc bỏ các trường hợp mua kèm ngẫu nhiên hoặc quá hiếm gặp. Một luật muốn có giá trị kinh doanh thì phải có đủ số lượng khách hàng thực hiện hành vi đó.
* **Ngưỡng sử dụng trong đề tài:** `min_support = 0.01` (1% tổng số giỏ hàng, tương đương xuất hiện trong ít nhất 128 hóa đơn).

### 3.2. Chỉ số Confidence (Độ tin cậy)
* **Khái niệm:** Đo lường xác suất một khách hàng sẽ mua thêm sản phẩm B khi họ đã bỏ sản phẩm A vào giỏ hàng.
* **Công thức tính:**
  $$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = \frac{\text{Số giỏ hàng chứa cả A và B}}{\text{Số giỏ hàng chứa A}}$$
* **Ý nghĩa thực tế:** Cho biết độ chắc chắn của lời gợi ý. Nếu Confidence đạt 70%, có nghĩa là cứ 10 người mua món A thì có tới 7 người đồng ý mua thêm món B.
* **Ngưỡng sử dụng trong đề tài:** `min_confidence = 0.30` (30%).

### 3.3. Chỉ số Lift (Độ nâng)
* **Khái niệm:** Đo lường mức độ tương quan thực sự giữa việc mua A và mua B, so sánh xác suất mua kèm thực tế với trường hợp hai sản phẩm được mua hoàn toàn độc lập ngẫu nhiên.
* **Công thức tính:**
  $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
* **Thang đánh giá giá trị của Lift:**
  * **$\text{Lift} = 1$:** A và B hoàn toàn độc lập. Khách mua A không làm tăng khả năng mua B. Việc mua cùng nhau chỉ là ngẫu nhiên thuần túy.
  * **$\text{Lift} < 1$:** Tương quan âm (bài trừ lẫn nhau). Mua sản phẩm này làm giảm khả năng mua sản phẩm kia.
  * **$\text{Lift} > 1$:** Tương quan dương. Mua A làm tăng khả năng mua B. Giá trị Lift càng cao chứng minh mối liên kết giữa 2 sản phẩm càng mạnh.
* **Ngưỡng sử dụng trong đề tài:** `min_lift > 1.0`.

---

## 4. TIÊU CHÍ SO SÁNH VÀ KẾT QUẢ THỰC NGHIỆM: APRIORI VÀ FP-GROWTH

Thực nghiệm được thực hiện trên cùng một ma trận giỏ hàng (12.851 giỏ hàng × 150 mặt hàng) nhằm đảm bảo tính công bằng và chính xác tuyệt đối.

### 4.1. Bảng so sánh bản chất kỹ thuật giữa hai thuật toán

| Tiêu chí so sánh | Thuật toán Apriori | Thuật toán FP-Growth |
| :--- | :--- | :--- |
| **Chiến lược duyệt** | Duyệt theo chiều rộng (BFS), duyệt theo từng tầng kích thước tập mục | Duyệt theo chiều sâu (DFS), áp dụng phương pháp chia để trị |
| **Cấu trúc dữ liệu** | Sử dụng ma trận phẳng, sinh mảng các tập ứng viên | Nén toàn bộ cơ sở dữ liệu vào cây FP-Tree kèm bảng Header Table |
| **Số lần quét cơ sở dữ liệu** | Quét lặp nhiều lần (mỗi lần tăng kích thước tập mục đều phải quét lại toàn bộ dữ liệu) | Chỉ quét cơ sở dữ liệu đúng 2 lần duy nhất |
| **Sinh tập ứng viên (Candidate Generation)** | **Bắt buộc sinh ứng viên.** Tốn chi phí tính toán tổ hợp cực lớn khi dữ liệu mở rộng | **Hoàn toàn không sinh ứng viên.** Trực tiếp khai phá từ các nhánh cây FP-Tree điều kiện |
| **Sử dụng bộ nhớ** | Tốn ít bộ nhớ ban đầu nhưng tốn bộ nhớ trung gian để lưu mảng ứng viên | Tốn bộ nhớ RAM để lưu cấu trúc cây FP-Tree nén |
| **Khả năng mở rộng (Scalability)** | Kém khi dữ liệu lớn hoặc khi giảm ngưỡng hỗ trợ | Vượt trội và ổn định trên các tập dữ liệu giao dịch quy mô lớn |

---

### 4.2. Kết quả thực nghiệm đa ngưỡng Support (0.5% • 1.0% • 2.0%)

Nhằm đánh giá sức chịu tải của hai thuật toán khi số lượng tổ hợp tập mục bùng nổ, hệ thống thực nghiệm trên 3 mức Support khác nhau:

| Ngưỡng Support | Số Frequent Itemsets | Số Association Rules | Thời gian Apriori | Thời gian FP-Growth | Độ tăng tốc (Speedup) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2.0% (Mức cao)** | 582 tập | 468 luật | 0.2842 giây | 0.1915 giây | FP-Growth nhanh hơn **1.48 lần** |
| **1.0% (Mức chuẩn)** | 3.949 tập | 5.241 luật | 1.0306 giây | 0.7851 giây | FP-Growth nhanh hơn **1.31 lần** |
| **0.5% (Mức thấp)** | 18.450 tập | 24.120 luật | 4.8215 giây | 1.6152 giây | FP-Growth nhanh hơn **2.98 lần** |

### 4.3. Đánh giá và nhận xét khoa học
1. **Bảo toàn kết quả toán học (Output Consistency):** Cả hai thuật toán đều cho ra số lượng tập mục phổ biến và số lượng luật kết hợp **trùng khớp 100%** trên mọi ngưỡng thử nghiệm. Điều này khẳng định tính đúng đắn về mặt logic của việc cài đặt thuật toán.
2. **Hiệu năng khi tổ hợp bùng nổ:**
   * Ở ngưỡng Support cao (2.0%), sự khác biệt giữa hai thuật toán chưa thực sự rõ rệt vì số lượng tập ứng viên ít.
   * Khi hạ Support xuống mức thấp (0.5%), số tập mục phổ biến tăng từ 582 lên 18.450 tập. Lúc này, Apriori bắt đầu lộ rõ nhược điểm do phải sinh và kiểm tra hàng chục ngàn tập ứng viên, khiến thời gian chạy tăng vọt từ 0.28s lên 4.82s (tăng gấp 17 lần).
   * Trong khi đó, FP-Growth duy trì tốc độ rất tốt (chỉ 1.61s), chứng minh ưu thế vượt trội của cấu trúc cây nén FP-Tree trong việc xử lý khai phá dữ liệu lớn.

---

## 5. PHÂN TÍCH TOP 10 LUẬT MUA KÈM MẠNH NHẤT VÀ GIẢI PHÁP KINH DOANH

Dưới đây là danh sách 10 luật có chỉ số Lift cao nhất được trích xuất từ dữ liệu thực tế:

| STT | Sản phẩm tiền đề (Vế trước A) | ➔ | Sản phẩm mua kèm (Vế sau B) | Support | Confidence | Lift | Phân tích hành vi & Đề xuất kinh doanh |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Đèn ngủ nấm LED + Bộ in làm mứt | ➔ | Túi đeo vai Suki | 1.03% | 71.1% | **18.17x** | **Hành vi:** Mua quà tặng lưu niệm kèm đồ trang trí nhà cửa.<br>**Đề xuất:** Tạo combo quà tặng trọn gói, giảm 5% khi mua kèm. |
| **2** | Túi tái chế chấm bi + Bộ in làm mứt | ➔ | Túi đeo vai Suki | 1.04% | 68.4% | **17.47x** | **Hành vi:** Mua đồ làm bếp kết hợp phụ kiện túi vải đựng đồ.<br>**Đề xuất:** Trưng bày cạnh kệ hàng dụng cụ nhà bếp. |
| **3** | Túi Jumbo quả táo + Túi lá vintage + Túi ren vintage | ➔ | Túi Jumbo quả lê | 1.39% | 73.6% | **17.31x** | **Hành vi:** Khách mua sắm trọn bộ túi lưu trữ đồng kích cỡ theo các mẫu hoa quả.<br>**Đề xuất:** Áp dụng chương trình mua 3 tặng 1 hoặc giảm 15% cho túi tiếp theo. |
| **4** | Túi dâu tây Charlotte + Túi Suki Charlotte + Khay bánh Regency | ➔ | Túi rừng cây Woodland Charlotte | 1.09% | 90.9% | **17.21x** | **Hành vi:** Khách hàng sưu tầm đầy đủ các họa tiết của dòng túi Charlotte Bag.<br>**Đề xuất:** Gợi ý ngay tại giỏ hàng thông điệp 'Hoàn thiện bộ sưu tập túi Charlotte'. |
| **5** | Túi dâu tây Charlotte + Túi chấm bi hồng Charlotte + Khay bánh Regency | ➔ | Túi rừng cây Woodland Charlotte | 1.00% | 88.4% | **16.72x** | **Hành vi:** Tương tự nhóm sưu tầm túi Charlotte đi kèm phụ kiện bàn trà.<br>**Đề xuất:** Thiết lập tính năng gợi ý mua kèm tự động trên web. |
| **6** | Túi Suki Charlotte + Túi chấm bi hồng Charlotte + Khay bánh Regency | ➔ | Túi rừng cây Woodland Charlotte | 1.03% | 88.1% | **16.67x** | **Hành vi:** Thói quen mua sắm theo bộ của khách hàng mua quà tặng.<br>**Đề xuất:** Đóng gói set combo quà tặng sẵn. |
| **7** | Túi dâu tây Charlotte + Túi chấm bi đỏ Charlotte + Khay bánh Regency | ➔ | Túi rừng cây Woodland Charlotte | 1.09% | 88.1% | **16.66x** | **Hành vi:** Sự gắn kết cao giữa các biến thể màu sắc của cùng dòng sản phẩm.<br>**Đề xuất:** Đặt các biến thể màu cạnh nhau trên gian hàng. |
| **8** | Hộp giấy cupcake chấm bi + Túi Suki Charlotte + Khay bánh Regency | ➔ | Túi rừng cây Woodland Charlotte | 1.03% | 87.5% | **16.56x** | **Hành vi:** Mua đồ chuẩn bị cho tiệc sinh nhật hoặc dã ngoại.<br>**Đề xuất:** Bán chéo đồ làm bánh kèm túi đựng đồ dã ngoại. |
| **9** | Hộp giấy cupcake chấm bi + Túi Suki Charlotte + Túi chấm bi hồng Charlotte | ➔ | Túi rừng cây Woodland Charlotte | 1.25% | 87.4% | **16.55x** | **Hành vi:** Nhóm khách hàng gia đình chuẩn bị tiệc ngọt.<br>**Đề xuất:** Tặng voucher giảm giá cho danh mục phụ kiện tiệc ngọt. |
| **10** | Tách trà xanh Regency | ➔ | Tách trà hồng Regency | 1.45% | 71.5% | **16.12x** | **Hành vi:** Khách mua tách trà quý tộc phong cách Anh thường gom đủ cặp đôi màu xanh và hồng.<br>**Đề xuất:** Giảm ngay 10% khi mua theo cặp tách trà đôi. |

---

## 6. KIẾN TRÚC HỆ THỐNG VÀ ỨNG DỤNG THỰC TẾ TRÊN WEBSITE

Hệ thống được thiết kế theo kiến trúc chuẩn gồm hai tầng độc lập:

```
[ Dữ liệu thô Online Retail.xlsx ]
             │
             ▼ (Tầng 1: Khai phá dữ liệu ngoại tuyến)
[ Tiền xử lý dữ liệu: prepare_data.py ] ──► Xuất ma trận basket_matrix.csv
             │
             ▼
[ Khai phá luật kết hợp: mine_rules.py ] ──► Chạy Apriori & FP-Growth
             │
             ├──────────────────────────► Xuất kết quả rules_*.csv
             └──────────────────────────► Xuất thông số benchmark.json
             │
             ▼ (Tầng 2: Vận hành trực tuyến)
[ Ứng dụng Web Flask: app.py ]
             │
             ├─ Quản lý giỏ hàng & Gợi ý sản phẩm mua kèm thời gian thực
             ├─ Bảng điều khiển đối sánh hiệu năng thuật toán
             └─ Nút kích hoạt tự động chạy lại toàn bộ quy trình khi có dữ liệu mới
```

### 6.1. Tầng khai phá ngoại tuyến
* Chịu trách nhiệm xử lý các tác vụ tính toán nặng: làm sạch nửa triệu dòng giao dịch, xây dựng ma trận One-Hot và chạy hai thuật toán khai phá.
* Kết quả sau khi tính toán được lưu trữ thành các tệp dữ liệu trung gian có cấu trúc (`rules_fpgrowth.csv`, `rules_apriori.csv`, `product_catalog.csv`, `benchmark.json`).

### 6.2. Tầng ứng dụng trực tuyến
* Xây dựng trên nền tảng Flask và giao diện web hiện đại.
* **Thời gian đáp ứng siêu tốc:** Website chỉ đọc các luật đã được tính toán sẵn từ tệp kết quả, cho phép trả về các sản phẩm gợi ý mua kèm trong vòng chưa tới 0.01 giây, không gây nghẽn máy chủ.
* **Tích hợp nút cập nhật tự động:** Cung cấp tính năng kích hoạt chạy lại toàn bộ quy trình ngay trên giao diện web khi có dữ liệu mới phát sinh, hoàn toàn tự động hóa mà không cần thao tác qua dòng lệnh.
