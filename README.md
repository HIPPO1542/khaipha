# Đồ án Khai phá dữ liệu: Gợi ý sản phẩm mua kèm

Sử dụng **hai thuật toán Apriori và FP-Growth** trên **Online Retail (UCI), ID 352** để khai phá luật kết hợp và gợi ý sản phẩm trên web Flask.

## Dữ liệu CHÍNH THỨC (không phải dữ liệu tự tạo)

- Trang UCI: https://archive.ics.uci.edu/dataset/352/online+retail
- Tải trực tiếp (ZIP): https://archive.ics.uci.edu/static/public/352/online+retail.zip
- DOI: https://doi.org/10.24432/C5BW33
- Tên file gốc: `Online Retail.xlsx`, 541.909 dòng chi tiết; 8 cột. Giấy phép CC BY 4.0.
- Nguồn trích: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

**Lưu ý:** Bộ mã nén này KHÔNG đính kèm file Excel UCI gốc do hạn chế tải dữ liệu vào môi trường tạo tài liệu. Bạn tải file thật bằng link trên hoặc dùng `python download_uci.py` trên máy có Internet, rồi đặt ở `data/Online Retail.xlsx`.

## Cách chạy trên Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_uci.py
python src/prepare_data.py --input "data/Online Retail.xlsx" --country "United Kingdom" --top-products 150
python src/mine_rules.py --min-support 0.01 --min-confidence 0.30 --min-lift 1.0
python src/app.py
```

Truy cập http://127.0.0.1:5000. Có thể chọn **Luật Apriori** hoặc **Luật FP-Growth** trên giao diện để so sánh.

## File sinh sau khai phá

- `results/cleaned_transactions.csv`: các dòng chi tiết sau lọc.
- `results/basket_matrix.csv`: ma trận hóa đơn × sản phẩm dạng 0/1.
- `results/product_catalog.csv`: mã hàng, tên, trung vị đơn giá (GBP).
- `results/frequent_apriori.csv`, `results/frequent_fpgrowth.csv`: tập mục phổ biến.
- `results/rules_apriori.csv`, `results/rules_fpgrowth.csv`: luật mua kèm.
- `results/benchmark.json`: số tập phổ biến, số luật, thời gian khai phá tập phổ biến.

## API demo

`GET /api/recommend?items=85123A&model=fpgrowth` (thay mã sản phẩm bằng mã xuất hiện trong `product_catalog.csv`).

## Phạm vi kỹ thuật và hạn chế

- Chỉ đo thời gian **tìm tập mục phổ biến**; không gồm thời gian tiền xử lý và sinh luật. Để so sánh tổng thời gian, cần bổ sung phép đo tương ứng.
- Chọn Top N sản phẩm và bỏ giỏ chỉ còn một mặt hàng làm thay đổi mẫu thử; ghi rõ tham số và số giỏ trước/sau lọc.
- Dữ liệu gồm khách bán sỉ và hàng quà tặng tại Anh (2010–2011), không khẳng định đại diện thị trường Việt Nam.
- Website chỉ là demo giỏ hàng và gợi ý; không có thanh toán, tồn kho hay đăng ký tài khoản.
- Chỉ số Lift không đồng nghĩa quan hệ nhân quả. Gợi ý chỉ được sinh từ những luật có vế trái phù hợp các sản phẩm trong giỏ.
