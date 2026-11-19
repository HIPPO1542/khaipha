"""Tải bộ dữ liệu UCI gốc. Chạy trên máy có internet."""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT=Path(__file__).parent
OUT=ROOT/'data';OUT.mkdir(exist_ok=True)
URL='https://archive.ics.uci.edu/static/public/352/online+retail.zip'
zip_path=OUT/'online_retail_uci_download.zip'
request=Request(URL,headers={'User-Agent':'Mozilla/5.0'})
with urlopen(request,timeout=120) as res, zip_path.open('wb') as dest:
    while chunk:=res.read(1024*1024):dest.write(chunk)
with ZipFile(zip_path) as zf:
    xls=[n for n in zf.namelist() if n.lower().endswith('.xlsx')]
    if not xls:raise RuntimeError('Không tìm thấy XLSX trong ZIP UCI')
    data=zf.read(xls[0])
    (OUT/'Online Retail.xlsx').write_bytes(data)
print('Đã lưu bản gốc: data/Online Retail.xlsx')
