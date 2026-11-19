"""Làm sạch hóa đơn và tạo ma trận giỏ hàng cho Apriori/FP-Growth."""
import argparse
from pathlib import Path
import pandas as pd

COLS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']


def process_data(input_file='data/Online Retail.xlsx', outdir='results', country='United Kingdom', top_products=150):
    """Hàm tiền xử lý dữ liệu có thể gọi trực tiếp từ Python hoặc API web."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(input_file)
    if not source.exists():
        raise FileNotFoundError(f'Không thấy dữ liệu: {source}. Hãy tải bản gốc Online Retail.xlsx vào thư mục data/')
    
    if source.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(source, dtype={'InvoiceNo': 'string', 'StockCode': 'string'})
    else:
        df = pd.read_csv(source, dtype={'InvoiceNo': 'string', 'StockCode': 'string'}, encoding='utf-8-sig')
        
    missing = set(COLS) - set(df.columns)
    if missing:
        raise ValueError(f'File thiếu các cột bắt buộc: {sorted(missing)}')
        
    original = len(df)
    df = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']).copy()
    for col in ('InvoiceNo', 'StockCode', 'Description', 'Country'):
        df[col] = df[col].astype('string').str.strip()
        
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df = df[df['InvoiceNo'].ne('') & df['StockCode'].ne('') & df['Description'].ne('')]
    df = df[~df['InvoiceNo'].str.upper().str.startswith('C', na=False)]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    
    # Loại mã phi hàng hóa
    df = df[~df['StockCode'].str.upper().isin({'POST', 'DOT', 'M', 'BANK CHARGES', 'C2', 'AMAZONFEE'})]
    if country:
        df = df[df['Country'].eq(country)]
        
    df.to_csv(out / 'cleaned_transactions.csv', index=False, encoding='utf-8-sig')

    counts = df.groupby('StockCode')['InvoiceNo'].nunique().sort_values(ascending=False)
    top_codes = counts.head(top_products).index
    selected = df[df['StockCode'].isin(top_codes)]
    
    matrix = (selected.assign(Flag=1)
              .pivot_table(index='InvoiceNo', columns='StockCode', values='Flag', aggfunc='max', fill_value=0)
              .astype(bool))
    matrix = matrix.loc[matrix.sum(axis=1) >= 2]
    
    matrix.astype('int8').to_csv(out / 'basket_matrix.csv', encoding='utf-8-sig')
    
    names = (selected.groupby('StockCode')['Description']
             .agg(lambda x: x.value_counts().index[0]))
    prices = selected.groupby('StockCode')['UnitPrice'].median().round(2)
    catalog = pd.DataFrame({
        'StockCode': names.index,
        'Description': names.values,
        'UnitPrice': prices.reindex(names.index).values,
        'InvoiceCount': counts.reindex(names.index).values
    })
    catalog.to_csv(out / 'product_catalog.csv', index=False, encoding='utf-8-sig')
    
    return {
        'original_rows': original,
        'cleaned_rows': len(df),
        'active_baskets': len(matrix),
        'selected_products': matrix.shape[1]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/Online Retail.xlsx')
    p.add_argument('--outdir', default='results')
    p.add_argument('--country', default=None, help='VD: United Kingdom; bỏ trống để lấy tất cả')
    p.add_argument('--top-products', type=int, default=150)
    a = p.parse_args()
    
    stats = process_data(a.input, a.outdir, a.country, a.top_products)
    print(f"Input rows={stats['original_rows']}; cleaned rows={stats['cleaned_rows']}; "
          f"active baskets={stats['active_baskets']}; selected products={stats['selected_products']}")


if __name__ == '__main__':
    main()
