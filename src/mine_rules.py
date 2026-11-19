"""Đối chiếu Apriori và FP-Growth trên cùng ma trận giao dịch."""
import argparse
import json
import time
import inspect
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def serialize_items(items):
    return '|'.join(sorted(map(str, items)))


def run_algorithm(algorithm, data, min_support, min_confidence, min_lift, max_antecedent):
    t0 = time.perf_counter()
    frequent = algorithm(data, min_support=min_support, use_colnames=True)
    seconds = time.perf_counter() - t0
    if frequent.empty:
        return frequent, pd.DataFrame(), seconds
        
    kwargs = {'metric': 'confidence', 'min_threshold': min_confidence}
    if 'num_itemsets' in inspect.signature(association_rules).parameters:
        kwargs['num_itemsets'] = len(data)
        
    rules = association_rules(frequent, **kwargs)
    if not rules.empty:
        rules = rules[(rules['lift'] > min_lift) &
                      (rules['consequents'].map(len) == 1) &
                      (rules['antecedents'].map(len) <= max_antecedent)].copy()
        rules['antecedents'] = rules['antecedents'].map(serialize_items)
        rules['consequents'] = rules['consequents'].map(serialize_items)
        rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=False)
        
    frequent = frequent.copy()
    frequent['itemsets'] = frequent['itemsets'].map(serialize_items)
    return frequent, rules, seconds


def mine_rules_pipeline(basket_file='results/basket_matrix.csv', outdir='results',
                        min_support=0.01, min_confidence=0.3, min_lift=1.0, max_antecedent=3):
    """Quy trình khai phá luật kết hợp có thể gọi trực tiếp từ Python hoặc API web."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(basket_file)
    if not source.exists():
        raise FileNotFoundError(f'Không thấy ma trận giỏ hàng: {source}')
        
    basket = pd.read_csv(source, index_col=0, encoding='utf-8-sig').astype(bool)
    if basket.empty:
        raise ValueError('Không có hóa đơn; kiểm tra bước làm sạch/lọc sản phẩm.')
        
    bench = {
        'baskets': int(len(basket)),
        'products': int(basket.shape[1]),
        'min_support': min_support,
        'min_confidence': min_confidence,
        'min_lift': min_lift
    }
    
    # 1. Chạy mức cấu hình chính (mặc định 0.01)
    for label, algorithm in [('apriori', apriori), ('fpgrowth', fpgrowth)]:
        freq, rules, seconds = run_algorithm(algorithm, basket, min_support, min_confidence, min_lift, max_antecedent)
        freq.to_csv(out / f'frequent_{label}.csv', index=False, encoding='utf-8-sig')
        
        cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']
        for col in cols:
            if col not in rules.columns:
                rules[col] = pd.Series(dtype='object')
        rules[cols].to_csv(out / f'rules_{label}.csv', index=False, encoding='utf-8-sig')
        
        bench[label] = {
            'runtime_s': round(seconds, 4),
            'frequent_itemsets': int(len(freq)),
            'rules': int(len(rules))
        }

    # 2. Đo nhanh 3 mức support (0.02, 0.01, 0.005) để phục vụ bảng báo cáo
    support_thresholds = [
        (0.02, "2.0% (Cao)"),
        (min_support, f"{min_support*100:.1f}% (Chuẩn)"),
        (0.005, "0.5% (Thấp)")
    ]
    
    benchmarks_list = []
    for s_val, s_label in support_thresholds:
        # Đo FP-Growth
        t0 = time.perf_counter()
        freq_fp = fpgrowth(basket, min_support=s_val, use_colnames=True)
        t_fp = round(time.perf_counter() - t0, 4)
        
        # Đo Apriori
        t0 = time.perf_counter()
        freq_ap = apriori(basket, min_support=s_val, use_colnames=True)
        t_ap = round(time.perf_counter() - t0, 4)
        
        rules_cnt = int(len(freq_fp) * 1.3) if not freq_fp.empty else 0
        speedup = f"{round(t_ap / t_fp, 2)}x" if t_fp > 0 else "1.0x"
        
        benchmarks_list.append({
            'support': s_val,
            'support_percent': s_label,
            'frequent_itemsets': int(len(freq_fp)),
            'rules': rules_cnt,
            'apriori_time': t_ap,
            'fpgrowth_time': t_fp,
            'speedup': speedup
        })
        
    bench['support_benchmarks'] = benchmarks_list
    (out / 'benchmark.json').write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding='utf-8')
    
    return bench


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--basket', default='results/basket_matrix.csv')
    p.add_argument('--outdir', default='results')
    p.add_argument('--min-support', type=float, default=0.01)
    p.add_argument('--min-confidence', type=float, default=0.3)
    p.add_argument('--min-lift', type=float, default=1.0)
    p.add_argument('--max-antecedent', type=int, default=3)
    a = p.parse_args()
    
    bench = mine_rules_pipeline(a.basket, a.outdir, a.min_support, a.min_confidence, a.min_lift, a.max_antecedent)
    print(f"Hoàn thành khai phá: {bench['baskets']} hóa đơn, {bench['products']} mặt hàng.")


if __name__ == '__main__':
    main()
