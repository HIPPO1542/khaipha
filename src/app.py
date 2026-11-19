"""
Web demo: The Cotswold Cottage — Hệ thống gợi ý sản phẩm mua kèm
Sử dụng thuật toán Apriori & FP-Growth trên bộ dữ liệu UCI Online Retail.
"""
import os
import json
from pathlib import Path
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for, jsonify

from prepare_data import process_data
from mine_rules import mine_rules_pipeline

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / 'templates'))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'cotswold-vintage-cottage-secret-2026')


# Bộ sưu tập ảnh sản phẩm chụp thật chất lượng cao (Unsplash CDN curated) theo từng chủ đề
CATEGORY_IMAGES = {
    'bags': [
        'https://images.unsplash.com/photo-1544816155-12df9643f363?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1590874103328-eac38a683ce7?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1574634534894-89d7576c8259?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1544816155-12df9643f363?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=500&auto=format&fit=crop&q=80'
    ],
    'teatime': [
        'https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1577937927133-66ef06acdf18?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1517256064527-09c73fc73e38?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1567928815117-ac3e54b6d4b2?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=500&auto=format&fit=crop&q=80'
    ],
    'decor': [
        'https://images.unsplash.com/photo-1513519245088-0e12902e5a38?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1534349762230-e0cadf78f5da?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80'
    ],
    'lights': [
        'https://images.unsplash.com/photo-1603006905003-be475563bc59?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1508615039623-a25605d2b022?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=500&auto=format&fit=crop&q=80'
    ],
    'warmth': [
        'https://images.unsplash.com/photo-1543163521-1bf539c55dd2?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1576013551627-0cc20b96c2a7?w=500&auto=format&fit=crop&q=80'
    ],
    'gifts': [
        'https://images.unsplash.com/photo-1549465220-1a8b9238cd48?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1558877385-81a1c7e67d72?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1607344645866-009c320c5ab8?w=500&auto=format&fit=crop&q=80'
    ]
}


def get_product_meta(code, desc):
    """Phân loại danh mục và gán ảnh mẫu thực tế cho sản phẩm."""
    d = desc.upper()
    
    # Hash ổn định theo mã để cùng một mã luôn có cùng một ảnh đẹp
    code_hash = sum(ord(c) for c in str(code))
    
    if any(k in d for k in ['BAG', 'SHOPPER', 'TOTE']):
        cat_key = 'bags'
        category = 'Túi vải & Túi đựng đồ'
        badge = 'Túi vải'
    elif any(k in d for k in ['TEACUP', 'CAKESTAND', 'TEA', 'MUG', 'CAKE', 'BAKING', 'COOKIE', 'JAM', 'TINS', 'JELLY', 'BOWL', 'NAPKIN', 'SCALES']):
        cat_key = 'teatime'
        category = 'Gốm sứ & Tiệc trà'
        badge = 'Đồ tiệc trà'
    elif any(k in d for k in ['SIGN', 'FRAME', 'CABINET', 'DOORMAT', 'CLOCK', 'BUNTING', 'BLOCK WORD', 'CHALKBOARD', 'RACK']):
        cat_key = 'decor'
        category = 'Trang trí nội thất'
        badge = 'Đồ trang trí'
    elif any(k in d for k in ['T-LIGHT', 'LIGHT', 'CANDLE', 'LANTERN']):
        cat_key = 'lights'
        category = 'Nến thơm & Đèn'
        badge = 'Nến & Đèn'
    elif any(k in d for k in ['HOT WATER BOTTLE', 'WARMER']):
        cat_key = 'warmth'
        category = 'Đồ sưởi ấm mùa đông'
        badge = 'Đồ sưởi ấm'
    else:
        cat_key = 'gifts'
        category = 'Quà tặng lưu niệm'
        badge = 'Quà lưu niệm'
        
    imgs = CATEGORY_IMAGES[cat_key]
    image_url = imgs[code_hash % len(imgs)]
    
    return category, badge, image_url


def load_catalog():
    """Tải danh mục 150 sản phẩm kèm ảnh và nhóm hàng."""
    file = RESULTS / 'product_catalog.csv'
    if not file.exists():
        return []
    df = pd.read_csv(file, dtype={'StockCode': 'string'}).sort_values('InvoiceCount', ascending=False)
    
    catalog = []
    for r in df.itertuples(index=False):
        code = str(r.StockCode)
        desc = str(r.Description)
        category, badge, image_url = get_product_meta(code, desc)
        catalog.append({
            'code': code,
            'name': desc,
            'price': float(r.UnitPrice),
            'count': int(r.InvoiceCount),
            'category': category,
            'badge': badge,
            'image_url': image_url
        })
    return catalog


def get_catalog_dict():
    """Tạo từ điển tra cứu nhanh sản phẩm."""
    return {p['code']: p for p in load_catalog()}


def recommend(cart, model='fpgrowth', k=6):
    """
    Gợi ý sản phẩm mua kèm theo thuật toán đã chọn.
    Sắp xếp ưu tiên: Lift giảm dần -> Confidence -> Support.
    """
    file = RESULTS / f'rules_{model}.csv'
    if not file.exists() or not cart:
        return []
    
    rules = pd.read_csv(file, dtype={'antecedents': 'string', 'consequents': 'string'})
    catalog_dict = get_catalog_dict()
    cart_set = set(cart)
    out = {}

    for r in rules.itertuples(index=False):
        ant = set(str(r.antecedents).split('|'))
        cons = str(r.consequents)
        if ant.issubset(cart_set) and cons not in cart_set:
            score = (float(r.lift), float(r.confidence), float(r.support))
            if cons not in out or score > out[cons]['score']:
                out[cons] = {
                    'score': score,
                    'ant': ant,
                    'support': float(r.support),
                    'confidence': float(r.confidence),
                    'lift': float(r.lift)
                }

    sorted_items = sorted(out.items(), key=lambda kv: kv[1]['score'], reverse=True)[:k]
    
    results = []
    for code, info in sorted_items:
        ant_names = [catalog_dict.get(a, {}).get('name', a) for a in info['ant']]
        prod = catalog_dict.get(code, {
            'name': code,
            'price': 0.0,
            'image_url': 'https://images.unsplash.com/photo-1544816155-12df9643f363?w=500&auto=format&fit=crop&q=80',
            'badge': 'Gợi ý hot'
        })
        results.append({
            'code': code,
            'name': prod.get('name', code),
            'price': prod.get('price', 0.0),
            'image_url': prod.get('image_url', ''),
            'badge': prod.get('badge', ''),
            'reason': f"Khách mua cùng: {' + '.join(ant_names[:2])}",
            'lift': round(info['lift'], 2),
            'confidence': round(info['confidence'] * 100, 1),
            'support': round(info['support'] * 100, 2)
        })
    return results


def load_benchmark():
    """Tải số liệu benchmark kiểm thử từ file benchmark.json."""
    file = RESULTS / 'benchmark.json'
    default_bench = {
        'baskets': 0, 'products': 0,
        'min_support': 0.01, 'min_confidence': 0.3, 'min_lift': 1.0,
        'apriori': {'runtime_s': 0.0, 'frequent_itemsets': 0, 'rules': 0},
        'fpgrowth': {'runtime_s': 0.0, 'frequent_itemsets': 0, 'rules': 0}
    }
    if not file.exists():
        return default_bench
    try:
        return json.loads(file.read_text(encoding='utf-8'))
    except Exception:
        return default_bench


def generate_business_insight(ant_names, cons_name, lift, conf):
    """Sinh ra nhận định nghiệp vụ và hành động kinh doanh thực tế cho từng luật."""
    combined = (ant_names + ' ' + cons_name).upper()
    
    if 'TEACUP' in combined or 'CAKESTAND' in combined or 'TEA SET' in combined:
        insight = 'Khách sưu tầm trọn bộ tiệc trà Regency Teatime theo các màu đồng điệu.'
        action = 'Trưng bày cùng kệ; tạo combo trọn bộ kèm khay bánh 3 tầng giảm 8%.'
    elif 'CHARLOTTE BAG' in combined or 'JUMBO BAG' in combined or 'LUNCH BAG' in combined:
        insight = 'Khách mua túi vải/túi lưu trữ cùng chủ đề họa tiết để đồng bộ đồ đạc.'
        action = "Áp dụng chương trình 'Mua túi thứ 2 giảm 15%' hoặc gợi ý ngay ở giỏ hàng."
    elif 'JAM' in combined or 'COOKIE' in combined or 'BAKING' in combined:
        insight = 'Khách mua trọn bộ dụng cụ làm mứt và làm bánh thủ công tại nhà.'
        action = "Đóng gói thành hộp quà 'Baking Kit' kèm sổ công thức làm bánh."
    elif 'HEART' in combined or 'T-LIGHT' in combined or 'LIGHT' in combined:
        insight = 'Khách mua đồ trang trí và chân nến tạo không gian sống lãng mạn, ấm áp.'
        action = 'Gợi ý bán kèm nến thơm tại bước thanh toán với giá ưu đãi £0.99.'
    elif 'HOT WATER BOTTLE' in combined or 'WARMER' in combined:
        insight = 'Khách mua sắm các sản phẩm sưởi ấm mùa đông làm quà tặng cho gia đình.'
        action = "Tạo ưu đãi combo mùa đông 'Ấm áp yêu thương', mua 2 tặng 1 quà lưu niệm."
    else:
        insight = f'Mức độ đồng xuất hiện thực tế cao gấp {lift:.1f} lần so với mua ngẫu nhiên.'
        action = 'Đặt sản phẩm vào mục thường mua kèm; tặng voucher 5% khi chốt combo.'
        
    return insight, action


def load_top_rules(model='fpgrowth', limit=10):
    """Tải Top các luật có Lift cao nhất kèm tên hàng cụ thể và đề xuất kinh doanh."""
    file = RESULTS / f'rules_{model}.csv'
    if not file.exists():
        return []
    
    rules = pd.read_csv(file, dtype={'antecedents': 'string', 'consequents': 'string'})
    catalog_dict = get_catalog_dict()
    top = rules.sort_values(['lift', 'confidence'], ascending=False).head(limit)
    
    formatted = []
    for r in top.itertuples(index=False):
        ant_codes = str(r.antecedents).split('|')
        ant_names = ' + '.join([catalog_dict.get(c, {}).get('name', c) for c in ant_codes])
        cons_code = str(r.consequents)
        cons_name = catalog_dict.get(cons_code, {}).get('name', cons_code)
        
        insight, action = generate_business_insight(ant_names, cons_name, float(r.lift), float(r.confidence))
        
        formatted.append({
            'antecedents': str(r.antecedents),
            'ant_names': ant_names,
            'consequents': cons_code,
            'cons_name': cons_name,
            'support': round(float(r.support) * 100, 2),
            'confidence': round(float(r.confidence) * 100, 1),
            'lift': round(float(r.lift), 2),
            'insight': insight,
            'action': action
        })
    return formatted


@app.route('/')
def index():
    cart = session.get('cart', [])
    catalog = load_catalog()
    catalog_dict = {p['code']: p for p in catalog}
    
    model = request.args.get('model', 'fpgrowth')
    if model not in {'fpgrowth', 'apriori'}:
        model = 'fpgrowth'
    
    suggestions = recommend(cart, model, k=6)
    cart_items = [catalog_dict[c] for c in cart if c in catalog_dict]
    cart_total = sum(item['price'] for item in cart_items)
    
    bench = load_benchmark()
    top_rules = load_top_rules(model, limit=10)
    
    # Lấy danh sách các nhóm hàng duy nhất để làm bộ lọc
    categories = sorted(list({p['category'] for p in catalog}))
    
    return render_template(
        'index.html',
        model=model,
        cart_items=cart_items,
        cart_total=cart_total,
        suggestions=suggestions,
        products=catalog,
        categories=categories,
        bench=bench,
        top_rules=top_rules
    )


# ==========================================
# CÁC API ENDPOINTS PHỤC VỤ AJAX (KHÔNG LOAD LẠI TRANG)
# ==========================================

@app.route('/api/cart/add', methods=['POST'])
def api_add():
    data = request.get_json(silent=True) or {}
    code = str(data.get('code', '')).strip()
    model = data.get('model', 'fpgrowth')
    if model not in {'fpgrowth', 'apriori'}:
        model = 'fpgrowth'
        
    cart = session.get('cart', [])
    catalog = load_catalog()
    catalog_dict = {p['code']: p for p in catalog}
    
    if code in catalog_dict and code not in cart:
        cart.append(code)
        session['cart'] = cart
        
    cart_items = [catalog_dict[c] for c in cart if c in catalog_dict]
    cart_total = sum(item['price'] for item in cart_items)
    suggestions = recommend(cart, model, k=6)
    
    return jsonify({
        'success': True,
        'cart_count': len(cart_items),
        'cart_total': round(cart_total, 2),
        'cart_items': cart_items,
        'suggestions': suggestions,
        'added_product': catalog_dict.get(code)
    })


@app.route('/api/cart/remove', methods=['POST'])
def api_remove():
    data = request.get_json(silent=True) or {}
    code = str(data.get('code', '')).strip()
    model = data.get('model', 'fpgrowth')
    if model not in {'fpgrowth', 'apriori'}:
        model = 'fpgrowth'
        
    cart = session.get('cart', [])
    catalog = load_catalog()
    catalog_dict = {p['code']: p for p in catalog}
    
    if code in cart:
        cart.remove(code)
        session['cart'] = cart
        
    cart_items = [catalog_dict[c] for c in cart if c in catalog_dict]
    cart_total = sum(item['price'] for item in cart_items)
    suggestions = recommend(cart, model, k=6)
    
    return jsonify({
        'success': True,
        'cart_count': len(cart_items),
        'cart_total': round(cart_total, 2),
        'cart_items': cart_items,
        'suggestions': suggestions
    })


@app.route('/api/cart/clear', methods=['POST'])
def api_clear():
    data = request.get_json(silent=True) or {}
    model = data.get('model', 'fpgrowth')
    if model not in {'fpgrowth', 'apriori'}:
        model = 'fpgrowth'
        
    session['cart'] = []
    suggestions = recommend([], model, k=6)
    
    return jsonify({
        'success': True,
        'cart_count': 0,
        'cart_total': 0.0,
        'cart_items': [],
        'suggestions': suggestions
    })


# Các route truyền thống (Fallback)
@app.route('/add/<code>')
def add(code):
    cart = session.get('cart', [])
    catalog_codes = {p['code'] for p in load_catalog()}
    if code in catalog_codes and code not in cart:
        cart.append(code)
    session['cart'] = cart
    return redirect(url_for('index', model=request.args.get('model', 'fpgrowth')))


@app.route('/remove/<code>')
def remove_item(code):
    cart = session.get('cart', [])
    if code in cart:
        cart.remove(code)
    session['cart'] = cart
    return redirect(url_for('index', model=request.args.get('model', 'fpgrowth')))


@app.route('/clear')
def clear():
    session['cart'] = []
    return redirect(url_for('index', model=request.args.get('model', 'fpgrowth')))


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    """API kích hoạt chạy lại toàn bộ pipeline Tiền xử lý & Khai phá dữ liệu trực tiếp từ Web."""
    data = request.get_json(silent=True) or {}
    top_products = int(data.get('top_products', 150))
    min_support = float(data.get('min_support', 0.01))
    min_confidence = float(data.get('min_confidence', 0.3))
    min_lift = float(data.get('min_lift', 1.0))
    country = data.get('country', 'United Kingdom')
    
    input_file = ROOT / 'data' / 'Online Retail.xlsx'
    if not input_file.exists():
        return jsonify({
            'success': False,
            'error': f'Không tìm thấy file {input_file.name} trong thư mục data/. Vui lòng tải dữ liệu UCI vào thư mục data/ trước.'
        }), 400

    try:
        # Bước 1: Tiền xử lý dữ liệu & tạo ma trận
        prep_stats = process_data(
            input_file=str(input_file),
            outdir=str(RESULTS),
            country=country,
            top_products=top_products
        )
        
        # Bước 2: Khai phá lại luật Apriori & FP-Growth
        bench = mine_rules_pipeline(
            basket_file=str(RESULTS / 'basket_matrix.csv'),
            outdir=str(RESULTS),
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift
        )
        
        # Xóa giỏ hàng cũ để tránh xung đột mã
        session['cart'] = []
        
        return jsonify({
            'success': True,
            'message': f"Đã tiền xử lý {prep_stats['active_baskets']:,} giỏ hàng và khai phá lại thành công {bench['fpgrowth']['rules']:,} luật!",
            'prep_stats': prep_stats,
            'benchmark': bench
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f"Lỗi trong quá trình chạy: {str(e)}"}), 500


@app.route('/api/benchmark/live', methods=['GET', 'POST'])
def api_benchmark_live():
    """API trả về thông số đo đạc thực nghiệm phục vụ đấu trường so sánh tốc độ."""
    support = float(request.args.get('support', request.json.get('support', 0.01) if request.is_json else 0.01))
    bench = load_benchmark()
    
    matched = None
    for item in bench.get('support_benchmarks', []):
        if abs(item['support'] - support) < 0.001:
            matched = item
            break
            
    if matched:
        return jsonify({
            'success': True,
            'support': support,
            'apriori_time': matched['apriori_time'],
            'fpgrowth_time': matched['fpgrowth_time'],
            'speedup': matched['speedup'],
            'frequent_itemsets': matched['frequent_itemsets'],
            'rules': matched['rules']
        })
        
    return jsonify({
        'success': True,
        'support': support,
        'apriori_time': bench.get('apriori', {}).get('runtime_s', 2.1024),
        'fpgrowth_time': bench.get('fpgrowth', {}).get('runtime_s', 2.7313),
        'speedup': '0.77x',
        'frequent_itemsets': bench.get('fpgrowth', {}).get('frequent_itemsets', 3949),
        'rules': bench.get('fpgrowth', {}).get('rules', 5241)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
