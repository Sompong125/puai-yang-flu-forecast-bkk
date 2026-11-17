"""ขั้นที่ 2: วิเคราะห์และสร้างโมเดล (R3 EDA, R4 baseline เทียบโมเดล, R5 จุดที่พลาด)
อ่าน data/processed/flu_district_week_2025_2026.csv -> กราฟ/ตารางผลลัพธ์ใน results/
แบ่ง train/test ตามเวลา (ล็อกไว้ก่อนรันโมเดล): test = 6 ก.ค.–19 ต.ค. 2026
(โค้ดตัดมาจาก notebook ส่วน R3–R5 โดยไม่เปลี่ยนตรรกะ)
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from utils import *
install_show('analysis')

# ค่าคงที่ที่ล็อกไว้ (เหมือน notebook)
TEST_START = pd.Timestamp('2026-07-06')
SUSPECT_WEEKS = [pd.Timestamp('2025-12-22'), pd.Timestamp('2025-12-29')]   # รายงานไม่ครบปลายไฟล์ 2025
AGE_LABELS = ['age_0_4', 'age_5_14', 'age_15_24', 'age_25_59', 'age_60plus']

csv_path = PROC_DIR / 'flu_district_week_2025_2026.csv'
if not csv_path.exists():
    raise SystemExit(f'ไม่พบ {csv_path} : รัน src/preprocess.py ก่อน (หรือ python src/main.py)')
full = pd.read_csv(csv_path, parse_dates=['week_start'])
print('โหลดตาราง', csv_path.name, full.shape)

# ===== จาก notebook cell 36 =====
tab = full.copy()    # ตารางเขต × สัปดาห์จาก R2
p_all = tab.pivot(index='district', columns='week_start', values='cases')   # แถว = เขต, คอลัมน์ = สัปดาห์
tot = p_all.sum()                                                            # ผู้ป่วยรวมทั้ง กทม. รายสัปดาห์

# ===== จาก notebook cell 38 =====
fig, ax = plt.subplots(figsize=(10, 4.3))
ax.axvspan(TEST_START, tot.index[-1] + pd.Timedelta(days=6), color=GRID, alpha=0.7, lw=0)
ax.plot(tot.index, tot.values, color=BLUE, lw=2, marker='o', ms=4)
sus = tot[SUSPECT_WEEKS]
ax.plot(sus.index, sus.values, 'o', ms=9, mfc='none', mec='#d9622b', mew=2)
ax.text(sus.index[0], sus.values[0] - 900, T('รายงานไม่ครบ', 'under-reported'), color='#b8461c', fontsize=9, ha='center')

peak1 = tot[:'2025-12-15'].idxmax()                                        # ระลอก 1
low = tot['2026-01-05':TEST_START - pd.Timedelta(days=7)].idxmin()         # ช่วงต่ำระหว่างฤดู
peak2 = tot[TEST_START:].idxmax()                                          # ระลอก 2 (ใน test)
for w, dy in [(peak1, 700), (low, 900), (peak2, 700)]:
    ax.annotate(f'{tot[w]:,.0f}', (w, tot[w]), xytext=(w, tot[w] + dy), ha='center', fontsize=10,
                color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
ax.text(TEST_START + pd.Timedelta(days=2), tot.max() * 0.97, 'test', color=INK2, fontsize=10, va='top')
ax.set_title(T('ผู้ป่วยไข้หวัดใหญ่รวมทั้ง กทม. รายสัปดาห์ (2 ระลอก)', 'Weekly influenza cases, all Bangkok (two waves)'))
ax.set_ylabel(T('ผู้ป่วย (ราย/สัปดาห์)', 'cases per week'))
ax.set_ylim(0, tot.max() * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
plt.tight_layout(); plt.show()

print(f'ระลอก 1 สูงสุด: สัปดาห์ {peak1.date()} = {tot[peak1]:,.0f} ราย')
print(f'ช่วงต่ำระหว่างฤดู: สัปดาห์ {low.date()} = {tot[low]:,.0f} ราย')
print(f'ระลอก 2 สูงสุด (ใน test): สัปดาห์ {peak2.date()} = {tot[peak2]:,.0f} ราย')
print(f'ระลอก 1 สูงกว่าระลอก 2: {tot[peak1] / tot[peak2]:.2f} เท่า | ระลอก 2 สูงกว่าช่วงต่ำ {tot[peak2] / tot[low]:.1f} เท่า')
print(f'สัปดาห์สุดท้าย {tot.index[-1].date()}: {tot.iloc[-1]:,.0f} ราย (สัปดาห์ก่อน {tot.iloc[-2]:,.0f})')

# ===== จาก notebook cell 40 =====
age_tot = tab[AGE_LABELS].sum()
age_share = 100 * age_tot / tab['cases'].sum()
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.bar(['0–4', '5–14', '15–24', '25–59', '60+'], age_tot.values, color=BLUE, width=0.5)
for i in age_tot.sort_values().index[-2:]:
    k = AGE_LABELS.index(i)
    ax.text(k, age_tot[i] + age_tot.max() * 0.02, f'{age_share[i]:.0f}%', ha='center', color=INK2, fontsize=10)
ax.set_title(T('ผู้ป่วยตามกลุ่มอายุ (ปี)', 'Cases by age group (years)'))
ax.set_ylabel(T('ผู้ป่วย (ราย)', 'cases'))
ax.grid(axis='x', visible=False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
plt.tight_layout(); plt.show()
print((age_share.round(1).rename(dict(zip(AGE_LABELS, ['0-4', '5-14', '15-24', '25-59', '60+'])))).to_string())
print('ผู้ป่วยใน: %.1f%% ของผู้ป่วยทั้งหมด' % (100 * tab['inpatient_cases'].sum() / tab['cases'].sum()))
print('หมายเหตุ: ช่วงอายุกว้างไม่เท่ากัน (5, 10, 10, 35 ปี และ 60+) จึงเทียบความสูงแท่งตรงๆ เป็นอัตราป่วยไม่ได้')

# ===== จาก notebook cell 42 =====
dist_tot = tab.groupby('district')['cases'].sum().sort_values()
top10 = dist_tot.tail(10)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(top10.index, top10.values, color=BLUE, height=0.55)
ax.set_title(T('10 เขตที่มีผู้ป่วยสะสมสูงสุด (มิ.ย. 2025–ต.ค. 2026)', 'Top 10 districts by total cases (Jun 2025 - Oct 2026)'))
ax.set_xlabel(T('ผู้ป่วยสะสม (ราย)', 'total cases'))
ax.grid(axis='y', visible=False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
plt.tight_layout(); plt.show()

print('เขตสูงสุด: %s = %d ราย | เขตต่ำสุด: %s = %d ราย' % (dist_tot.index[-1], dist_tot.iloc[-1], dist_tot.index[0], dist_tot.iloc[0]))
print('10 เขตสูงสุดรวมกัน = %.1f%% ของผู้ป่วยทั้งหมด' % (100 * top10.sum() / dist_tot.sum()))
print('หมายเหตุ: นี่เป็นจำนวนผู้ป่วย ไม่ใช่อัตราต่อประชากร (ไม่มีข้อมูลประชากรรายเขต)')

# ===== จาก notebook cell 44 =====
norm = p_all.div(p_all.max(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(10, 4.3))
for dname in norm.index:
    ax.plot(norm.columns, norm.loc[dname], color=GRAY, lw=1, alpha=0.7)
ax.plot(norm.columns, tot / tot.max(), color=BLUE, lw=2.5)
ax.axvspan(TEST_START, tot.index[-1] + pd.Timedelta(days=6), color=GRID, alpha=0.5, lw=0)
ax.set_title(T('ทั้ง 50 เขตขึ้นลงตามระลอกเดียวกัน', 'All 50 districts follow the same waves'))
ax.set_ylabel(T('สัดส่วนของค่าสูงสุด', 'share of own peak'))
ax.text(tot.idxmax(), 1.03, T('รวมทั้ง กทม.', 'all Bangkok'), color=INK2, fontsize=10, ha='center')
plt.tight_layout(); plt.show()

corr = p_all.apply(lambda r: np.corrcoef(r.values, tot.values)[0, 1], axis=1)
print('สหสัมพันธ์ของแต่ละเขตกับยอดรวม กทม.: มัธยฐาน %.2f | ต่ำสุด %.2f (%s)' % (corr.median(), corr.min(), corr.idxmin()))
lag_r = np.corrcoef(p_all.shift(1, axis=1).values[:, 1:].ravel(), p_all.values[:, 1:].ravel())[0, 1]
print('สหสัมพันธ์ระหว่างผู้ป่วยสัปดาห์ก่อนกับสัปดาห์นี้ (ทุกเขต-สัปดาห์): %.2f' % lag_r)

# ===== จาก notebook cell 47 =====
tab = full.copy()    # ตารางเขต × สัปดาห์ที่สร้างจาก R2 ด้านบน

p = tab.pivot(index='district', columns='week_start', values='cases')   # แถว=เขต คอลัมน์=สัปดาห์
naive = p.shift(1, axis=1)          # ค่าที่ทำนาย = จำนวนผู้ป่วยของสัปดาห์ก่อนหน้า

weeks_all = list(p.columns)
t0 = weeks_all.index(TEST_START)                       # ตำแหน่งสัปดาห์แรกของ test
bad_idx = {weeks_all.index(w) for w in SUSPECT_WEEKS}  # สัปดาห์ที่รายงานไม่ครบ

def usable(k):
    """เป้าหมายสัปดาห์ k ใช้ได้เมื่อ k และ k-1..k-4 (ที่ baseline/feature ใช้) ไม่ใช่สัปดาห์ที่รายงานไม่ครบ"""
    return k >= 4 and not any((k - j) in bad_idx for j in range(0, 5))

train_idx = [k for k in range(4, t0) if usable(k)]
test_idx = list(range(t0, len(weeks_all)))
assert all(usable(k) for k in test_idx)
dropped = [weeks_all[k].date() for k in range(4, t0) if not usable(k)]
print('ตัดเป้าหมายออกจาก train (รายงานไม่ครบ/lag แตะสัปดาห์เหล่านั้น):', dropped[0], '→', dropped[-1], f'({len(dropped)} สัปดาห์)')
print('train:', weeks_all[train_idx[0]].date(), '→', weeks_all[train_idx[-1]].date(), '| สัปดาห์:', len(train_idx), '| แถว:', len(train_idx) * len(p))
print('test :', weeks_all[test_idx[0]].date(), '→', weeks_all[test_idx[-1]].date(), '| สัปดาห์:', len(test_idx), '| แถว:', len(test_idx) * len(p))
assert len(train_idx) * len(p) == 2350 and len(test_idx) * len(p) == 800

actual, pred = p.iloc[:, test_idx].values, naive.iloc[:, test_idx].values
BASE_MAE, BASE_UNDER = np.abs(actual - pred).mean(), 100 * (pred < actual).mean()
print('MAE = %.2f' % BASE_MAE)
print('ทำนายต่ำกว่าจริง = %.1f%%' % BASE_UNDER)

hits = []
for k in test_idx:
    top_actual = set(p.iloc[:, k].nlargest(10).index)      # 10 เขตสูงสุดจริง
    top_pred = set(naive.iloc[:, k].nlargest(10).index)    # 10 เขตที่ baseline ทำนายสูงสุด
    hits.append(len(top_actual & top_pred) / 10)
print('Top-10 hit rate = %.3f' % np.mean(hits))

# --- ทดสอบความไว: ไม่รวมสัปดาห์สุดท้ายของ test (ปลายข้อมูลอาจรายงานไม่ครบ) ---
a2, b2 = p.iloc[:, test_idx[:-1]].values, naive.iloc[:, test_idx[:-1]].values
print('\nไม่รวมสัปดาห์สุดท้ายของ test: MAE = %.2f | ต่ำกว่าจริง = %.1f%% | แถว: %d' % (np.abs(a2 - b2).mean(), 100 * (b2 < a2).mean(), a2.size))

# ===== จาก notebook cell 49 =====
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

P = p.values                       # แถว = เขต, คอลัมน์ = สัปดาห์ (73)
TOT = P.sum(axis=0)                # ยอดรวม กทม. รายสัปดาห์
rows = []
for k in range(4, P.shape[1]):     # k = สัปดาห์เป้าหมาย
    if not usable(k):
        continue                   # ข้ามสัปดาห์ที่ lag แตะข้อมูลที่รายงานไม่ครบ
    for i, dname in enumerate(p.index):
        lags = [P[i, k - j] for j in range(1, 5)]        # สัปดาห์ k-1 ... k-4 (รู้แล้ว ณ เช้าวันจันทร์)
        rows.append({'district': dname, 'week_idx': k, 'week': weeks_all[k], 'y': P[i, k],
                     'lag1': lags[0], 'lag2': lags[1], 'lag3': lags[2], 'lag4': lags[3],
                     'ma3': np.mean(lags[:3]), 'diff1': lags[0] - lags[1],
                     'bkk_lag1': TOT[k - 1], 'bkk_diff1': TOT[k - 1] - TOT[k - 2]})
data = pd.DataFrame(rows)
FEATURES = ['lag1', 'lag2', 'lag3', 'lag4', 'ma3', 'diff1', 'bkk_lag1', 'bkk_diff1']

# ตรวจ data leakage: feature ต้องมาจากสัปดาห์ก่อนหน้าเท่านั้น
district_pos = {d_: j for j, d_ in enumerate(p.index)}
for _, r in data.sample(300, random_state=0).iterrows():
    j, k = district_pos[r['district']], int(r['week_idx'])
    assert r['lag1'] == P[j, k - 1] and r['lag4'] == P[j, k - 4] and r['bkk_lag1'] == TOT[k - 1]
    assert r['y'] == P[j, k]

train = data[data['week_idx'] < t0].copy()
test = data[data['week_idx'] >= t0].copy()
print('train:', len(train), 'แถว | test:', len(test), 'แถว | features:', len(FEATURES))
assert len(train) == 2350 and len(test) == 800

# ===== จาก notebook cell 51 =====
def make_ridge(a):
    return lambda: make_pipeline(StandardScaler(), Ridge(alpha=a))
def make_rf(depth, leaf):
    return lambda: RandomForestRegressor(n_estimators=100, max_depth=depth, min_samples_leaf=leaf,
                                         random_state=0, n_jobs=-1)

CANDIDATES = {
    'Ridge': {f'alpha={a}': make_ridge(a) for a in [1, 10, 100, 1000]},
    'RandomForest': {f'depth={dp}, leaf={lf}': make_rf(dp, lf) for dp, lf in [(3, 20), (5, 10)]},
}
# ตรวจบนช่วงถัดไปภายใน train 3 ช่วง (ขยายหน้าต่างฝึกทีละช่วง ไม่ย้อนอนาคต)
FOLD_VAL = [range(35, 43), range(43, 50), range(50, 57)]   # 2 ก.พ.–23 มี.ค. | 30 มี.ค.–11 พ.ค. | 18 พ.ค.–29 มิ.ย. 2026
assert all(k in train_idx for va in FOLD_VAL for k in va)
for va in FOLD_VAL:
    print('ตรวจ:', weeks_all[va[0]].date(), '→', weeks_all[va[-1]].date(),
          '| ฝึกด้วยสัปดาห์เป้าหมายก่อนหน้า', sum(k < va[0] for k in train_idx), 'สัปดาห์')

def cv_eval(factory=None):
    maes, unders = [], []
    for va_w in FOLD_VAL:
        tr_w = [k for k in train_idx if k < va_w[0]]
        a = train[train['week_idx'].isin(tr_w)]
        b = train[train['week_idx'].isin(va_w)]
        pr = b['lag1'].values.astype(float) if factory is None else factory().fit(a[FEATURES], a['y']).predict(b[FEATURES])
        maes.append(np.abs(pr - b['y'].values).mean()); unders.append((pr < b['y'].values).mean())
    return np.mean(maes), 100 * np.mean(unders)

cv_rows = [('baseline (ค่าสัปดาห์ก่อน)', '-') + cv_eval(None)]
best = {}
for fam, cands in CANDIDATES.items():
    scores = {name: cv_eval(f) for name, f in cands.items()}
    for name, (m_, u_) in scores.items():
        cv_rows.append((fam, name, m_, u_))
    best[fam] = min(scores, key=lambda n: scores[n][0])
cv_table = pd.DataFrame(cv_rows, columns=['โมเดล', 'พารามิเตอร์', 'CV MAE', 'ต่ำกว่าจริง %'])
print(cv_table.round(2).to_string(index=False))
print('\nพารามิเตอร์ที่เลือก (MAE ต่ำสุดใน CV):', best)

# ===== จาก notebook cell 53 =====
final = {fam: CANDIDATES[fam][best[fam]]().fit(train[FEATURES], train['y']) for fam in CANDIDATES}
preds = {'baseline': test['lag1'].values.astype(float)}
for fam, m_ in final.items():
    preds[fam] = m_.predict(test[FEATURES])

y_te = test['y'].values
last_w = test['week_idx'].max()
mask_ex = (test['week_idx'] < last_w).values          # ไม่รวมสัปดาห์สุดท้ายของ test (19 ต.ค.)

def top10_hit(pred):
    hits = []
    for w, g in test.assign(pred=pred).groupby('week_idx'):
        hits.append(len(set(g.nlargest(10, 'y')['district']) & set(g.nlargest(10, 'pred')['district'])) / 10)
    return np.mean(hits)

res_rows = []
for name, pr in preds.items():
    res_rows.append({'โมเดล': name,
                     'MAE': np.abs(pr - y_te).mean(),
                     'ต่ำกว่าจริง %': 100 * (pr < y_te).mean(),
                     'Top-10 hit': top10_hit(pr),
                     'MAE (ไม่รวมสัปดาห์สุดท้าย)': np.abs(pr - y_te)[mask_ex].mean(),
                     'ต่ำกว่าจริง % (ไม่รวมสัปดาห์สุดท้าย)': 100 * (pr < y_te)[mask_ex].mean()})
res_table = pd.DataFrame(res_rows).set_index('โมเดล')
print(res_table.round(2).to_string())
assert abs(res_table.loc['baseline', 'MAE'] - BASE_MAE) < 1e-9, 'baseline ไม่ตรงกับส่วนที่ 1'

# ===== จาก notebook cell 55 =====
rng = np.random.default_rng(42)
wk_list = sorted(test['week_idx'].unique())
wk_mae = {n: np.array([np.abs(pr - y_te)[(test['week_idx'] == w).values].mean() for w in wk_list])
          for n, pr in preds.items()}
boot = rng.integers(0, len(wk_list), size=(5000, len(wk_list)))      # สุ่มสัปดาห์ 8 ครั้ง คืนที่ 5,000 รอบ

print('MAE เฉลี่ยรายสัปดาห์ และช่วงความเชื่อมั่น 95% (bootstrap สัปดาห์):')
ci = {}
for n in preds:
    b_ = wk_mae[n][boot].mean(axis=1)
    ci[n] = np.percentile(b_, [2.5, 97.5])
    print(f'  {n:13s} {wk_mae[n].mean():6.2f}  [{ci[n][0]:.2f}, {ci[n][1]:.2f}]')
print('\nผลต่างเทียบ baseline (ลบ = โมเดลดีกว่า):')
for n in preds:
    if n == 'baseline':
        continue
    dlt = (wk_mae[n] - wk_mae['baseline'])[boot].mean(axis=1)
    lo, hi = np.percentile(dlt, [2.5, 97.5])
    print(f'  {n:13s} {np.mean(wk_mae[n] - wk_mae["baseline"]):+6.2f}  [{lo:+.2f}, {hi:+.2f}]  | โมเดลดีกว่าใน {100 * (dlt < 0).mean():.0f}% ของรอบ bootstrap')
print('\nจำนวนสัปดาห์ test ที่โมเดลมี MAE ต่ำกว่า baseline:')
for n in preds:
    if n != 'baseline':
        print(f'  {n:13s} {int((wk_mae[n] < wk_mae["baseline"]).sum())} จาก {len(wk_list)} สัปดาห์')

# ===== จาก notebook cell 56 =====
fig, ax = plt.subplots(figsize=(8, 3.4))
names = list(preds)
for y_, n in enumerate(reversed(names)):
    m_ = wk_mae[n].mean(); lo, hi = ci[n]
    ax.errorbar(m_, y_, xerr=[[m_ - lo], [hi - m_]], fmt='o', ms=8, lw=2, capsize=3,
                color=GRAY if n == 'baseline' else BLUE)
    ax.text(hi + 0.3, y_, f"{m_:.1f}  (ต่ำกว่าจริง {res_table.loc[n, 'ต่ำกว่าจริง %']:.0f}%)" if TH
            else f"{m_:.1f}  (under {res_table.loc[n, 'ต่ำกว่าจริง %']:.0f}%)",
            va='center', color=INK2, fontsize=10)
ax.set_yticks(range(len(names))); ax.set_yticklabels(list(reversed(names)))
ax.set_ylim(-0.6, len(names) - 0.4)
ax.set_xlim(min(c[0] for c in ci.values()) - 2, max(c[1] for c in ci.values()) + 11)
ax.set_xlabel(T('MAE บน test (ราย/เขต/สัปดาห์) พร้อมช่วงความเชื่อมั่น 95%', 'test MAE (cases/district/week), 95% CI'))
ax.set_title(T(f'โมเดลเทียบ baseline บน test {len(wk_list)} สัปดาห์', f'Models vs baseline on the {len(wk_list)}-week test set'))
ax.grid(axis='y', visible=False)
plt.tight_layout(); plt.show()

# เก็บผลทำนายไว้ใช้ใน R5 (วิเคราะห์จุดที่พลาด)
results = test[['district', 'week', 'week_idx', 'y']].copy()
for n, pr in preds.items():
    results[n] = pr
results.head()

# ===== จาก notebook cell 60 =====

MODELS = [c for c in results.columns if c not in ('district', 'week', 'week_idx', 'y')]   # baseline, Ridge, RandomForest
wk = results.groupby('week')[['y'] + MODELS].sum()
wk.index = pd.to_datetime(wk.index)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(wk.index, wk['y'], color=INK, lw=2.5, marker='o', label=T('ค่าจริง', 'actual'))
ax.plot(wk.index, wk['baseline'], color=GRAY, lw=2, marker='o', label=T('baseline (ค่าสัปดาห์ก่อน)', 'baseline (last week)'))
ax.plot(wk.index, wk['Ridge'], color=BLUE, lw=2, marker='o', label='Ridge')
ax.set_ylim(bottom=0)
ax.set_ylabel(T('ผู้ป่วยรวม กทม. (ราย/สัปดาห์)', 'BKK total cases / week'))
ax.set_title(T('ค่าทำนายตามหลังค่าจริงอยู่ 1 สัปดาห์', 'Predictions lag the actual by one week'))
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
print((wk.round(0)).astype(int).to_string())

# ===== จาก notebook cell 62 =====

tot_actual = pd.Series(TOT, index=pd.to_datetime(weeks_all))
chg = tot_actual.diff()
phase = chg.map(lambda v: 'ขาขึ้น' if v > 0 else 'ขาลง')
results['phase'] = pd.to_datetime(results['week']).map(phase)

rows = []
for ph, g in results.groupby('phase'):
    for m_ in MODELS:
        e = g[m_] - g['y']
        rows.append({'ช่วง': ph, 'โมเดล': m_, 'สัปดาห์': g['week'].nunique(),
                     'MAE': e.abs().mean(), 'ต่ำกว่าจริง %': 100 * (e < 0).mean(),
                     'พลาดต่ำเฉลี่ย (ราย/เขต)': e[e < 0].mean() if (e < 0).any() else 0.0})
ph_tab = pd.DataFrame(rows)
print(ph_tab.round(1).to_string(index=False))

print('\nรายสัปดาห์ (baseline): ผลต่างยอดรวม กทม. สัปดาห์นั้นเทียบสัปดาห์ก่อน, ทำนายต่ำ %, ผู้ป่วยที่ "ขาด" รวม กทม.')
b = results.copy(); b['err'] = b['baseline'] - b['y']
w_ = b.groupby('week').agg(ต่ำกว่าจริง_pct=('err', lambda s: 100 * (s < 0).mean()),
                           ขาดรวม=('err', lambda s: -s[s < 0].sum()), เกินรวม=('err', lambda s: s[s > 0].sum()))
w_['เปลี่ยนจากสัปดาห์ก่อน'] = chg.reindex(pd.to_datetime(w_.index)).values
print(w_.round(0).astype(int).to_string())

# ===== จาก notebook cell 63 =====

fig, ax = plt.subplots(figsize=(8, 3.6))
ph_b = ph_tab[ph_tab['โมเดล'] == 'baseline'].set_index('ช่วง')
xs = np.arange(len(ph_b))
ax.bar(xs, ph_b['ต่ำกว่าจริง %'], color=[BLUE if i == 'ขาขึ้น' else GRAY for i in ph_b.index], width=0.55)
for x_, v_ in zip(xs, ph_b['ต่ำกว่าจริง %']):
    ax.text(x_, v_ + 2, f'{v_:.0f}%', ha='center', color=INK, fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels([f"{i}\n({int(ph_b.loc[i, 'สัปดาห์'])} สัปดาห์)" for i in ph_b.index])
ax.set_ylim(0, 110); ax.set_ylabel(T('% ของเขต-สัปดาห์ที่ทำนายต่ำกว่าจริง', '% of district-weeks under-predicted'))
ax.set_title(T('baseline ทำนายต่ำเกือบทุกเขตตอนระลอกกำลังขึ้น', 'Baseline under-predicts almost everywhere when the wave is rising'))
ax.grid(axis='x', visible=False)
plt.tight_layout(); plt.show()

# ===== จาก notebook cell 65 =====

b['abs_err'] = b['err'].abs()
print('10 เขต-สัปดาห์ที่ baseline พลาดมากที่สุด:')
print(b.nlargest(10, 'abs_err')[['district', 'week', 'y', 'baseline', 'err']]
      .rename(columns={'y': 'จริง', 'baseline': 'ทำนาย', 'err': 'ทำนาย-จริง'}).to_string(index=False))

d = b.groupby('district').agg(MAE=('abs_err', 'mean'), ค่าจริงเฉลี่ย=('y', 'mean'))
d['MAE เทียบค่าจริง %'] = 100 * d['MAE'] / d['ค่าจริงเฉลี่ย']
print('\n5 เขตที่ MAE สูงสุด:'); print(d.nlargest(5, 'MAE').round(1).to_string())
print('\nสหสัมพันธ์ ขนาดเขต (ค่าจริงเฉลี่ย) กับ MAE:', round(d['ค่าจริงเฉลี่ย'].corr(d['MAE']), 2))
print('MAE เทียบค่าจริง %: มัธยฐาน', round(d['MAE เทียบค่าจริง %'].median(), 1), '| ต่ำสุด', round(d['MAE เทียบค่าจริง %'].min(), 1), '| สูงสุด', round(d['MAE เทียบค่าจริง %'].max(), 1))
share = b.groupby('district')['abs_err'].sum().sort_values(ascending=False)
print('10 เขตแรก คิดเป็น', round(100 * share.head(10).sum() / share.sum(), 1), '% ของ error รวม (ถ้ากระจายเท่ากัน = 20%)')

# ===== จาก notebook cell 67 =====

hit_rows = []
for w, g in results.groupby('week'):
    true10 = set(g.nlargest(10, 'y')['district'])
    hit_rows.append({'week': w, **{m_: len(true10 & set(g.nlargest(10, m_)['district'])) for m_ in MODELS}})
hit = pd.DataFrame(hit_rows).set_index('week')
print('จำนวนเขตที่ถูก จากทั้งหมด 10 ในแต่ละสัปดาห์:'); print(hit.to_string())
print('\nเฉลี่ย:', (hit.mean() / 10).round(3).to_dict())

# ===== บันทึกตารางผลลัพธ์ลง results/output/ =====
cv_table.round(3).to_csv(OUT_DIR / 'r4_cv_table.csv', index=False, encoding='utf-8-sig')
res_table.round(3).to_csv(OUT_DIR / 'r4_test_results.csv', encoding='utf-8-sig')
ph_tab.round(2).to_csv(OUT_DIR / 'r5_phase_table.csv', index=False, encoding='utf-8-sig')
w_.to_csv(OUT_DIR / 'r5_weekly_baseline.csv', encoding='utf-8-sig')
hit.to_csv(OUT_DIR / 'r5_top10_hits.csv', encoding='utf-8-sig')
wk.round(0).astype(int).to_csv(OUT_DIR / 'r5_weekly_totals.csv', encoding='utf-8-sig')
print('บันทึกตารางผลลัพธ์ใน results/output/ แล้ว')
