"""ขั้นที่ 1: ทำความสะอาดข้อมูล (R2)
อ่านไฟล์ดิบ 3 ไฟล์ใน data/raw/ -> กรองไข้หวัดใหญ่ -> ตรวจ/แก้ปัญหา -> ตารางเขต x สัปดาห์
เขียน data/processed/flu_district_week_2025_2026.csv และ results/output/r2_problem_summary.csv
(โค้ดตัดมาจาก notebook ส่วน R2 โดยไม่เปลี่ยนตรรกะ)
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from utils import *
install_show('preprocess')

import os
import numpy as np
import pandas as pd

# ข้อมูล 3 ไฟล์ ต่อกันเป็นช่วงเวลาเดียวต่อเนื่อง (วางทุกไฟล์ไว้ในโฟลเดอร์เดียวกัน)
FILES = [('2025 มิ.ย.–ธ.ค.', '5_2025_sent_2025-12-316-12.csv'),
         ('2026 ม.ค.–มิ.ย.', '5_2026_1-6.csv'),
         ('2026 ก.ค.–ต.ค.', '5_2026_7-10.csv')]

# โฟลเดอร์ไฟล์ดิบ: ตั้งด้วยตัวแปรแวดล้อม FLU_DATA_DIR หรือ main.py --data-dir (ค่าเริ่มต้น data/raw/)
DATA_DIR = os.environ.get('FLU_DATA_DIR', str(RAW_DIR)).rstrip('/\\') + os.sep

START, END = pd.Timestamp('2025-06-02'), pd.Timestamp('2026-10-25')   # 73 สัปดาห์เต็ม จันทร์–อาทิตย์
EXPECT_DISTRICTS, EXPECT_WEEKS = 50, 73                               # กทม. มี 50 เขต
AGE_MAX = 110
DUP_POLICY = 'keep'  # 'keep' = เก็บแถวซ้ำ | 'drop' = ลบแถวซ้ำ (เหตุผลดูหัวข้อ 4)
problems = []   # (ปัญหา, จำนวนที่พบ, วิธีจัดการ) -> ตารางโปสเตอร์

# ---- การออกแบบ train/test: ล็อกไว้ก่อนรันโมเดล (ดูหัวข้อ "วิธีแบ่ง train/test") ----
TEST_START = pd.Timestamp('2026-07-06')                               # test = ก.ค.–ต.ค. 2026 (ฤดูที่ 2)
SUSPECT_WEEKS = [pd.Timestamp('2025-12-22'), pd.Timestamp('2025-12-29')]   # รายงานไม่ครบปลายไฟล์ 2025 (ดูหัวข้อ 3.1)


# ===== จาก notebook cell 9 =====
frames, file_rows = [], []
for tag, fname in FILES:
    x = pd.read_csv(DATA_DIR + fname, dtype=str)
    n_all = len(x)
    n_blank = int(x.isna().all(axis=1).sum())          # แถวที่ว่างทุกคอลัมน์
    x = x.dropna(how='all').copy()
    x['ชื่อกลุ่มโรค'] = x['ชื่อกลุ่มโรค'].str.strip()
    x['src'] = tag
    frames.append(x)
    file_rows.append((tag, n_all, n_blank, len(x), int((x['ชื่อกลุ่มโรค'] == 'ไข้หวัดใหญ่').sum())))
print(pd.DataFrame(file_rows, columns=['ไฟล์', 'แถวในไฟล์', 'แถวว่างทั้งแถว', 'แถวข้อมูลจริง', 'ไข้หวัดใหญ่']).to_string(index=False))
assert all(list(f.columns) == list(frames[0].columns) for f in frames), 'คอลัมน์ของ 3 ไฟล์ไม่ตรงกัน'

raw = pd.concat(frames, ignore_index=True)
RAW_COLS = [c for c in raw.columns if c != 'src']
N_BLANK = sum(r[2] for r in file_rows)
problems.append(('แถวว่างทั้งแถว (ไฟล์ ม.ค.–มิ.ย. 2026)', N_BLANK, 'ลบออกก่อนใช้งาน เพราะไม่มีข้อมูลใดๆ'))

flu = raw[raw['ชื่อกลุ่มโรค'] == 'ไข้หวัดใหญ่'].copy()
N_FLU = len(flu)
print('\nไข้หวัดใหญ่รวม 3 ไฟล์:', N_FLU, '(คาด 209,402)', '✓' if N_FLU == 209402 else '✗ ตรวจสอบ')

# ===== จาก notebook cell 11 =====
flu['date'] = pd.to_datetime(flu['วันที่เริ่มป่วย'].str.strip(), format='%d/%m/%Y', errors='coerce')
n_bad_date = int(flu['date'].isna().sum())
print('แปลงวันที่ไม่ได้:', n_bad_date)
print(flu.groupby('src')['date'].agg(['min', 'max']).to_string())      # ต้องต่อกันเป็นช่วงเดียว ปี ค.ศ. ทุกไฟล์

# week_start = วันจันทร์ของสัปดาห์นั้น (dayofweek: จันทร์=0 ... อาทิตย์=6)
flu['week_start'] = flu['date'] - pd.to_timedelta(flu['date'].dt.dayofweek, unit='D')

# ===== จาก notebook cell 13 =====
n_before = int((flu['date'] < START).sum())
n_after  = int((flu['date'] > END).sum())
in_range = flu['date'].between(START, END)
d = flu[in_range].copy()

print(f'ก่อน {START.date()} (สัปดาห์แรกมีแค่ 1 มิ.ย. 2025): {n_before} แถว')
print(f'หลัง {END.date()} (26–31 ต.ค. 2026 สัปดาห์ไม่เต็ม และรายงานยังไม่ครบ): {n_after} แถว')
print(f'วันที่แปลงไม่ได้: {n_bad_date} แถว')
print('คงเหลือในช่วง:', len(d), '| ตรวจ:', N_FLU, '=', len(d), '+', n_before + n_after + n_bad_date)
assert len(d) + n_before + n_after + n_bad_date == N_FLU

problems.append(('วันที่อยู่นอกช่วง 2 มิ.ย. 2025–25 ต.ค. 2026 (1 มิ.ย. 2025 และ 26–31 ต.ค. 2026)', n_before + n_after,
                 'ตัดออก เพื่อให้ทุกสัปดาห์ครบ 7 วัน (จันทร์–อาทิตย์)'))
problems.append(('วันที่เริ่มป่วยแปลงไม่ได้', n_bad_date, 'ตัดออก (ไม่มีสัปดาห์ให้จัดกลุ่ม)'))

# ===== จาก notebook cell 15 =====
daily = flu['date'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (lo, hi, ttl) in zip(axes, [('2025-12-08', '2026-01-18', T('รอยต่อไฟล์ 2025–2026', 'File boundary 2025 to 2026')),
                                    ('2026-10-05', '2026-10-31', T('ปลายข้อมูล ต.ค. 2026', 'End of data, Oct 2026'))]):
    s = daily[lo:hi]
    ax.plot(s.index, s.values, color=BLUE, lw=1.8, marker='o', ms=3.5)
    ax.set_title(ttl); ax.set_ylabel(T('ผู้ป่วยรายวัน', 'cases per day'))
    ax.set_ylim(0)
    ax.tick_params(axis='x', labelrotation=30)
axes[0].axvline(pd.Timestamp('2025-12-31'), color=MUTED, lw=1, ls='--')
axes[0].text(pd.Timestamp('2025-12-31'), daily['2025-12-08':'2026-01-18'].max() * 0.97, T(' วันสุดท้ายของไฟล์ 2025', ' last day of 2025 file'), color=INK2, fontsize=9, va='top')
for w in SUSPECT_WEEKS:
    axes[0].axvspan(w, w + pd.Timedelta(days=6), color='#d9622b', alpha=0.12, lw=0)
plt.tight_layout(); plt.show()

print('รายวัน 22 ธ.ค. 2025 – 4 ม.ค. 2026 (31 ธ.ค. ไม่มีรายงานเลย):')
print(daily['2025-12-22':'2026-01-04'].reindex(pd.date_range('2025-12-22', '2026-01-04'), fill_value=0).astype(int).to_string())
print('\nรายวัน 24–31 ต.ค. 2026:'); print(daily['2026-10-24':'2026-10-31'].astype(int).to_string())

SUSPECT_N = int(d['week_start'].isin(SUSPECT_WEEKS).sum())
print(f'\nผู้ป่วยในสัปดาห์ที่ไม่น่าเชื่อถือ {[w.date().isoformat() for w in SUSPECT_WEEKS]}: {SUSPECT_N} ราย')
problems.append(('รายงานล่าช้าปลายไฟล์ 2025 (สัปดาห์ 22 และ 29 ธ.ค. 2025): 22–28 ธ.ค. ลดจาก 382 เหลือ 154 ราย/วัน, 30 ธ.ค. เหลือ 23, 31 ธ.ค. ไม่มี แต่ 1 ม.ค. กลับมา 274',
                 SUSPECT_N, 'เก็บไว้ในตาราง (ผลรวมยังตรวจได้) แต่ไม่ใช้เป็นเป้าหมายหรือ lag ของโมเดล (ตัดเป้าหมาย 22 ธ.ค.–26 ม.ค. รวม 6 สัปดาห์)'))

# ===== จาก notebook cell 17 =====
dup_extra  = int(d.duplicated(subset=RAW_COLS, keep='first').sum())   # แถวที่ "เกิน" ซ้ำ
dup_in_grp = int(d.duplicated(subset=RAW_COLS, keep=False).sum())     # แถวทั้งหมดที่อยู่ในกลุ่มซ้ำ
print(f'แถวซ้ำเต็มแถว: เกิน {dup_extra} แถว (รวมทุกแถวที่อยู่ในกลุ่มซ้ำ {dup_in_grp}) = {dup_extra/len(d):.2%}')

if DUP_POLICY == 'drop':
    d = d.drop_duplicates(subset=RAW_COLS, keep='first').copy()
    handling = 'ลบแถวซ้ำ เก็บแถวแรกไว้'
else:
    handling = ('เก็บไว้ทั้งหมด: ไฟล์ไม่มีรหัสผู้ป่วย แยกรายงานซ้ำออกจากผู้ป่วยคนละคนที่ข้อมูลตรงกันไม่ได้ '
                'และการลบอาจทำให้จำนวนผู้ป่วยต่ำกว่าจริง')
problems.append(('แถวซ้ำเต็มแถว (ทุกคอลัมน์เหมือนกัน)', dup_extra, handling))
print('นโยบาย:', DUP_POLICY, '| แถวที่ใช้ต่อ:', len(d))

# ===== จาก notebook cell 19 =====
# --- อายุ (ใช้คอลัมน์ 'อายุ (เต็ม) ปี') ---
age_raw = d['อายุ (เต็ม) ปี'].str.strip()
age = pd.to_numeric(age_raw, errors='coerce')
n_age_blank  = int(age_raw.isna().sum())
n_age_nonnum = int((age.isna() & age_raw.notna()).sum())
n_age_over   = int((age > AGE_MAX).sum())
n_age_neg    = int((age < 0).sum())
d['age'] = age.mask((age > AGE_MAX) | (age < 0))
print(f'อายุว่าง {n_age_blank} | แปลงไม่ได้ {n_age_nonnum} | เกิน {AGE_MAX} {n_age_over} | ติดลบ {n_age_neg} | NaN รวมหลังทำความสะอาด {int(d["age"].isna().sum())}')
if n_age_blank:  problems.append(('อายุว่าง', n_age_blank, 'เป็น NaN (นับใน cases แต่ไม่อยู่ในกลุ่มอายุใด)'))
problems.append((f'อายุแปลงเป็นตัวเลขไม่ได้', n_age_nonnum, 'เป็น NaN'))
problems.append((f'อายุผิดปกติ (>{AGE_MAX} หรือ <0)', n_age_over + n_age_neg, 'เป็น NaN'))

# --- เพศ / สภาพผู้ป่วย / ประเภทผู้ป่วย: "0" -> NaN ---
for col in ['เพศ', 'สภาพผู้ป่วย', 'ประเภทผู้ป่วย']:
    d[col] = d[col].str.strip()
    n0 = int((d[col] == '0').sum())
    d[col] = d[col].replace('0', np.nan)
    print(f'{col}: "0" = {n0}')
    problems.append((f'{col} เป็น "0" (รหัสไม่มีข้อมูล)', n0, 'เปลี่ยนเป็น NaN'))

# --- ชื่อเขต ---
d['district'] = d['อำเภอ/เขต'].str.strip()
n_space = int((d['อำเภอ/เขต'] != d['district']).sum())
n_dist_na = int(d['district'].isna().sum())
print(f'ชื่อเขตมีช่องว่างหน้าหลัง: {n_space} | เขตว่าง: {n_dist_na}')
problems.append(('ชื่อเขตมีช่องว่างหน้า/หลัง', n_space, 'ตัดช่องว่าง (strip)'))
# --- ชื่อเขตสะกดต่างกัน: บางแถวมีคำนำหน้า "เขต" (เช่น "เขตบางพลัด") ซึ่งคือ "บางพลัด" เดิม ---
d['district_norm'] = d['district'].str.replace(r'^เขต', '', regex=True).str.strip()
n_prefix = int((d['district'] != d['district_norm']).sum())
print(f'ชื่อเขตมีคำนำหน้า "เขต": {n_prefix} แถว', d.loc[d['district'] != d['district_norm'], 'district'].unique())
d['district'] = d['district_norm']
problems.append(('ชื่อเขตสะกดต่างกัน (มีคำนำหน้า "เขต" เช่น เขตบางพลัด)', n_prefix, 'ตัดคำนำหน้า "เขต" ให้ตรงกับชื่อเขตมาตรฐาน (รวมเป็น 50 เขต)'))
assert n_dist_na == 0, 'มีแถวที่ไม่มีชื่อเขต: ต้องตัดสินใจก่อน (จะหายไปตอน groupby)'

print('\nจังหวัด:'); print(d['จังหวัด'].value_counts(dropna=False))
print('\nประเภทผู้ป่วย:'); print(d['ประเภทผู้ป่วย'].value_counts(dropna=False))

# ===== จาก notebook cell 21 =====
districts = sorted(d['district'].unique())
print(len(districts), 'เขต')
print(districts)
if len(districts) != EXPECT_DISTRICTS:
    print(f'⚠ ได้ {len(districts)} ไม่ใช่ {EXPECT_DISTRICTS}: ดูรายชื่อด้านบน (สะกดต่างกัน? มีค่าแปลก เช่น "ไม่ทราบ"/เขตนอก กทม.?) แล้วแจ้งในกลุ่ม')

# ===== จาก notebook cell 23 =====
AGE_LABELS = ['age_0_4', 'age_5_14', 'age_15_24', 'age_25_59', 'age_60plus']
d['age_group'] = pd.cut(d['age'], bins=[0, 5, 15, 25, 60, np.inf], right=False, labels=AGE_LABELS)  # [0,5) [5,15) ...

d['inpatient_cases'] = (d['ประเภทผู้ป่วย'] == 'ผู้ป่วยใน').astype(int)
for g in AGE_LABELS:
    d[g] = (d['age_group'] == g).astype(int)
d['age_nan'] = d['age'].isna().astype(int)
d['cases'] = 1

agg = d.groupby(['district', 'week_start'])[['cases', 'inpatient_cases'] + AGE_LABELS + ['age_nan']].sum()

weeks = pd.date_range(START, periods=EXPECT_WEEKS, freq='7D')
assert weeks[-1] + pd.Timedelta(days=6) == END, 'สัปดาห์สุดท้ายต้องจบที่ 25 ต.ค. 2026'
grid = pd.MultiIndex.from_product([districts, weeks], names=['district', 'week_start'])
full = agg.reindex(grid, fill_value=0).reset_index()
print(full.shape)
full.head()

# ===== จาก notebook cell 25 =====
n_rows_expected = len(districts) * EXPECT_WEEKS
print('1) จำนวนแถว', len(full), '(คาด', n_rows_expected, ')')
assert len(full) == n_rows_expected
if len(districts) == EXPECT_DISTRICTS: assert len(full) == 3650
assert not full.duplicated(['district', 'week_start']).any(), 'มีคู่เขต-สัปดาห์ซ้ำ'

print('2) sum(cases) =', full['cases'].sum(), '| แถวที่ใช้ในช่วงวันที่ =', len(d))
assert full['cases'].sum() == len(d)
expl = N_FLU - full['cases'].sum()
print(f'   ส่วนต่างจากไข้หวัดใหญ่ทั้งไฟล์ {expl} = นอกช่วง {n_before + n_after} + วันที่เสีย {n_bad_date}' +
      (f' + ลบซ้ำ {dup_extra}' if DUP_POLICY == 'drop' else ''))

print('3) ติดลบ/NaN:', int((full['cases'] < 0).sum()), '/', int(full['cases'].isna().sum()))
assert (full['cases'] >= 0).all() and full['cases'].notna().all()

print('4) กลุ่มอายุ + age NaN = cases')
assert (full[AGE_LABELS].sum(axis=1) + full['age_nan'] == full['cases']).all()
assert (full['inpatient_cases'] <= full['cases']).all()
print('ผ่านครบ ✓')

print(f'\nสัดส่วนเขต-สัปดาห์ที่ cases=0: {(full.cases==0).mean():.2%} | cases เฉลี่ย {full.cases.mean():.1f} ราย/เขต/สัปดาห์')

# ===== บันทึกผล (cell 27 ปรับให้เขียนลง repo) =====
out = full[['district', 'week_start', 'cases', 'inpatient_cases'] + AGE_LABELS].copy()
out['week_start'] = out['week_start'].dt.strftime('%Y-%m-%d')
out.to_csv(PROC_DIR / 'flu_district_week_2025_2026.csv', index=False, encoding='utf-8-sig')

summary = pd.DataFrame(problems, columns=['ปัญหา', 'จำนวนที่พบ', 'วิธีจัดการ'])
summary.to_csv(OUT_DIR / 'r2_problem_summary.csv', index=False, encoding='utf-8-sig')
print(summary.to_string())
print('บันทึกแล้ว:', PROC_DIR / 'flu_district_week_2025_2026.csv')

# ===== จาก notebook cell 29 =====
hosp = d.groupby('สถานที่รักษา')['district'].agg(
    n_cases='size', n_districts='nunique',
    top_share=lambda s: s.value_counts(normalize=True).iloc[0])
big = hosp[hosp['n_cases'] >= 200]
print('สถานที่รักษาที่มีผู้ป่วย >= 200 ราย:', len(big), 'แห่ง')
print('จำนวนเขตต่อสถานที่รักษา (มัธยฐาน):', big['n_districts'].median())
print('สัดส่วนผู้ป่วยจากเขตที่พบมากที่สุด ต่อสถานที่รักษา (มัธยฐาน): %.2f' % big['top_share'].median())

# ===== จาก notebook cell 32 =====
def weekly_table(rows):
    t = rows.groupby(['district', 'week_start']).size().unstack(fill_value=0)
    return t.reindex(index=districts, columns=weeks, fill_value=0)

def naive_scores(p_):
    nv = p_.shift(1, axis=1)
    tw = [w for w in p_.columns if w >= TEST_START]      # สัปดาห์ test
    a, b = p_[tw].values, nv[tw].values
    return np.abs(a - b).mean(), 100 * (b < a).mean()

dup_rows = int(d.duplicated(subset=RAW_COLS, keep='first').sum())
for name, rows in [('เก็บแถวซ้ำ (ที่ใช้)', d), ('ลบแถวซ้ำ', d.drop_duplicates(subset=RAW_COLS, keep='first'))]:
    mae_, under_ = naive_scores(weekly_table(rows))
    print(f'{name:22s} แถว = {len(rows):,} | baseline MAE = {mae_:.2f} | ต่ำกว่าจริง = {under_:.1f}%')
print('แถวซ้ำส่วนเกินที่พบ:', dup_rows)
