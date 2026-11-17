"""ตัวช่วยร่วมของ src/: ฟอนต์ไทย + สไตล์กราฟ + ที่เก็บกราฟอัตโนมัติ (ตัดมาจาก notebook ส่วน "ตั้งค่ากราฟ")"""
import os, sys, shutil, subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')                       # ไม่เปิดหน้าต่างกราฟ บันทึกเป็นไฟล์แทน
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ให้พิมพ์ภาษาไทยได้บน Windows (คอนโซลเดิมอาจใช้ encoding อื่น)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]  # โฟลเดอร์หลักของ repo
FIG_DIR = ROOT / 'results' / 'figures'
OUT_DIR = ROOT / 'results' / 'output'
RAW_DIR = ROOT / 'data' / 'raw'
PROC_DIR = ROOT / 'data' / 'processed'
for _d in (FIG_DIR, OUT_DIR, PROC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- ตั้งค่ากราฟ: ฟอนต์ไทย + สไตล์ ----------

def find_thai_font():
    keys = ['loma', 'garuda', 'norasi', 'kinnari', 'waree', 'laksaman', 'sarabun', 'notosansthai', 'tahoma', 'leelawui', 'thsarabun', 'angsana', 'cordia', 'browallia']
    for f in fm.findSystemFonts():
        if any(k in os.path.basename(f).lower() for k in keys):
            return f
    return None

font_path = find_thai_font()
if font_path is None and shutil.which('apt-get'):      # Colab: ติดตั้งฟอนต์ไทย (ประมาณครึ่งนาที)
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'fonts-thai-tlwg'], check=False, capture_output=True)
    font_path = find_thai_font()

TH = font_path is not None
if TH:
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    print('ใช้ฟอนต์ไทย:', plt.rcParams['font.family'])
else:
    print('ไม่พบฟอนต์ไทย: ใช้ข้อความภาษาอังกฤษบนกราฟ (ชื่อเขตอาจเป็นสี่เหลี่ยม)')

def T(th, en):
    return th if TH else en

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, GRAY = '#2a78d6', '#c3c2b7'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': AXIS, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1, 'axes.axisbelow': True,
    'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 13,
    'axes.titleweight': 'normal', 'axes.titlelocation': 'left'})


_counter = {}
def install_show(prefix):
    """แทน plt.show() ด้วยการบันทึกกราฟเป็น results/figures/<prefix>_NN.png"""
    def _save(*a, **k):
        _counter[prefix] = _counter.get(prefix, 0) + 1
        fig = plt.gcf()
        path = FIG_DIR / f'{prefix}_{_counter[prefix]:02d}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close('all')
        print(f'[บันทึกกราฟ] {path.relative_to(ROOT)}')
    plt.show = _save
