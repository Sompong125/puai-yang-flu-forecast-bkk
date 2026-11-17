"""จุดเริ่มรันทั้งโครงงาน:  python src/main.py
  python src/main.py                       # ทำความสะอาดข้อมูลดิบ (data/raw/) แล้ววิเคราะห์
  python src/main.py --skip-preprocess     # ข้ามขั้นทำความสะอาด ใช้ data/processed/ ที่มีให้แล้ว
  python src/main.py --data-dir D:/ข้อมูล   # ระบุที่วางไฟล์ดิบเอง
log เต็มบันทึกที่ results/output/run_log.txt
"""
import argparse, os, runpy, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from utils import OUT_DIR

class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for f in self.s: f.write(x)
    def flush(self):
        for f in self.s: f.flush()

ap = argparse.ArgumentParser()
ap.add_argument('--skip-preprocess', action='store_true')
ap.add_argument('--data-dir')
args = ap.parse_args()
if args.data_dir:
    os.environ['FLU_DATA_DIR'] = args.data_dir

with open(OUT_DIR / 'run_log.txt', 'w', encoding='utf-8') as log:
    sys.stdout = Tee(sys.__stdout__, log)
    try:
        if not args.skip_preprocess:
            print('=== ขั้นที่ 1: preprocess.py ===')
            runpy.run_path(str(SRC / 'preprocess.py'), run_name='__main__')
        print('\n=== ขั้นที่ 2: analysis.py ===')
        runpy.run_path(str(SRC / 'analysis.py'), run_name='__main__')
        print('\nเสร็จสิ้น: ดูกราฟใน results/figures/ และตารางใน results/output/')
    finally:
        sys.stdout = sys.__stdout__
