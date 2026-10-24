import os
import sqlite3
import re
import json
from datetime import datetime, date, timedelta, time

# Default DB Path
DB_PATH = os.path.join("lid_data", "inspections.db")

# 24-Hour Shift Boundaries:
# Shift A: 08:00 to 16:30
# Shift B: 16:30 to 01:00 (next day)
# Shift C: 01:00 to 08:00
SHIFT_A_START = time(8, 0, 0)
SHIFT_A_END   = time(16, 30, 0)
SHIFT_B_START = time(16, 30, 0)
SHIFT_B_END   = time(1, 0, 0)
SHIFT_C_START = time(1, 0, 0)
SHIFT_C_END   = time(8, 0, 0)

def get_shift_and_production_date(dt=None):
    """
    Computes the 24-hour shift ('A', 'B', 'C') and manufacturing production date.
    Shifts:
      - Shift A: 08:00 to 16:30
      - Shift B: 16:30 to 01:00 (crosses midnight)
      - Shift C: 01:00 to 08:00
    Production Day:
      A plant production day runs from 08:00 AM to 08:00 AM the following morning.
      Times from 00:00 to 07:59 belong to the previous calendar day's production cycle.
    """
    if dt is None:
        dt = datetime.now()
    
    t_min = dt.hour * 60 + dt.minute
    
    # 08:00 is 480 mins, 16:30 is 990 mins, 01:00 is 60 mins
    if 480 <= t_min < 990:
        shift = 'A'
        prod_date = dt.date()
    elif t_min >= 990:
        shift = 'B'
        prod_date = dt.date()
    elif t_min < 60:
        # 00:00 to 00:59:59 is still Shift B of the previous manufacturing date
        shift = 'B'
        prod_date = dt.date() - timedelta(days=1)
    else:
        # 01:00 to 07:59:59 is Shift C of the previous manufacturing date
        shift = 'C'
        prod_date = dt.date() - timedelta(days=1)
        
    return shift, prod_date.strftime("%Y-%m-%d")


def get_db_connection(db_path=DB_PATH):
    """Returns a SQLite connection with WAL mode enabled for high-concurrency and fast I/O."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path=DB_PATH):
    """Creates the inspections table and indexes if they do not already exist."""
    with get_db_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date_str TEXT NOT NULL,
                time_str TEXT NOT NULL,
                production_date TEXT NOT NULL,
                shift TEXT NOT NULL,
                serial TEXT NOT NULL,
                status TEXT NOT NULL,
                defects TEXT,
                confidence REAL,
                folder_path TEXT UNIQUE,
                report_path TEXT,
                defect_image TEXT,
                front_image TEXT,
                back_image TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prod_date ON inspections(production_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_shift ON inspections(shift);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON inspections(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON inspections(timestamp);")
        conn.commit()


def log_inspection(
    serial,
    status,
    confidence=0.0,
    defects=None,
    folder_path="",
    report_path="",
    defect_image="",
    front_image="",
    back_image="",
    custom_dt=None,
    db_path=DB_PATH
):
    """
    Safely logs an inspection cycle into SQLite with zero overhead.
    status: 'OK' / 'PASS' -> saved as 'OK', 'NG' / 'FAIL' -> saved as 'NG'
    """
    try:
        init_db(db_path)
        dt = custom_dt if custom_dt else datetime.now()
        
        # Normalize status to standard 'OK' or 'NG'
        norm_status = "NG" if (status and status.upper() in ["FAIL", "NG"]) or (defects and len(defects) > 0) or (defect_image and os.path.exists(defect_image)) else "OK"
        
        # Check if 14-char serial provides shift letter directly: DDMMYY(6)+NNN(3)+[ABC](1)+HHMM(4)
        shift_from_serial = None
        if serial and len(serial) == 14 and serial != "serial_missing":
            if re.fullmatch(r"\d{9}[ABC]\d{4}", serial):
                shift_from_serial = serial[9]
                
        calc_shift, prod_date = get_shift_and_production_date(dt)
        final_shift = calc_shift
        
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        defects_json = json.dumps(defects if defects else [])
        
        with get_db_connection(db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO inspections (
                    timestamp, date_str, time_str, production_date, shift,
                    serial, status, defects, confidence, folder_path,
                    report_path, defect_image, front_image, back_image
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp_str,
                date_str,
                time_str,
                prod_date,
                final_shift,
                serial,
                norm_status,
                defects_json,
                float(confidence) if confidence is not None else 0.0,
                folder_path,
                report_path,
                defect_image,
                front_image,
                back_image
            ))
            conn.commit()
            return True
    except Exception as e:
        print(f"[DB Logger] Warning: Failed to log inspection to DB: {e}")
        return False


def sync_existing_lid_data(lid_data_dir="lid_data", db_path=DB_PATH):
    """
    Scans the lid_data folder and imports any existing past inspections into SQLite.
    This ensures that historical scans are immediately available in the dashboard.
    """
    if not os.path.exists(lid_data_dir):
        return 0
        
    init_db(db_path)
    imported_count = 0
    
    with get_db_connection(db_path) as conn:
        existing_folders = set(
            row[0] for row in conn.execute("SELECT folder_path FROM inspections WHERE folder_path IS NOT NULL").fetchall()
        )
    
    for item in os.listdir(lid_data_dir):
        day_path = os.path.join(lid_data_dir, item)
        # Check if directory name matches YYYY-MM-DD
        if not os.path.isdir(day_path) or not re.match(r"^\d{4}-\d{2}-\d{2}$", item):
            continue
            
        for folder_name in os.listdir(day_path):
            folder_path = os.path.join(day_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_path in existing_folders:
                continue
                
            # Inspect folder contents
            files = os.listdir(folder_path)
            is_ng = folder_name.startswith("defect_") or any("defect" in f.lower() for f in files)
            status = "NG" if is_ng else "OK"
            
            # Determine serial
            clean_name = folder_name[7:] if folder_name.startswith("defect_") else folder_name
            serial = clean_name.split("_")[0] if "_" in clean_name else clean_name
            if "serial_missing" in folder_name:
                serial = "serial_missing"
            elif folder_name.startswith("temp_capture_"):
                serial = f"SCAN_{folder_name.split('_')[-1]}"
                
            # Locate pdf report and images
            report_path = ""
            defect_img = ""
            front_img = ""
            back_img = ""
            for f in files:
                f_low = f.lower()
                if f_low.endswith(".pdf"):
                    report_path = os.path.join(folder_path, f)
                elif "defect" in f_low and f_low.endswith((".jpg", ".png", ".jpeg")):
                    defect_img = os.path.join(folder_path, f)
                elif ("front" in f_low) and f_low.endswith((".jpg", ".png", ".jpeg")):
                    front_img = os.path.join(folder_path, f)
                elif ("back" in f_low) and f_low.endswith((".jpg", ".png", ".jpeg")):
                    back_img = os.path.join(folder_path, f)
                    
            # Get folder creation / mod time
            try:
                mtime = os.path.getmtime(folder_path)
                folder_dt = datetime.fromtimestamp(mtime)
            except Exception:
                folder_dt = datetime.now()
                
            # Try to infer defects if defect image exists
            defects = ["defect_detected"] if is_ng else []
            
            log_inspection(
                serial=serial,
                status=status,
                confidence=0.0,
                defects=defects,
                folder_path=folder_path,
                report_path=report_path,
                defect_image=defect_img,
                front_image=front_img,
                back_image=back_img,
                custom_dt=folder_dt,
                db_path=db_path
            )
            imported_count += 1
            
    return imported_count


if __name__ == "__main__":
    print("Initializing Database & testing sync...")
    init_db()
    count = sync_existing_lid_data()
    print(f"Sync complete. Backfilled {count} inspections into {DB_PATH}.")
