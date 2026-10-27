import os
import sys
import sqlite3
import json
import csv
import io
import webbrowser
from datetime import datetime, date, timedelta
from flask import Flask, jsonify, request, send_file, send_from_directory, Response

# Ensure db_logger is available
import db_logger

app = Flask(__name__, static_folder=".", static_url_path="")

# Disable Flask default request logging to keep console clean and fast
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def get_db():
    conn = db_logger.get_db_connection()
    return conn


@app.route("/")
def index():
    return send_file("dashboard.html")


@app.route("/logo_shi.webp")
def logo():
    if os.path.exists("logo_shi.webp"):
        return send_file("logo_shi.webp")
    return "", 404


@app.route("/sungwoo_logo.webp")
def logo_sungwoo():
    if os.path.exists("sungwoo_logo.webp"):
        return send_file("sungwoo_logo.webp")
    return "", 404


@app.route("/api/dates")
def get_dates():
    """Returns all available production dates and total counts."""
    conn = get_db()
    rows = conn.execute("""
        SELECT production_date, COUNT(*) as count 
        FROM inspections 
        GROUP BY production_date 
        ORDER BY production_date DESC
    """).fetchall()
    
    dates = [{"date": r["production_date"], "count": r["count"]} for r in rows]
    current_shift, current_prod_date = db_logger.get_shift_and_production_date()
    
    return jsonify({
        "dates": dates,
        "current_production_date": current_prod_date,
        "current_shift": current_shift
    })


@app.route("/api/summary")
def get_summary():
    """
    Returns shift-wise analytics, defect breakdowns, and hourly trends for a selected date.
    Query params:
      date: 'YYYY-MM-DD' (default: today's production date)
      shift: 'A', 'B', 'C', or 'all' (default: 'all')
    """
    selected_date = request.args.get("date")
    selected_shift = request.args.get("shift", "all").upper()
    
    current_shift, current_prod_date = db_logger.get_shift_and_production_date()
    if not selected_date:
        selected_date = current_prod_date
        
    conn = get_db()
    
    # 1. Total & Shift-wise summary for the day
    shift_query = """
        SELECT 
            shift,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) as ok_count,
            SUM(CASE WHEN status = 'NG' THEN 1 ELSE 0 END) as ng_count
        FROM inspections
        WHERE production_date = ?
        GROUP BY shift
    """
    rows = conn.execute(shift_query, (selected_date,)).fetchall()
    
    shifts_data = {
        "A": {"total": 0, "ok": 0, "ng": 0, "yield_rate": 0.0, "ng_rate": 0.0, "time_range": "08:00 - 16:30"},
        "B": {"total": 0, "ok": 0, "ng": 0, "yield_rate": 0.0, "ng_rate": 0.0, "time_range": "16:30 - 01:00"},
        "C": {"total": 0, "ok": 0, "ng": 0, "yield_rate": 0.0, "ng_rate": 0.0, "time_range": "01:00 - 08:00"}
    }
    
    total_scanned = 0
    total_ok = 0
    total_ng = 0
    
    for r in rows:
        s = r["shift"]
        if s in shifts_data:
            tot = r["total"] or 0
            ok = r["ok_count"] or 0
            ng = r["ng_count"] or 0
            shifts_data[s]["total"] = tot
            shifts_data[s]["ok"] = ok
            shifts_data[s]["ng"] = ng
            shifts_data[s]["yield_rate"] = round((ok / tot * 100), 1) if tot > 0 else 0.0
            shifts_data[s]["ng_rate"] = round((ng / tot * 100), 1) if tot > 0 else 0.0
            
            total_scanned += tot
            total_ok += ok
            total_ng += ng
            
    overall_yield = round((total_ok / total_scanned * 100), 1) if total_scanned > 0 else 0.0
    overall_ng_rate = round((total_ng / total_scanned * 100), 1) if total_scanned > 0 else 0.0
    
    # 2. Defect Breakdown
    defect_where = "WHERE production_date = ? AND status = 'NG'"
    defect_params = [selected_date]
    if selected_shift in ["A", "B", "C"]:
        defect_where += " AND shift = ?"
        defect_params.append(selected_shift)
        
    defect_rows = conn.execute(f"SELECT defects FROM inspections {defect_where}", defect_params).fetchall()
    defect_counts = {}
    for r in defect_rows:
        try:
            d_list = json.loads(r["defects"]) if r["defects"] else []
        except Exception:
            d_list = []
        if not d_list:
            d_list = ["Unspecified Defect"]
        for d in d_list:
            clean_d = d.replace("_", " ").title()
            defect_counts[clean_d] = defect_counts.get(clean_d, 0) + 1
            
    # Sort defects by count descending
    sorted_defects = sorted([{"name": k, "count": v} for k, v in defect_counts.items()], key=lambda x: x["count"], reverse=True)
    
    # 3. Hourly Production Trend (24-hour cycle)
    # We display hours 00 through 23
    hourly_where = "WHERE production_date = ?"
    hourly_params = [selected_date]
    if selected_shift in ["A", "B", "C"]:
        hourly_where += " AND shift = ?"
        hourly_params.append(selected_shift)
        
    hourly_rows = conn.execute(f"""
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) as ok_count,
            SUM(CASE WHEN status = 'NG' THEN 1 ELSE 0 END) as ng_count
        FROM inspections
        {hourly_where}
        GROUP BY hour
        ORDER BY hour ASC
    """, hourly_params).fetchall()
    
    hourly_map = {f"{h:02d}": {"total": 0, "ok": 0, "ng": 0} for h in range(24)}
    for r in hourly_rows:
        h = r["hour"]
        if h in hourly_map:
            hourly_map[h] = {
                "total": r["total"] or 0,
                "ok": r["ok_count"] or 0,
                "ng": r["ng_count"] or 0
            }
            
    hourly_list = [{"hour": f"{h}:00", **hourly_map[h]} for h in sorted(hourly_map.keys())]

    return jsonify({
        "production_date": selected_date,
        "selected_shift": selected_shift,
        "current_active_shift": current_shift,
        "overall": {
            "total_scanned": total_scanned,
            "total_ok": total_ok,
            "total_ng": total_ng,
            "yield_rate": overall_yield,
            "ng_rate": overall_ng_rate
        },
        "shifts": shifts_data,
        "defects": sorted_defects,
        "hourly_trend": hourly_list
    })


@app.route("/api/records")
def get_records():
    """
    Returns individual inspection scan records.
    Filters: date, shift ('A', 'B', 'C', 'all'), status ('OK', 'NG', 'all'), search
    """
    selected_date = request.args.get("date")
    selected_shift = request.args.get("shift", "all").upper()
    selected_status = request.args.get("status", "all").upper()
    search = request.args.get("search", "").strip()
    all_dates = request.args.get("all_dates", "0") == "1"
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    
    _, current_prod_date = db_logger.get_shift_and_production_date()
    if not selected_date:
        selected_date = current_prod_date
        
    where_clauses = []
    params = []

    # If searching across all dates or selected_date is 'all', omit the date filter
    if not (search and all_dates) and selected_date != "all":
        where_clauses.append("production_date = ?")
        params.append(selected_date)
    
    if selected_shift in ["A", "B", "C"]:
        where_clauses.append("shift = ?")
        params.append(selected_shift)
        
    if selected_status in ["OK", "NG"]:
        where_clauses.append("status = ?")
        params.append(selected_status)
        
    if search:
        where_clauses.append("(serial LIKE ? OR defects LIKE ? OR folder_path LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    
    conn = get_db()
    total_count = conn.execute(f"SELECT COUNT(*) FROM inspections {where_sql}", params).fetchone()[0]
    
    query = f"""
        SELECT id, timestamp, time_str, shift, serial, status, defects, confidence,
               folder_path, report_path, defect_image, front_image, back_image
        FROM inspections
        {where_sql}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    
    records = []
    for r in rows:
        try:
            d_list = json.loads(r["defects"]) if r["defects"] else []
        except Exception:
            d_list = []
            
        file_paths = resolve_record_paths(
            folder_path=r["folder_path"],
            report_path=r["report_path"] or "",
            defect_image=r["defect_image"] or "",
            front_image=r["front_image"] or "",
            back_image=r["back_image"] or ""
        )
            
        record_data = {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "time_str": r["time_str"],
            "shift": r["shift"],
            "serial": r["serial"],
            "status": r["status"],
            "defects": d_list,
            "confidence": r["confidence"],
            "folder_path": r["folder_path"]
        }
        record_data.update(file_paths)
        records.append(record_data)
        
    return jsonify({
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "records": records
    })


def resolve_record_paths(folder_path, report_path="", defect_image="", front_image="", back_image=""):
    """
    Robustly resolves and verifies physical paths for reports and images.
    If a stored path is missing or does not exist on disk, scans the folder
    to locate the actual report PDF and captured inspection images.
    Returns normalized forward-slash relative web paths.
    """
    res = {
        "report_path": "",
        "has_report": False,
        "defect_image": "",
        "has_defect_image": False,
        "front_image": "",
        "has_front_image": False,
        "back_image": "",
        "has_back_image": False,
        "photo_path": "",
        "has_photo": False,
        "photos": []
    }
    
    # 1. Verify or find report_path
    if report_path and os.path.exists(report_path):
        res["report_path"] = report_path.replace("\\", "/")
        res["has_report"] = True
    elif folder_path and os.path.isdir(folder_path):
        try:
            files = os.listdir(folder_path)
            pdfs = [f for f in files if f.lower().endswith(".pdf")]
            if pdfs:
                res["report_path"] = os.path.join(folder_path, pdfs[0]).replace("\\", "/")
                res["has_report"] = True
        except Exception:
            pass

    # 2. Verify or find defect_image
    if defect_image and os.path.exists(defect_image):
        res["defect_image"] = defect_image.replace("\\", "/")
        res["has_defect_image"] = True
    elif folder_path and os.path.isdir(folder_path):
        try:
            files = os.listdir(folder_path)
            dis = [f for f in files if "defect" in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if dis:
                res["defect_image"] = os.path.join(folder_path, dis[0]).replace("\\", "/")
                res["has_defect_image"] = True
        except Exception:
            pass

    # 3. Verify or find front_image
    if front_image and os.path.exists(front_image):
        res["front_image"] = front_image.replace("\\", "/")
        res["has_front_image"] = True
    elif folder_path and os.path.isdir(folder_path):
        try:
            files = os.listdir(folder_path)
            fis = [f for f in files if "front" in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if fis:
                res["front_image"] = os.path.join(folder_path, fis[0]).replace("\\", "/")
                res["has_front_image"] = True
        except Exception:
            pass

    # 4. Verify or find back_image
    if back_image and os.path.exists(back_image):
        res["back_image"] = back_image.replace("\\", "/")
        res["has_back_image"] = True
    elif folder_path and os.path.isdir(folder_path):
        try:
            files = os.listdir(folder_path)
            bis = [f for f in files if "back" in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if bis:
                res["back_image"] = os.path.join(folder_path, bis[0]).replace("\\", "/")
                res["has_back_image"] = True
        except Exception:
            pass

    # 5. Build photo list and determine primary photo
    photos = []
    if res["has_defect_image"]:
        photos.append({"type": "defect", "label": "Defect View", "path": res["defect_image"]})
    if res["has_front_image"]:
        photos.append({"type": "front", "label": "Front View", "path": res["front_image"]})
    if res["has_back_image"]:
        photos.append({"type": "back", "label": "Back View", "path": res["back_image"]})
        
    # If no defect/front/back tagged image was found but other images exist in folder:
    if not photos and folder_path and os.path.isdir(folder_path):
        try:
            files = os.listdir(folder_path)
            other_imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for img in other_imgs:
                p = os.path.join(folder_path, img).replace("\\", "/")
                photos.append({"type": "other", "label": img, "path": p})
        except Exception:
            pass

    res["photos"] = photos
    if photos:
        res["has_photo"] = True
        res["photo_path"] = photos[0]["path"]
        
    return res


@app.route("/api/view_file")
def view_file():
    """Safely serves PDF reports or inspection images from lid_data."""
    file_path = request.args.get("path")
    if not file_path:
        return "File path missing", 400
        
    clean_path = os.path.normpath(file_path.strip().replace("/", os.sep))
    abs_path = os.path.abspath(clean_path)
    base_dir = os.path.abspath("lid_data")
    
    # Secure normalized case-insensitive directory check for Windows
    if not os.path.normcase(abs_path).startswith(os.path.normcase(base_dir)):
        return "Access denied", 403
        
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        return "File not found", 404
        
    ext = os.path.splitext(abs_path)[1].lower()
    mimetype = "application/pdf" if ext == ".pdf" else (
        "image/jpeg" if ext in [".jpg", ".jpeg"] else (
            "image/png" if ext == ".png" else None
        )
    )
    
    from flask import make_response
    response = make_response(send_file(abs_path, mimetype=mimetype))
    response.headers["Content-Disposition"] = f"inline; filename=\"{os.path.basename(abs_path)}\""
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/api/open_folder", methods=["POST", "GET"])
def open_folder():
    """Opens the inspection folder directly in Windows File Explorer."""
    folder_path = request.args.get("path")
    if not folder_path:
        return jsonify({"status": "error", "message": "Folder path missing"}), 400
        
    clean_path = os.path.normpath(folder_path.strip().replace("/", os.sep))
    abs_path = os.path.abspath(clean_path)
    base_dir = os.path.abspath("lid_data")
    if not os.path.normcase(abs_path).startswith(os.path.normcase(base_dir)):
        return jsonify({"status": "error", "message": "Access denied"}), 403
        
    if not os.path.exists(abs_path):
        return jsonify({"status": "error", "message": "Folder not found"}), 404
        
    try:
        if sys.platform == "win32" and hasattr(os, "startfile"):
            os.startfile(abs_path)
        else:
            import subprocess
            subprocess.Popen(["explorer", abs_path])
        return jsonify({"status": "success", "message": "Folder opened in File Explorer"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/open_file", methods=["POST", "GET"])
def open_file():
    """Opens a PDF report or image directly in the default Windows desktop viewer."""
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"status": "error", "message": "File path missing"}), 400
        
    clean_path = os.path.normpath(file_path.strip().replace("/", os.sep))
    abs_path = os.path.abspath(clean_path)
    base_dir = os.path.abspath("lid_data")
    if not os.path.normcase(abs_path).startswith(os.path.normcase(base_dir)):
        return jsonify({"status": "error", "message": "Access denied"}), 403
        
    if not os.path.exists(abs_path):
        return jsonify({"status": "error", "message": "File not found"}), 404
        
    try:
        if sys.platform == "win32" and hasattr(os, "startfile"):
            os.startfile(abs_path)
        else:
            import subprocess
            subprocess.Popen(["start", "", abs_path], shell=True)
        return jsonify({"status": "success", "message": "File opened in system viewer"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/export_csv")
def export_csv():
    """Generates a downloadable CSV of inspection records for quality auditing."""
    selected_date = request.args.get("date")
    selected_shift = request.args.get("shift", "all").upper()
    
    _, current_prod_date = db_logger.get_shift_and_production_date()
    if not selected_date:
        selected_date = current_prod_date
        
    where_clauses = ["production_date = ?"]
    params = [selected_date]
    if selected_shift in ["A", "B", "C"]:
        where_clauses.append("shift = ?")
        params.append(selected_shift)
        
    where_sql = "WHERE " + " AND ".join(where_clauses)
    
    conn = get_db()
    rows = conn.execute(f"""
        SELECT timestamp, time_str, production_date, shift, serial, status, defects, confidence, folder_path
        FROM inspections
        {where_sql}
        ORDER BY timestamp ASC
    """, params).fetchall()
    
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["Timestamp", "Time", "Production Date", "Shift", "Serial Number", "Result Status", "Defects Detected", "Confidence %", "Folder Path"])
    
    for r in rows:
        try:
            d_list = json.loads(r["defects"]) if r["defects"] else []
            d_str = "; ".join(d_list)
        except Exception:
            d_str = r["defects"] or ""
            
        writer.writerow([
            r["timestamp"],
            r["time_str"],
            r["production_date"],
            f"Shift {r['shift']}",
            r["serial"],
            r["status"],
            d_str,
            f"{r['confidence']:.1f}%",
            r["folder_path"]
        ])
        
    filename = f"fuel_door_inspections_{selected_date}_Shift_{selected_shift}.csv"
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )


@app.route("/api/sync", methods=["POST", "GET"])
def sync_data():
    """Triggers an on-demand scan of lid_data to index any uncatalogued scans."""
    count = db_logger.sync_existing_lid_data()
    return jsonify({"status": "success", "synced_new_records": count})


def kill_process_on_port(port=5050):
    try:
        import subprocess
        output = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode('utf-8', errors='ignore')
        my_pid = str(os.getpid())
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and 'LISTENING' in line:
                pid = parts[-1]
                if pid != my_pid:
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


if __name__ == "__main__":
    PORT = 5050
    kill_process_on_port(PORT)
    # Ensure database is created and past scans in lid_data are indexed
    db_logger.init_db()
    db_logger.sync_existing_lid_data()
    print("=" * 60)
    print("  SHI FUEL DOOR - SHIFT ANALYTICS DASHBOARD")
    print(f"  Running on: http://localhost:{PORT}")
    print("  Low-Memory Standalone Process (Zero GPU / AI imports)")
    print("=" * 60)
    
    # Auto-open browser on launch if not disabled
    if "--no-browser" not in sys.argv:
        try:
            import threading
            threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
        except Exception:
            pass

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
