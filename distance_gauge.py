"""
Distance Gauge Engine for SHI Fuel Door Production Inspection
BC4I: Catch plug -> trimline distance measurement (nominal 128.4 mm +/- 0.7 mm).
Integrates with the live vision deployment pipeline.
"""

import os
import json
import logging
import cv2
import numpy as np

logger = logging.getLogger("SHI_Production")


class DistanceGauge:
    def __init__(self, model: str = "BC4I", base_dir: str = "gauge_cfg"):
        self.model = model
        self.cfg_dir = os.path.join(base_dir, model)
        self.cfg_file = os.path.join(self.cfg_dir, "config.json")
        self.is_valid = False

        if not os.path.exists(self.cfg_file):
            logger.warning(f"[DistanceGauge] Config file not found: {self.cfg_file}")
            return

        try:
            with open(self.cfg_file, "r") as f:
                self.cfg = json.load(f)

            self.nominal_mm = float(self.cfg.get("nominal_mm", 128.4))
            self.tolerance_mm = float(self.cfg.get("tolerance_mm", 0.7))
            self.mm_per_px = float(self.cfg.get("mm_per_px", 0.129828))
            self.axis = self.cfg.get("axis", "x")

            tmpl_a_path = os.path.join(self.cfg_dir, "template_A.png")
            tmpl_b_path = os.path.join(self.cfg_dir, "template_B.png")
            ref_path = os.path.join(self.cfg_dir, "reference.png")

            self.tmpl_A = cv2.imread(tmpl_a_path, cv2.IMREAD_GRAYSCALE)
            self.tmpl_B = cv2.imread(tmpl_b_path, cv2.IMREAD_GRAYSCALE)

            if self.tmpl_A is None or self.tmpl_B is None:
                logger.error(f"[DistanceGauge] Failed to load templates from {self.cfg_dir}")
                return

            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path)
                if ref_img is not None:
                    self.ref_h, self.ref_w = ref_img.shape[:2]
                else:
                    self.ref_h, self.ref_w = 2048, 2448
            else:
                self.ref_h, self.ref_w = 2048, 2448

            pr = self.cfg.get("part_roi", [0, 0, 100, 100])
            self.calib_center = (pr[0] + pr[2] / 2.0, pr[1] + pr[3] / 2.0)
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.is_valid = True
            logger.info(f"[DistanceGauge] Initialized {model}: Nominal={self.nominal_mm} +/- {self.tolerance_mm} mm, Ref Shape=({self.ref_w}x{self.ref_h})")

        except Exception as e:
            logger.error(f"[DistanceGauge] Initialization exception: {e}")
            self.is_valid = False

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return self.clahe.apply(gray)

    def _locate(self, gray: np.ndarray, key: str, tmpl: np.ndarray, offset_shift=(0.0, 0.0), margin: int = 280):
        roi = self.cfg[f"roi_{key}"]
        ox, oy = self.cfg[f"offset_{key}"]
        x = int(round(roi[0] + offset_shift[0]))
        y = int(round(roi[1] + offset_shift[1]))
        w, h = roi[2], roi[3]

        sx = max(x - margin, 0)
        sy = max(y - margin, 0)
        ex = min(x + w + margin, gray.shape[1])
        ey = min(y + h + margin, gray.shape[0])

        area = gray[sy:ey, sx:ex]
        if area.shape[0] < tmpl.shape[0] or area.shape[1] < tmpl.shape[1]:
            return None, 0.0

        res = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Parabolic subpixel refinement
        px, py = max_loc
        dx = dy = 0.0
        if 0 < px < res.shape[1] - 1:
            l, c, r = float(res[py, px - 1]), float(res[py, px]), float(res[py, px + 1])
            d = l - 2 * c + r
            if d != 0:
                dx = 0.5 * (l - r) / d
        if 0 < py < res.shape[0] - 1:
            t, c, b = float(res[py - 1, px]), float(res[py, px]), float(res[py + 1, px])
            d = t - 2 * c + b
            if d != 0:
                dy = 0.5 * (t - b) / d

        pt = (float(sx + px + dx + ox), float(sy + py + dy + oy))
        return pt, float(max_val)

    def measure(self, frame: np.ndarray, back_box=None, min_score: float = 0.60) -> dict:
        """
        Measures Catch Plug -> Trimline distance on the frame.
        Handles both 5MP industrial camera frames and resized video files seamlessly.
        """
        if not self.is_valid:
            return {"status": "UNCONFIGURED", "passed": False, "mm": None, "dev": None}

        h_in, w_in = frame.shape[:2]

        # Resolution normalization: bring frame to reference calibration resolution (ref_w, ref_h)
        if (w_in, h_in) != (self.ref_w, self.ref_h):
            work_frame = cv2.resize(frame, (self.ref_w, self.ref_h), interpolation=cv2.INTER_CUBIC)
            scale_x = self.ref_w / float(w_in)
            scale_y = self.ref_h / float(h_in)
            scaled_box = [back_box[0] * scale_x, back_box[1] * scale_y,
                          back_box[2] * scale_x, back_box[3] * scale_y] if back_box else None
        else:
            work_frame = frame
            scaled_box = back_box

        gray = self.preprocess(work_frame)

        shift = (0.0, 0.0)
        if scaled_box is not None:
            cur_cx = (scaled_box[0] + scaled_box[2]) / 2.0
            cur_cy = (scaled_box[1] + scaled_box[3]) / 2.0
            shift = (cur_cx - self.calib_center[0], cur_cy - self.calib_center[1])

        pt_a, score_a = self._locate(gray, "A", self.tmpl_A, shift, margin=280)
        pt_b, score_b = self._locate(gray, "B", self.tmpl_B, shift, margin=280)

        if pt_a is None or pt_b is None or min(score_a, score_b) < min_score:
            missing = []
            if score_a < min_score:
                missing.append("CATCH_PLUG")
            if score_b < min_score:
                missing.append("TRIMLINE")
            return {
                "status": "NOT_FOUND",
                "passed": False,
                "mm": None,
                "dev": None,
                "nominal": self.nominal_mm,
                "tolerance": self.tolerance_mm,
                "score_A": round(score_a, 3),
                "score_B": round(score_b, 3),
                "pt_A": pt_a,
                "pt_B": pt_b,
                "ref_w": self.ref_w,
                "ref_h": self.ref_h,
                "missing": missing
            }

        dist_px = abs(pt_b[0] - pt_a[0]) if self.axis == "x" else float(np.hypot(pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]))
        dist_mm = dist_px * self.mm_per_px
        dev = dist_mm - self.nominal_mm
        passed = abs(dev) <= (self.tolerance_mm + 1e-6)

        return {
            "status": "OK" if passed else "NG",
            "passed": bool(passed),
            "mm": round(float(dist_mm), 2),
            "dev": round(float(dev), 2),
            "nominal": self.nominal_mm,
            "tolerance": self.tolerance_mm,
            "score_A": round(score_a, 3),
            "score_B": round(score_b, 3),
            "pt_A": pt_a,
            "pt_B": pt_b,
            "ref_w": self.ref_w,
            "ref_h": self.ref_h,
            "missing": []
        }

    def draw_overlay(self, vis: np.ndarray, result: dict, scale: float = 1.0) -> np.ndarray:
        """
        Draws high-visibility measurement dimension line, end ticks, and mm badge directly onto vis frame.
        Properly maps reference coordinates to vis display coordinates.
        """
        if not result or result.get("pt_A") is None or result.get("pt_B") is None or result.get("mm") is None:
            return vis

        # Calculate coordinate scale from reference resolution to current vis frame
        ref_w = result.get("ref_w", self.ref_w)
        coord_scale = vis.shape[1] / float(ref_w)

        pt_a = (int(round(result["pt_A"][0] * coord_scale)), int(round(result["pt_A"][1] * coord_scale)))
        pt_b_x = int(round(result["pt_B"][0] * coord_scale))
        pt_b_y = pt_a[1] if self.axis == "x" else int(round(result["pt_B"][1] * coord_scale))
        pt_b = (pt_b_x, pt_b_y)

        is_ok = result.get("passed", False)
        col = (0, 230, 0) if is_ok else (0, 0, 255)  # Vivid Green for OK, Red for NG
        line_thick = max(2, int(round(3 * (vis.shape[1] / 1000.0))))

        # 1. Dimension line with black contrast border for 100% visibility on all backgrounds
        cv2.line(vis, pt_a, pt_b, (0, 0, 0), line_thick + 2)
        cv2.line(vis, pt_a, pt_b, col, line_thick)

        # 2. End tick marks & endpoints
        tick_len = max(8, int(round(16 * (vis.shape[1] / 1000.0))))
        for p in (pt_a, pt_b):
            cv2.line(vis, (p[0], p[1] - tick_len), (p[0], p[1] + tick_len), (0, 0, 0), line_thick + 2)
            cv2.line(vis, (p[0], p[1] - tick_len), (p[0], p[1] + tick_len), col, line_thick)
            cv2.circle(vis, p, max(4, int(round(5 * (vis.shape[1] / 1000.0)))), (0, 0, 0), -1)
            cv2.circle(vis, p, max(3, int(round(4 * (vis.shape[1] / 1000.0)))), col, -1)

        # 3. Target point labels (Catch Plug / Trimline)
        lbl_scale = max(0.38, 0.45 * (vis.shape[1] / 1000.0))
        for p, name, align_left in [(pt_a, "Catch Plug", False), (pt_b, "Trimline", True)]:
            (nw, nh), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, 1)
            lx = p[0] + 6 if align_left else p[0] - nw - 6
            ly = p[1] + tick_len + nh + 4
            cv2.rectangle(vis, (lx - 3, ly - nh - 3), (lx + nw + 3, ly + 3), (0, 0, 0), -1)
            cv2.rectangle(vis, (lx - 3, ly - nh - 3), (lx + nw + 3, ly + 3), col, 1)
            cv2.putText(vis, name, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, lbl_scale, (255, 255, 255), 1)

        # 4. Center Measurement text badge
        status_str = "OK" if is_ok else "NG"
        text = f"{result['mm']:.2f} mm ({result['dev']:+.2f}) [{status_str}]"
        font_scale = max(0.48, 0.70 * (vis.shape[1] / 1000.0))
        font_thick = max(1, int(round(2 * (vis.shape[1] / 1000.0))))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        mid_x = (pt_a[0] + pt_b[0]) // 2
        mid_y = pt_a[1] - max(14, int(round(18 * (vis.shape[1] / 1000.0))))

        bx1 = mid_x - tw // 2 - 8
        by1 = mid_y - th - 6
        bx2 = mid_x + tw // 2 + 8
        by2 = mid_y + 6
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), col, max(1, int(round(1.5 * (vis.shape[1] / 1000.0)))))
        cv2.putText(vis, text, (mid_x - tw // 2, mid_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

        return vis
