# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:12:56 2025

@author: Kenneth Kwabena Kenney
"""
# Library imports for script
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import ezdxf
import copy
import datetime
from openpyxl import Workbook
from openpyxl.styles import Font

# Library imports for GUI 
import sys
import time
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QCheckBox,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QMessageBox, QProgressBar, QSplashScreen
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QTextCursor, QIcon
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------


# -----------------------------
# GLOBALS / NAMESPACES / CONFIG
# -----------------------------
dummy_holes = {"LR", "RR", "FS", "BS"}

namespaces = {
    'IR': 'http://www.iredes.org/xml',
    '': 'http://www.iredes.org/xml/DrillRig'
}
for prefix, uri in namespaces.items():
    ET.register_namespace(prefix, uri)

meters_to_usft = 1 / 0.3048006096  # convert meters -> US survey ft


# -----------------------------
# UTILS / FUNCTIONS
# -----------------------------
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundles."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Regex based key extraction 1
def extract_design_key(filename):
    """
    For design files like: base_R{n}.xml
    Returns tuple (base, ring_id) e.g. ('xxx', 'R1') or (None,None)
    """
    m = re.match(r'^(?P<base>.+?)_(R\d+)\.xml$', filename, flags=re.IGNORECASE)
    if not m:
        return None, None
    base = m.group('base')
    ring = re.search(r'_(R\d+)\.xml$', filename, flags=re.IGNORECASE).group(1)
    return base, ring

# Regex based key extraction 2
def extract_dq_key(filename):
    """
    For DQ files like: DQ{base}_R{n}_whatever.xml
    Returns (base, ring_id)
    """
    m = re.match(r'^DQ(?P<base>.+?)_(R\d+)', filename, flags=re.IGNORECASE)
    if not m:
        return None, None
    base = m.group('base')
    ring = re.search(r'_(R\d+)', filename, flags=re.IGNORECASE).group(1)
    return base, ring

# Parse holes
def parse_holes(xml_path, is_dq=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    plan_id = root.find('.//IR:PlanNameRef', namespaces).text if is_dq and root.find('.//IR:PlanNameRef', namespaces) is not None \
              else root.find('.//IR:PlanName', namespaces).text if not is_dq and root.find('.//IR:PlanName', namespaces) is not None \
              else "UNKNOWN"

    holes_data = []
    hole_nodes = root.findall('.//HoleQualityData/Hole', namespaces) if is_dq else root.findall('.//DrillPlan/Hole', namespaces)

    for hole in hole_nodes:
        hole_name_node = hole.find('HoleName', namespaces)
        if hole_name_node is None:
            continue
        hole_name = hole_name_node.text.strip()
        start = hole.find('StartPoint', namespaces)
        end = hole.find('EndPoint', namespaces)
        if start is None or end is None:
            continue

        holes_data.append({
            "RingID": plan_id,
            "HoleName": hole_name,
            "CollarX": float(start.find('IR:PointX', namespaces).text),
            "CollarY": float(start.find('IR:PointY', namespaces).text),
            "CollarZ": float(start.find('IR:PointZ', namespaces).text),
            "ToeX": float(end.find('IR:PointX', namespaces).text),
            "ToeY": float(end.find('IR:PointY', namespaces).text),
            "ToeZ": float(end.find('IR:PointZ', namespaces).text)
        })
    return holes_data

# Parse DQ metadata
def parse_dq_metadata(xml_path, is_dq=True):
    """
    Parse DQ XML and return a list of dicts, one per HoleQualityData block:
    RingID, HoleID, DrillSequence, AvgPenetration, HoleStatus, DrilledInRock
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # reuse your global 'namespaces' dict (must be defined in module)
    # e.g. namespaces = {'IR': 'http://www.iredes.org/xml', '': 'http://www.iredes.org/xml/DrillRig'}
    # We will use the IR prefix for PlanNameRef if present.
    ns = namespaces

    # safe helper
    def get_text(node):
        return node.text.strip() if node is not None and node.text else ""

    # Ring/plan id (use IR:PlanNameRef for DQ files if present)
    ring_id = ""
    try:
        ring_id = root.findtext('.//IR:PlanNameRef', default="", namespaces=ns) or ""
    except Exception:
        # fallback: try non-namespaced PlanNameRef
        ring_id = root.findtext('.//PlanNameRef', default="") or ""

    results = []

    # Iterate each HoleQualityData block — this contains both <Hole> and the per-block metadata
    # Use namespaced search; if your file has no namespace for these nodes, the findall will still work
    # because you've used the same pattern elsewhere. If it doesn't find them, try non-ns below.
    hqd_nodes = root.findall('.//HoleQualityData', ns)
    if not hqd_nodes:
        # fallback without namespaces
        hqd_nodes = root.findall('.//HoleQualityData')

    for hqd in hqd_nodes:
        # hole subnode may be namespaced or not; try both
        hole_node = hqd.find('Hole', ns) or hqd.find('Hole')
        if hole_node is None:
            continue

        # hole name (inside Hole)
        hole_name = hole_node.findtext('HoleName', default="", namespaces=ns) or hole_node.findtext('HoleName', default="") or ""
        hole_name = hole_name.strip()

        # per-block metadata
        # try namespaced then fallback to non-namespaced
        drill_seq = hqd.findtext('DrillSeq', default="", namespaces=ns) or hqd.findtext('DrillSeq', default="") or ""
        avg_pen   = hqd.findtext('AvgPenetr', default="", namespaces=ns) or hqd.findtext('AvgPenetr', default="") or ""
        hstatus   = hqd.findtext('Hstatus', default="", namespaces=ns) or hqd.findtext('Hstatus', default="") or ""
        drilled_in_rock = hqd.findtext('DrilledInRock', default="", namespaces=ns) or hqd.findtext('DrilledInRock', default="") or ""

        results.append({
            "RingID": ring_id,
            "HoleID": hole_name,
            "DrillSeq": drill_seq.strip() if drill_seq else "",
            "AvgPenetr": avg_pen.strip() if avg_pen else "",
            "HoleStatus": hstatus.strip() if hstatus else "",
            "DrilledInRock": drilled_in_rock.strip() if drilled_in_rock else ""
        })

    return results

# Align drilled holes in mine coord frame using rotated deviations using best-fit transform
def best_fit_transform(A, B):
    """
    Find rotation R and translation t such that B ≈ R*A + t
    A, B: (N,3) arrays
    """
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


# -----------------------------
# NAVDRILL PROCESS FUNCTION
# -----------------------------
def process_navdrill(input_folder, mine_excel, want_dxf=False, want_iredes=False, log_callback=None, runner=None):
    def log(msg): 
        if log_callback: log_callback(msg)
        else: print(msg)
    
    # Make output folders and paths | create output titles
    output_parent = os.path.join(input_folder, "NavDQ Output")
    os.makedirs(output_parent, exist_ok=True)
    metadata_output = os.path.join(output_parent, "Drill Quality Report.xlsx")
    
    # Build file maps and pairs
    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.xml')]
    design_files = [f for f in all_files if re.search(r'_R\d+\.xml$', f, flags=re.IGNORECASE)]
    dq_files = [f for f in all_files if f.lower().startswith('dq') and re.search(r'_R\d+', f, flags=re.IGNORECASE)]

    pairs = {}  # key = (base, ring) -> { 'design': filename, 'dq': filename }
    design_only = []
    dq_only = []

    # Map design files
    for f in design_files:
        base, ring = extract_design_key(f)
        if base is None:
            continue
        key = (base, ring.upper())
        pairs.setdefault(key, {})['design'] = f

    # Map DQ files
    for f in dq_files:
        base, ring = extract_dq_key(f)
        if base is None:
            continue
        key = (base, ring.upper())
        pairs.setdefault(key, {})['dq'] = f

    # Identify unmatched
    valid_pairs = []
    for key, d in pairs.items():
        has_design = 'design' in d
        has_dq = 'dq' in d
        if has_design and has_dq:
            valid_pairs.append((key, d['design'], d['dq']))
        else:
            if has_design and not has_dq:
                design_only.append((key, d.get('design')))
            if has_dq and not has_design:
                dq_only.append((key, d.get('dq')))

    # Report unmatched and prompt user
    if design_only or dq_only:
        log("⚠️  The following unmatched rings were found in the folder:")
        if design_only:
            log(" \nDesign rings:")
            for key, fname in design_only:
                log(f" {fname}")
        if dq_only:
            log("  \nDQ rings:")
            for key, fname in dq_only:
                log(f" {fname}")
                
        # Reset worker decision
        runner.user_decision = None
    
        # Ask GUI to show popup (runs on main GUI thread)
        runner.request_continue_abort.emit()
    
        # Poll until GUI sets the decision
        while runner.user_decision is None:
            time.sleep(0.05)  # keeps CPU low, does NOT block GUI
    
        # Act on decision
        if not runner.user_decision:
            log("\n\nAborting - no ring pairs processed. \nFind mising ring pairs for umatched rings and reprocess.\n\n")
            return

    if not valid_pairs:
        log("\n\nNo valid design+DQ pairs found. Exiting.\n\n")
        runner.no_valid_pairs_exit = True
        return

    log(f"\n\n🔎 Found {len(valid_pairs)} valid ring pair(s). Processing...")

    # Load mine coordinates once (same sheet for all rings)    
    try:
        if mine_excel.lower().endswith('.csv'):
            df_mine_full = pd.read_csv(mine_excel)
        else:
            df_mine_full = pd.read_excel(mine_excel)
    except Exception as e:
        raise RuntimeError(f"\n\nFailed to read design hole data file '{mine_excel}': {e}")
    df_mine_full = df_mine_full.rename(columns={'HoleID':'HoleName'}) #rename to HoleName when merging)

    # We'll accumulate meta records for the combined excel drill quality report
    combined_meta_frames = []
        
    # Process each valid pair
    start_time = time.time()
    for (base_ring_key, design_fname, dq_fname) in valid_pairs:
        base, ring = base_ring_key
        log(f"\n\n➡️ Processing {ring}: \ndesign={design_fname} \ndq={dq_fname}")

        design_path = os.path.join(input_folder, design_fname)
        dq_path = os.path.join(input_folder, dq_fname)

        # ---- Extract holes & metadata for this pair ----
        design_data = parse_holes(design_path, is_dq=False)
        dq_data = parse_holes(dq_path, is_dq=True)
        dq_metadata_list = parse_dq_metadata(dq_path, is_dq=True)

        df_design = pd.DataFrame(design_data)
        df_dq = pd.DataFrame(dq_data)
        df_dq_metadata = pd.DataFrame(dq_metadata_list)

        if df_design.empty:
            log(f"  \n⚠️ No design holes found in {design_fname} — skipping this pair.")
            continue
        if df_dq.empty:
            log(f"  \n⚠️ No DQ holes found in {dq_fname} — skipping this pair.")
            continue

        # Extract FS/BS/RR/LR points for this DQ (if present)
        fs_point_dq = None; bs_point_dq = None; rr_point_dq = None; lr_point_dq = None
        if 'HoleName' in df_dq.columns:
            try:
                if any(df_dq['HoleName'].str.contains('FS', case=False, na=False)):
                    fs_point_dq = df_dq.loc[df_dq['HoleName'].str.contains('FS', case=False), 
                                                      ['CollarX','CollarY','CollarZ']].values[0]
            except Exception:
                fs_point_dq = None
            try:
                if any(df_dq['HoleName'].str.contains('BS', case=False, na=False)):
                    bs_point_dq = df_dq.loc[df_dq['HoleName'].str.contains('BS', case=False), 
                                                      ['CollarX','CollarY','CollarZ']].values[0]
            except Exception:
                bs_point_dq = None
            try:
                if any(df_dq['HoleName'].str.contains('RR', case=False, na=False)):
                    rr_point_dq = df_dq.loc[df_dq['HoleName'].str.contains('RR', case=False), 
                                                      ['CollarX','CollarY','CollarZ']].values[0]
            except Exception:
                rr_point_dq = None
            try:
                if any(df_dq['HoleName'].str.contains('LR', case=False, na=False)):
                    lr_point_dq = df_dq.loc[df_dq['HoleName'].str.contains('LR', case=False), 
                                                      ['CollarX','CollarY','CollarZ']].values[0]
            except Exception:
                lr_point_dq = None

        # Remove dummy holes
        df_design = df_design[~df_design['HoleName'].isin(dummy_holes)].reset_index(drop=True)
        df_dq = df_dq[~df_dq['HoleName'].isin(dummy_holes)].reset_index(drop=True)
        df_dq_metadata = df_dq_metadata[~df_dq_metadata['HoleID'].isin(dummy_holes)].reset_index(drop=True)
        
        # Sort
        df_design = df_design.sort_values(by=["RingID","HoleName"]).reset_index(drop=True)
        df_dq = df_dq.sort_values(by=["RingID","HoleName"]).reset_index(drop=True)
        df_dq_metadata = df_dq_metadata.sort_values(by=["RingID","HoleID"]).reset_index(drop=True)

        # ---- Merge mine coordinates into design for this pair ----
        # Use only the relevant mine rows (match by RingID & HoleName)
        # First create a filtered mine df for this ring (if RingID exists in df_design)
        ring_ids = df_design['RingID'].unique().tolist()
        # Filter the mine sheet by RingID(s), if RingID column exists in mine sheet
        df_mine = df_mine_full.copy()
        if 'RingID' in df_mine.columns:
            df_mine = df_mine[df_mine['RingID'].isin(ring_ids)]
        # rename HoleID -> HoleName if necessary (done earlier globally)

        df_design_mine = pd.merge(df_design, df_mine, on=['RingID','HoleName'], suffixes=('_extracted','_mine'))

        cols_keep = ['RingID','HoleName','CollarX','CollarY','CollarZ','ToeX','ToeY','ToeZ']

        # if merge didn't find replacements for some holes, warn but continue
        if df_design_mine.empty:
            log(f"  \n⚠️ No matching mine coordinates found for design '{design_fname}'. Skipping pair.")
            continue

        # Replace extracted design coords with mine coords (if present)
        for col in ['CollarX','CollarY','CollarZ','ToeX','ToeY','ToeZ']:
            mine_col = f'{col}_mine'
            if mine_col in df_design_mine.columns:
                df_design_mine[col] = df_design_mine[mine_col]
        df_design_mine = df_design_mine[cols_keep]

        # ---- Convert original extracted coords to US survey ft ----
        df_design_ft = df_design.copy()
        df_dq_ft = df_dq.copy()
        for col in ['CollarX','CollarY','CollarZ','ToeX','ToeY','ToeZ']:
            df_design_ft[col] = df_design_ft[col]*meters_to_usft
            df_dq_ft[col] = df_dq_ft[col]*meters_to_usft

        if fs_point_dq is not None:
            try:
                fs_point_dq = fs_point_dq * meters_to_usft
            except Exception:
                fs_point_dq = None
        if bs_point_dq is not None:
            try:
                bs_point_dq = bs_point_dq * meters_to_usft
            except Exception:
                bs_point_dq = None
        if rr_point_dq is not None:
            try:
                rr_point_dq = rr_point_dq * meters_to_usft
            except Exception:
                rr_point_dq = None
        if lr_point_dq is not None:
            try:
                lr_point_dq = lr_point_dq * meters_to_usft
            except Exception:
                lr_point_dq = None

        # ---- Compute deviations (DQ - Design) in US survey ft ----
        df_delta = df_dq_ft.copy()
        delta_cols = ['CollarX','CollarY','CollarZ','ToeX','ToeY','ToeZ']
        for col in delta_cols:
            if col in df_dq_ft.columns and col in df_design_ft.columns:
                df_delta[col] = df_dq_ft[col] - df_design_ft[col]

        # # Get matching points for alignment
        # A = df_design_ft[['CollarX','CollarY','CollarZ','ToeX','ToeY','ToeZ']].values.reshape(-1, 6)
        # B = df_design_mine[['CollarX','CollarY','CollarZ','ToeX','ToeY','ToeZ']].values.reshape(-1, 6)

        try:
            A_points = np.vstack([df_design_ft[['CollarX','CollarY','CollarZ']].values,
                                  df_design_ft[['ToeX','ToeY','ToeZ']].values])
            B_points = np.vstack([df_design_mine[['CollarX','CollarY','CollarZ']].values,
                                  df_design_mine[['ToeX','ToeY','ToeZ']].values])
        except Exception as e:
            log(f"  \n⚠️ Error building A/B points: {e}  -> skipping pair.")
            continue

        # -------- HANDLE SINGLE-HOLE SCENARIO AMD MULTI HOLES ---------------
        df_dq_mine = df_design_mine.copy()
        
        if A_points.shape[0] < 4:
            log("\n⚠️ Single hole detected — using directional + DQ axis alignment.")
        
            # --- Extract single rows (already in ft) ---
            design_local_collar = df_design_ft[['CollarX','CollarY','CollarZ']].values[0]
            dq_local_collar     = df_dq_ft[['CollarX','CollarY','CollarZ']].values[0]
            design_local_toe    = df_design_ft[['ToeX','ToeY','ToeZ']].values[0]
            dq_local_toe        = df_dq_ft[['ToeX','ToeY','ToeZ']].values[0]
        
            design_mine_collar = df_design_mine[['CollarX','CollarY','CollarZ']].values[0]
            design_mine_toe    = df_design_mine[['ToeX','ToeY','ToeZ']].values[0]
        
            # 1️⃣ Compute local deltas
            delta_collar = dq_local_collar - design_local_collar
            delta_toe    = dq_local_toe    - design_local_toe
        
            use_fallback = False
        
            # 2️⃣ Try to build DQ local axes from FS/BS/LR/RR
            try:
                if all(x in globals() for x in ['fs_point_dq', 'bs_point_dq', 'lr_point_dq', 'rr_point_dq']):
                    y_local = fs_point_dq - bs_point_dq
                    y_local = y_local / np.linalg.norm(y_local)
                    x_local = lr_point_dq - rr_point_dq
                    x_local = x_local / np.linalg.norm(x_local)
                    axis_ok = True
                else:
                    axis_ok = False
            except Exception:
                axis_ok = False
        
            if axis_ok:
                # 3️⃣ Build mine axes
                y_mine = design_mine_toe - design_mine_collar
                y_mine = y_mine / np.linalg.norm(y_mine)
                up = np.array([0, 0, 1.0])
                x_mine = np.cross(y_mine, up)
                if np.linalg.norm(x_mine) < 1e-12:
                    x_mine = np.array([1.0, 0.0, 0.0])
                else:
                    x_mine = x_mine / np.linalg.norm(x_mine)
        
                # 4️⃣ Determine sign flips
                flip_x = False
                flip_y = False
                x_local_h = x_local[:2]
                x_mine_h  = x_mine[:2]
                y_local_h = y_local[:2]
                y_mine_h  = y_mine[:2]
                if np.dot(x_local_h, x_mine_h) < 0: flip_x = True
                if np.dot(y_local_h, y_mine_h) < 0: flip_y = True
        
                if flip_x:
                    delta_collar[0] *= -1; delta_toe[0] *= -1
                if flip_y:
                    delta_collar[1] *= -1; delta_toe[1] *= -1
        
                log("\n✅ Using DQ axis alignment (FS/BS/LR/RR).")
        
            else:
                log("\n⚠️ FS/BS/LR/RR missing or invalid → using simple directional alignment fallback.")
                use_fallback = True
        
            if use_fallback:
                # --- SIMPLE DIRECTIONAL ALIGNMENT FALLBACK ---
                dir_local = design_local_toe - design_local_collar
                dir_mine  = design_mine_toe  - design_mine_collar
                dir_local = dir_local / np.linalg.norm(dir_local)
                dir_mine  = dir_mine / np.linalg.norm(dir_mine)
        
                v = np.cross(dir_local, dir_mine)
                s = np.linalg.norm(v)
                c = np.dot(dir_local, dir_mine)
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                if s == 0:  # no rotation needed
                    R = np.eye(3)
                else:
                    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
                delta_collar = (R @ delta_collar.T).T
                delta_toe    = (R @ delta_toe.T).T
        
            # 5️⃣ Apply deltas directly (same units)
            drilled_mine_collar = design_mine_collar + delta_collar
            drilled_mine_toe    = design_mine_toe + delta_toe
        
            # Store back into dataframe
            df_dq_mine = df_design_mine.copy()
            df_dq_mine.loc[0, ['CollarX','CollarY','CollarZ']] = drilled_mine_collar
            df_dq_mine.loc[0, ['ToeX','ToeY','ToeZ']] = drilled_mine_toe
        
        else:
            # Multi-hole case: use best-fit transform
            R, t = best_fit_transform(A_points, B_points)
             
            # Apply the rotated deltas to design_mine
            df_dq_mine = df_design_mine.copy()
            
            for part in ['Collar', 'Toe']:
                local_design = df_design_ft[[f'{part}X', f'{part}Y', f'{part}Z']].values
                local_drilled = df_dq_ft[[f'{part}X', f'{part}Y', f'{part}Z']].values
                delta_local = local_drilled - local_design  # deviation in local
                delta_rotated = (R @ delta_local.T).T       # rotate deviation only
                design_mine = df_design_mine[[f'{part}X', f'{part}Y', f'{part}Z']].values
                drilled_mine = design_mine + delta_rotated  # add rotated deviation
                df_dq_mine[[f'{part}X', f'{part}Y', f'{part}Z']] = drilled_mine

        # DXF creation
        if want_dxf:
            output_dxf = os.path.join(output_parent, "DXF")
            os.makedirs(output_dxf, exist_ok=True)
            try:
                log("\n🔧 Creating DXF with mine-aligned as-drilled coordinates...")
                first_design_file = design_fname
                design_name = os.path.splitext(first_design_file)[0]
                dxf_path = os.path.join(output_dxf, f"{design_name}.dxf")
                doc = ezdxf.new(dxfversion='R2010')
                msp = doc.modelspace()
                
                # Add all drilled holes as 3D polylines (blue color)
                for idx, row in df_dq_mine.iterrows():
                    msp.add_polyline3d(
                        [(row['CollarX'], row['CollarY'], row['CollarZ']),
                         (row['ToeX'], row['ToeY'], row['ToeZ'])],
                        dxfattribs={'color': 5}  # blue
                    )
                
                # Save DXF
                doc.saveas(dxf_path)
                log(f"\n✅ Drilled holes 3D DXF saved → {dxf_path}")
            except Exception as e:
                log(f"  \n⚠️ Failed to create DXF for {design_fname}: {e}")
        
        # IREDES creation
        if want_iredes:
            output_iredes = os.path.join(output_parent, "DeswikImport_IREDES")
            os.makedirs(output_iredes, exist_ok=True)
            try:
                log("\n🔧 Creating new IREDES XML with mine-aligned as-drilled coordinates...")
                
                dq_tree = ET.parse(dq_path)
                dq_root = dq_tree.getroot()
                dq_root_copy = copy.deepcopy(dq_root)
    
                dq_output_path = os.path.join(output_iredes, f"{os.path.splitext(dq_fname)[0]}_MineAligned.xml")
                dq_mine_dict = df_dq_mine.set_index('HoleName').to_dict(orient='index')
                
                #  Traverse holes and replace coordinates
                for hole in dq_root_copy.findall('.//HoleQualityData/Hole', namespaces):
                    hole_name_node = hole.find('HoleName', namespaces)
                    if hole_name_node is None:
                        continue
                    hole_name = hole_name_node.text.strip()
            
                    if hole_name in dq_mine_dict:
                        data = dq_mine_dict[hole_name]
            
                        start_node = hole.find('StartPoint', namespaces)
                        end_node = hole.find('EndPoint', namespaces)
            
                        if start_node is not None:
                            for tag, key in zip(['IR:PointX', 'IR:PointY', 'IR:PointZ'],
                                                ['CollarX', 'CollarY', 'CollarZ']):
                                point = start_node.find(tag, namespaces)
                                if point is not None:
                                    point.text = f"{data[key]:.6f}"
            
                        if end_node is not None:
                            for tag, key in zip(['IR:PointX', 'IR:PointY', 'IR:PointZ'],
                                                ['ToeX', 'ToeY', 'ToeZ']):
                                point = end_node.find(tag, namespaces)
                                if point is not None:
                                    point.text = f"{data[key]:.6f}"
            
                # ✅ Prepare comment text with timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                comment_text = (
                    f"Generated from the original DQ XML on {timestamp}. "
                    "Drilled hole coordinates aligned to mine grid by NavDrill Quality."
                )
            
                # ✅ Write XML manually so the comment appears right after the declaration
                dq_tree_copy = ET.ElementTree(dq_root_copy)
                with open(dq_output_path, "wb") as f:
                    f.write(b"<?xml version='1.0' encoding='utf-8'?>\n")
                    f.write(ET.tostring(ET.Comment(comment_text), encoding='utf-8'))
                    f.write(b"\n")
                    dq_tree_copy.write(f, encoding='utf-8', xml_declaration=False)
            
                log(f"\n✅ IREDES XML with mine-aligned drilled holes saved → {dq_output_path}")
            except Exception as e:
                log(f"  \n⚠️ Failed to create mine-aligned IREDES XML for {dq_fname}: {e}")
            
        # ------ADD EXTRA COMPUTED METRICS TO METADATA
        try:
            df_designMine = df_design_mine[cols_keep].copy()
            df_dqMine = df_dq_mine[cols_keep].copy()

            df_designMine = df_designMine.rename(columns={"HoleName": "HoleID"})
            df_dqMine = df_dqMine.rename(columns={"HoleName": "HoleID"})

            df_meta = pd.merge(df_dq_metadata, df_designMine, on=["RingID", "HoleID"])
            df_meta = pd.merge(df_meta, df_dqMine, on=["RingID", "HoleID"], suffixes=("_Design", "_Actual"))

            # numeric conversions
            df_meta["DrillSeq"] = pd.to_numeric(df_meta["DrillSeq"], errors='ignore')
            df_meta["AvgPenetr"] = pd.to_numeric(df_meta["AvgPenetr"], errors='ignore')
            df_meta["DrilledInRock"] = pd.to_numeric(df_meta["DrilledInRock"], errors='ignore')

            # Convert AvgPenetr and DrilledInRock
            df_meta["AvgPenetr"] = df_meta["AvgPenetr"] * meters_to_usft
            df_meta["DrilledInRock"] = df_meta["DrilledInRock"] * meters_to_usft

            df_meta["ΔCollar"] = np.sqrt(
                (df_meta["CollarX_Actual"] - df_meta["CollarX_Design"])**2 +
                (df_meta["CollarY_Actual"] - df_meta["CollarY_Design"])**2 +
                (df_meta["CollarZ_Actual"] - df_meta["CollarZ_Design"])**2
            )
            df_meta["ΔToe"] = np.sqrt(
                (df_meta["ToeX_Actual"] - df_meta["ToeX_Design"])**2 +
                (df_meta["ToeY_Actual"] - df_meta["ToeY_Design"])**2 +
                (df_meta["ToeZ_Actual"] - df_meta["ToeZ_Design"])**2
            )
            df_meta["Design_Length"] = np.sqrt(
                (df_meta["ToeX_Design"] - df_meta["CollarX_Design"])**2 +
                (df_meta["ToeY_Design"] - df_meta["CollarY_Design"])**2 +
                (df_meta["ToeZ_Design"] - df_meta["CollarZ_Design"])**2
            )
            df_meta["Actual_Length"] = np.sqrt(
                (df_meta["ToeX_Actual"] - df_meta["CollarX_Actual"])**2 +
                (df_meta["ToeY_Actual"] - df_meta["CollarY_Actual"])**2 +
                (df_meta["ToeZ_Actual"] - df_meta["CollarZ_Actual"])**2
            )
            df_meta["ΔLength"] = df_meta["Actual_Length"] - df_meta["Design_Length"]
            
            # Calculate %DrillInRock and cap at 100%
            df_meta["%DrillInRock"] = (df_meta["DrilledInRock"] / df_meta["Actual_Length"] * 100).clip(upper=100).round(2)
            
            # reorder as before (only if columns exist)
            cols_order = [
                "RingID", "HoleID", "DrillSeq", "HoleStatus", "AvgPenetr", "DrilledInRock", "%DrillInRock",
                "ΔCollar", "ΔToe", "ΔLength", "Design_Length", "Actual_Length", 
                "CollarX_Design", "CollarY_Design", "CollarZ_Design",
                "ToeX_Design", "ToeY_Design", "ToeZ_Design",
                "CollarX_Actual", "CollarY_Actual", "CollarZ_Actual",
                "ToeX_Actual", "ToeY_Actual", "ToeZ_Actual"
            ]
            cols_order = [c for c in cols_order if c in df_meta.columns]
            df_meta = df_meta[cols_order]

            combined_meta_frames.append(df_meta)
            log(f"  \n✅ Drill Quality report generated for {base}_{ring} (rows: {len(df_meta)})")
        except Exception as e:
            log(f"  \n⚠️ Failed to generate Drill Qaulity report for {base}_{ring}: {e}")
            continue
        
    # -----After loop write combined Drill Quality Report ------------
    try:
        if combined_meta_frames:
            df_combined_meta = pd.concat(combined_meta_frames, ignore_index=True)
            # create workbook & write with headers bolded
            wb = Workbook()
            ws = wb.active
            ws.title = "Drill Quality Report"
            bold_font = Font(bold=True)
            for col_idx, col_name in enumerate(df_combined_meta.columns, start=1):
                cell = ws.cell(row=1, column=col_idx, value=col_name)
                cell.font = bold_font
            for row in df_combined_meta.itertuples(index=False, name=None):
                ws.append(row)
            wb.save(metadata_output)
            log(f"\n\n📘 Final Drill Quality Report saved → {metadata_output}")
        else:
            log("\n\nℹ️ No individual ring qaulity reports were accumulated; no Drill Quality Report produced.")
    except Exception as e:
        log(f"\n\n⚠️ Failed saving Drill Quality Report: {e}")

    log("\n\n🏁 Batch processing complete.")    
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    log(f"\n⏱️ Processing time: {int(minutes)} min {seconds:.2f} sec\n\n")
#---------------------------------------------------------------------------------------------------------------------------------

# -----------------------------
# Worker Thread (QThread)
# -----------------------------
class NavDrillWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # done, total
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    request_continue_abort = pyqtSignal()

    def __init__(self, input_folder, mine_file, want_dxf, want_iredes, parent=None):
        super().__init__(parent)
        self.input_folder = input_folder
        self.mine_file = mine_file
        self.want_dxf = want_dxf
        self.want_iredes = want_iredes
        self._is_running = True
        self.user_decision = None  # will be set by GUI
        self.abort_requested = False
        self.no_valid_pairs_exit = False

    def run(self):
        try:
            process_navdrill(
                self.input_folder,
                self.mine_file,
                want_dxf=self.want_dxf,
                want_iredes=self.want_iredes,
                log_callback=self.emit_log,
                runner=self
            )
            self.finished_signal.emit()
        except Exception as e:
            tb = traceback.format_exc()
            self.error_signal.emit(f"{e}\n{tb}")

    def emit_log(self, text):
        # text may contain trailing newline(s)
        self.log_signal.emit(text)

    def emit_progress(self, done, total):
        self.progress_signal.emit(done, total)

    def stop(self):
        self._is_running = False
        try:
            self.terminate()
        except Exception:
            pass

# -----------------------------
# Main GUI Window (as provided + integration)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self, script_path=None):
        super().__init__()
        self.script_path = script_path
        self.setWindowTitle("NavDrill Quality")
        self.setWindowIcon(QIcon(resource_path("app_icon.ico")))
        self.setMinimumSize(600, 400)

        # Widgets
        self.input_folder_edit = QLineEdit()
        self.mine_file_edit = QLineEdit()
        self.browse_input_btn = QPushButton("Browse Folder")
        self.browse_mine_btn = QPushButton("Browse File")
        self.iredes_cb = QCheckBox("IREDES")
        self.dxf_cb = QCheckBox("DXF")
        self.iredes_cb.setChecked(True)
        self.dxf_cb.setChecked(False)
        self.reset_btn = QPushButton("Reset")
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet('font-weight: bold; font-size: 14px')
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.progress = QProgressBar()
        self.progress.setRange(0,0)  # indeterminate until started
        self.progress.setVisible(False)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Input Folder (drag & drop or browse):"))
        left_layout.addWidget(self.input_folder_edit)
        left_layout.addWidget(self.browse_input_btn)
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Design Hole Data (Deswik Export):"))
        left_layout.addWidget(self.mine_file_edit)
        left_layout.addWidget(self.browse_mine_btn)
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Output Format:"))
        left_layout.addWidget(self.iredes_cb)
        left_layout.addWidget(self.dxf_cb)
        left_layout.addStretch(1)
        left_layout.addWidget(self.reset_btn)
        left_layout.addWidget(self.start_btn)
        left_layout.addWidget(self.progress)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Console log:"))
        right_layout.addWidget(self.log_area)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

        # Drag & drop
        self.setAcceptDrops(True)

        # Connections
        self.browse_input_btn.clicked.connect(self.on_browse_folder)
        self.browse_mine_btn.clicked.connect(self.on_browse_file)
        self.start_btn.clicked.connect(self.on_start)
        self.reset_btn.clicked.connect(self.on_reset)

        self.runner = None

    def show_about(self):
        text = (
            "NavDQ processes and converts navigation Drill returns IREDES in the Drill's local coordinate frame"
            " to the Mine's local coordinate frame from the exported Drill Design IREDES, Navigation Drill"
            " Return IREDES, and the Deswik exported Design Hole data.csv,"
            " and generating:<br>"
            "- Drill Quality Reports<br>"
            "- IREDES outputs for deswik import through UGDB as-drilled/as-built hole import wizard<br>"
            "- DXF outputs of as-drilled hole polylines<br><br>"
            "Developed to streamline the workflow of QAQC in Drill Designs vs As-drilled, "
            "with reference to stope scans<br><br>"
            "<b>Developer:</b><br>"
            "Kenneth Kwabena Kenney<br><br>"
            "<b>Contact Information:</b>"
            "<ul>"
            "<li>Work Email: <a href='mailto:kenneth.kenney@nevadagoldmines.com'>kenneth.kenney@nevadagoldmines.com</a></li>"
            "<li>Personal Email: <a href='mailto:kenneykennethkwabena@gmail.com'>kenneykennethkwabena@gmail.com</a></li>"
            "<li>Phone: +1 (458) 272-7638</li>"
            "</ul>"
            "© 2025 NavDrill Quality Application"
        )
        QMessageBox.information(self, "About NavDrill Quality", text)

    def append_log(self, text):
        # maintain cursor at end and append
        if not text:
            return
        self.log_area.moveCursor(QTextCursor.End)
        self.log_area.insertPlainText(text)
        self.log_area.ensureCursorVisible()

    def on_browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder containing IREDES XML file pairs")
        if d:
            self.input_folder_edit.setText(d)

    def on_browse_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Design Hole Data file", filter="Excel/CSV (*.xlsx *.xls *.csv);;All files (*)")
        if f:
            self.mine_file_edit.setText(f)

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        urls = md.urls()
        if not urls:
            return
        files = [u.toLocalFile() for u in urls]
        folder_candidates = [f for f in files if os.path.isdir(f)]
        file_candidates = [f for f in files if os.path.isfile(f)]
        if folder_candidates:
            self.input_folder_edit.setText(folder_candidates[0])
        for f in file_candidates:
            if f.lower().endswith(('.csv', '.xlsx', '.xls')):
                self.mine_file_edit.setText(f)
                break

    def on_reset(self):
        self.input_folder_edit.clear()
        self.mine_file_edit.clear()
        self.iredes_cb.setChecked(True)
        self.dxf_cb.setChecked(False)
        self.log_area.clear()
        self.progress.setVisible(False)
        self.progress.setRange(0,0)

    def on_start(self):
        input_folder = self.input_folder_edit.text().strip()
        mine_file = self.mine_file_edit.text().strip()
        want_iredes = self.iredes_cb.isChecked()
        want_dxf = self.dxf_cb.isChecked()
        
        if not (self.iredes_cb.isChecked() or self.dxf_cb.isChecked()):
            QMessageBox.warning(self, "No output format selected", "Please select output format, IREDES/DXF/Both.")
            return
        
        if not input_folder or not os.path.isdir(input_folder):
            QMessageBox.warning(self, "Input folder missing", "Please select a valid input folder containing IREDES XML files.")
            return
        if not mine_file or not os.path.isfile(mine_file):
            QMessageBox.warning(self, "Design Hole data missing", "Please select the deswik exported design hole data file (CSV or Excel).")
            return

        # disable UI while processing
        self.start_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.browse_input_btn.setEnabled(False)
        self.browse_mine_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0,0)  # indeterminate until worker reports total

        # start worker
        self.runner = NavDrillWorker(input_folder, mine_file, want_dxf, want_iredes)
        self.runner.request_continue_abort.connect(self.on_unmatched_pairs_decision)
        self.runner.log_signal.connect(self.append_log)
        self.runner.progress_signal.connect(self.on_progress_update)
        self.runner.finished_signal.connect(self.on_finished)
        self.runner.error_signal.connect(self.on_error)
        self.runner.start()
        self.append_log("🟢 Processing started...\n\n")


    def on_progress_update(self, done, total):
        # first time we see total, set range
        if self.progress.maximum() == 0 and total > 0:
            self.progress.setRange(0, total)
        self.progress.setValue(done)

    def on_finished(self):
        self.progress.setValue(self.progress.maximum())
        self.progress.setVisible(False)
        if self.runner.abort_requested:
            self.start_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.browse_input_btn.setEnabled(True)
            self.browse_mine_btn.setEnabled(True)
            return 
        if self.runner.no_valid_pairs_exit:
            self.start_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.browse_input_btn.setEnabled(True)
            self.browse_mine_btn.setEnabled(True)
            return
        self.start_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.browse_input_btn.setEnabled(True)
        self.browse_mine_btn.setEnabled(True)
        QMessageBox.information(self, "Finished", "All valid ring pairs processing complete.")

    def on_error(self, text):
        self.append_log(f"\n❌ Error during processing:\n{text}\n")
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.browse_input_btn.setEnabled(True)
        self.browse_mine_btn.setEnabled(True)
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{text[:1000]}")
    
    def on_unmatched_pairs_decision(self):
        reply = QMessageBox.question(
            self,
            "Unmatched Ring Pairs Detected",
            "Unmatched ring pairs were found. See Consol log. \n\n"
            "Do you want to continue processing valid ring pairs?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.runner.user_decision = True
        else:
            self.runner.user_decision = False
            self.runner.abort_requested = True  

# -----------------------------
# Splash screen painter (your function)
# -----------------------------
def show_splash(app):
    pixmap = QPixmap(resource_path("splash_screen.png"))

    splash = QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    time.sleep(1.0)
    return splash

# -----------------------------
# App entrypoint
# -----------------------------
if __name__ == "__main__":
    # Enable High-DPI scaling before QApplication is created
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    splash = show_splash(app)
    window = MainWindow(script_path=__file__)
    
    screen = app.primaryScreen()
    available = screen.availableGeometry()
    
    default_w, default_h = 800, 500
    
    # Fit to screen if necessary
    w = min(default_w, available.width())
    h = min(default_h, available.height())
    
    # Respect minimum
    w = max(w, window.minimumWidth())
    h = max(h, window.minimumHeight())
    
    window.resize(w, h)

    splash.finish(window)
    window.show()
    sys.exit(app.exec_())