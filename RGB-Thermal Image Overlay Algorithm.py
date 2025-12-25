import cv2
import numpy as np
import os
import sys
import re


INPUT_DIR = "input-images"
OUTPUT_DIR = "output-images"

WIDE_THERMAL_FILES = [
    "DJI_20250530121724_0004_T.JPG",
    "DJI_20250530122012_0010_T.JPG"
]

def parse_filename(filename):
    match = re.search(r'DJI_(\d{14})_(\d{4})_([TZ])', filename, re.IGNORECASE)
    if match:
        timestamp = int(match.group(1))
        sequence = match.group(2)
        type_char = match.group(3).upper()
        return timestamp, sequence, type_char
    return None, None, None

def find_image_pairs_smart(directory):
    all_files = sorted(os.listdir(directory))
    thermals = []
    rgbs = []
    
    for f in all_files:
        if f.startswith('.'): continue
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        ts, seq, ftype = parse_filename(f)
        if ts is None: continue
        entry = {'file': f, 'ts': ts, 'seq': seq}
        if ftype == 'T': thermals.append(entry)
        elif ftype == 'Z': rgbs.append(entry)
            
    pairs = {}
    for t in thermals:
        best_match = None
        min_time_diff = 1000000 
        for z in rgbs:
            if t['seq'] == z['seq']:
                diff = abs(t['ts'] - z['ts'])
                if diff < 10: 
                    if diff < min_time_diff:
                        min_time_diff = diff
                        best_match = z
        if best_match:
            pairs[t['file']] = {'rgb': best_match['file'], 'thermal': t['file']}
    return pairs

def get_structural_edges(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    return np.uint8(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))

def align_smart_zoom(rgb_img, thermal_img, thermal_filename):
    h_z, w_z = rgb_img.shape[:2]
    h_t, w_t = thermal_img.shape[:2]

    WORK_W = 800
    scale_factor = WORK_W / float(w_z)
    work_h = int(h_z * scale_factor)

    rgb_small = cv2.resize(rgb_img, (WORK_W, work_h))
    map_z = get_structural_edges(rgb_small)

    if thermal_filename in WIDE_THERMAL_FILES:
        print(" -> [Wide Thermal Mode] Searching 1.1x - 1.6x Zoom...", end=" ")
        scales = np.linspace(1.1, 1.6, 40)
    else:
        scales = np.linspace(0.8, 1.2, 40)

    best_score = -1
    best_scale_rel = 1.0
    best_loc = (0, 0)

    base_scale_t = work_h / float(h_t)

    for s in scales:
        curr_scale = base_scale_t * s
        curr_w = int(w_t * curr_scale)
        curr_h = int(h_t * curr_scale)

        t_small = cv2.resize(thermal_img, (curr_w, curr_h))
        map_t = get_structural_edges(t_small)
        crop_h = int(curr_h * 0.6)
        crop_w = int(curr_w * 0.6)
        cy, cx = curr_h // 2, curr_w // 2
        ty1, tx1 = cy - crop_h // 2, cx - crop_w // 2
        template = map_t[ty1:ty1+crop_h, tx1:tx1+crop_w]

        if template.shape[0] >= map_z.shape[0] or template.shape[1] >= map_z.shape[1]: 
            continue

        res = cv2.matchTemplate(map_z, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_scale_rel = curr_scale
            best_loc = (max_loc[0] - tx1, max_loc[1] - ty1)

    final_scale = best_scale_rel / scale_factor
    final_w = int(w_t * final_scale)
    final_h = int(h_t * final_scale)
    
    final_x = int(best_loc[0] / scale_factor)
    final_y = int(best_loc[1] / scale_factor)

    thermal_final = cv2.resize(thermal_img, (final_w, final_h), interpolation=cv2.INTER_CUBIC)
    
    aligned_canvas = np.zeros((h_z, w_z, 3), dtype=np.uint8)
    
    c_x1 = max(0, final_x)
    c_y1 = max(0, final_y)
    c_x2 = min(w_z, final_x + final_w)
    c_y2 = min(h_z, final_y + final_h)
    
    t_x1 = max(0, -final_x)
    t_y1 = max(0, -final_y)
    t_x2 = t_x1 + (c_x2 - c_x1)
    t_y2 = t_y1 + (c_y2 - c_y1)
    
    if (c_x2 > c_x1) and (c_y2 > c_y1):
        aligned_canvas[c_y1:c_y2, c_x1:c_x2] = thermal_final[t_y1:t_y2, t_x1:t_x2]
    else:
        print(" -> Fallback", end=" ")
        x = (w_z - final_w) // 2
        y = (h_z - final_h) // 2
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_z, x + final_w), min(h_z, y + final_h)
        tx1, ty1 = max(0, -x), max(0, -y)
        tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)
        aligned_canvas[y1:y2, x1:x2] = thermal_final[ty1:ty2, tx1:tx2]

    return aligned_canvas

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pairs = find_image_pairs_smart(INPUT_DIR)
    print(f"Found {len(pairs)} pairs. Starting Alignment...")

    count = 0
    for t_key in sorted(pairs.keys()):
        count += 1
        filenames = pairs[t_key]
        base_name = os.path.basename(t_key)
        print(f"[{count}/{len(pairs)}] {base_name}...", end=" ")
        
        rgb_path = os.path.join(INPUT_DIR, filenames['rgb'])
        therm_path = os.path.join(INPUT_DIR, filenames['thermal'])
        
        img_rgb = cv2.imread(rgb_path)
        img_therm = cv2.imread(therm_path)
        
        if img_rgb is None or img_therm is None:
            print("Error reading.")
            continue

        try:
            aligned_img = align_smart_zoom(img_rgb, img_therm, base_name)
            
            out_rgb = os.path.join(OUTPUT_DIR, filenames['rgb'])
            out_therm = os.path.join(OUTPUT_DIR, filenames['thermal'].replace(".JPG", "_ALIGNED.JPG").replace(".jpg", "_ALIGNED.jpg"))
            
            cv2.imwrite(out_rgb, img_rgb)
            cv2.imwrite(out_therm, aligned_img)
            print("Done.")
            
        except Exception as e:
            print(f"Failed ({e})")

    print(f"\nBatch processing complete. Total pairs processed: {count}")

if __name__ == "__main__":
    main()