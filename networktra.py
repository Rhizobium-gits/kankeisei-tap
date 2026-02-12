# ================================================================
# TouchDesigner 2025.31760
# Motion Network — 重要度で選別
# ================================================================
#
# 方針: 制限をかけるのではなく、検出基準を厳しくする
#   - MIN_AREA を大きく → 大きい動体だけ
#   - MIN_TRAVEL を大きく → しっかり動いてるものだけ
#   - キーポイントを少なく → 重心 + 最も離れた2点のみ
#   - エッジに上限なし → 自然に繋がる
#
# 使い方:
#   1. File → New → Textport にペースト
#   2. FINAL_OUT → a で拡大
# ================================================================

import os

home = os.path.expanduser('~')
VIDEO = None
found = []

for d in [home, home+'/Desktop', home+'/Downloads',
          home+'/Movies', home+'/Documents']:
    if os.path.isdir(d):
        try:
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp) and f.lower().endswith(
                        ('.mp4','.mov','.avi','.m4v','.mkv','.webm')):
                    found.append(fp)
                    if f == '21.mp4' and VIDEO is None:
                        VIDEO = fp
        except:
            pass

if VIDEO is None and found:
    VIDEO = found[0]

if VIDEO is None:
    VIDEO = home + '/Desktop/video.mp4'

print('')
print('=' * 60)
print('  Motion Network (High Importance)')
print('=' * 60)
print('  Video:', VIDEO)
print('')

p = op('/project1')

for c in p.children:
    c.destroy()

movie1 = p.create(moviefileinTOP, 'movie1')
movie1.par.file = VIDEO
movie1.nodeX = -800
movie1.nodeY = 400

res_main = p.create(resolutionTOP, 'res_main')
res_main.par.resolutionw = 1280
res_main.par.resolutionh = 720
res_main.inputConnectors[0].connect(movie1.outputConnectors[0])
res_main.nodeX = -600
res_main.nodeY = 400

level_bg = p.create(levelTOP, 'level_bg')
level_bg.par.opacity = 0.4
level_bg.inputConnectors[0].connect(res_main.outputConnectors[0])
level_bg.nodeX = -400
level_bg.nodeY = 400

null_bg = p.create(nullTOP, 'null_bg')
null_bg.inputConnectors[0].connect(level_bg.outputConnectors[0])
null_bg.nodeX = -200
null_bg.nodeY = 400

mono1 = p.create(monochromeTOP, 'mono1')
mono1.inputConnectors[0].connect(movie1.outputConnectors[0])
mono1.nodeX = -600
mono1.nodeY = 200

res_detect = p.create(resolutionTOP, 'res_detect')
res_detect.par.resolutionw = 320
res_detect.par.resolutionh = 180
res_detect.inputConnectors[0].connect(mono1.outputConnectors[0])
res_detect.nodeX = -400
res_detect.nodeY = 200

null_detect = p.create(nullTOP, 'null_detect')
null_detect.inputConnectors[0].connect(res_detect.outputConnectors[0])
null_detect.nodeX = -200
null_detect.nodeY = 200

print('  [1] Video pipeline OK')

script_draw = p.create(scriptTOP, 'script_draw')
script_draw.par.resolutionw = 1280
script_draw.par.resolutionh = 720
script_draw.nodeX = 200
script_draw.nodeY = 400

cb = script_draw.par.callbacks.eval()
cb.text = '''import numpy as np
import cv2
import math

# ===== 検出（厳しめ）=====
MOTION_THRESH = 20
MIN_AREA = 800             # ★ 大きい動体だけ（前は300）
DILATE_N = 6               # 膨張多めで近い塊を統合
BLUR_K = 13

# ===== 静止物フィルタ（厳しめ）=====
HISTORY_LEN = 20
MIN_TRAVEL = 40.0          # ★ しっかり動いてるものだけ（前は25）
CONTOUR_MATCH_R = 80

# ===== ノード =====
NODE_MATCH_R = 70
MAX_AGE = 3

# ===== エッジ（制限なし）=====
EDGE_MAX_PX = 280
BEZIER_STEPS = 14
DIST_CHANGE_THRESH = 3.0

# ===== 色 (BGR) =====
C_APPROACHING = (50, 30, 210)
C_SEPARATING = (210, 140, 60)
C_STABLE = (190, 185, 180)

C_NODE_HIGH = (80, 50, 230)
C_NODE_MED = (240, 230, 180)
C_NODE_LOW = (230, 230, 230)
C_GLOW_RED = (60, 35, 170)
C_GLOW_WHITE = (160, 155, 150)
C_LABEL = (160, 160, 160)

_prev = None
_frame = [0]
_contour_tracks = {}
_next_ct_id = [1]
_tracked = {}
_next_nid = [1]
_prev_dists = {}


def _travel(history):
    if len(history) < 2:
        return 0.0
    t = 0.0
    for i in range(1, len(history)):
        dx = history[i][0] - history[i-1][0]
        dy = history[i][1] - history[i-1][1]
        t += math.sqrt(dx*dx + dy*dy)
    return t


def _key_points(cnt, sx, sy, w, h):
    """重心 + 輪郭上で重心から最も遠い2点だけ"""
    pts_out = []

    M = cv2.moments(cnt)
    if M['m00'] <= 0:
        return pts_out

    raw_cx = M['m10'] / M['m00']
    raw_cy = M['m01'] / M['m00']
    cx = int(raw_cx * sx)
    cy = int(raw_cy * sy)
    cx = max(5, min(w - 5, cx))
    cy = max(5, min(h - 5, cy))
    pts_out.append((cx, cy, 'center'))

    pts = cnt.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return pts_out

    # 重心からの距離を計算し、最も遠い点を見つける
    dists = np.sqrt((pts[:, 0] - raw_cx)**2 + (pts[:, 1] - raw_cy)**2)

    # 最遠点1
    i1 = np.argmax(dists)
    px1 = int(pts[i1][0] * sx)
    py1 = int(pts[i1][1] * sy)
    px1 = max(5, min(w - 5, px1))
    py1 = max(5, min(h - 5, py1))
    pts_out.append((px1, py1, 'ext1'))

    # 最遠点1からも遠い点を探す（反対側）
    d_from_1 = np.sqrt((pts[:, 0] - pts[i1][0])**2 + (pts[:, 1] - pts[i1][1])**2)
    # 重心からの距離 + 点1からの距離を合算して最大
    combined = dists + d_from_1 * 0.5
    combined[i1] = 0  # 点1自身は除外
    i2 = np.argmax(combined)
    px2 = int(pts[i2][0] * sx)
    py2 = int(pts[i2][1] * sy)
    px2 = max(5, min(w - 5, px2))
    py2 = max(5, min(h - 5, py2))
    pts_out.append((px2, py2, 'ext2'))

    return pts_out


def _bezier(x1, y1, x2, y2, tv):
    dx = x2 - x1
    dy = y2 - y1
    ca = 0.3 * math.sin(tv * 0.08 + (x1 + y2) * 0.01)
    cpx = (x1 + x2) * 0.5 - dy * ca
    cpy = (y1 + y2) * 0.5 + dx * ca
    out = []
    for i in range(BEZIER_STEPS + 1):
        s = i / BEZIER_STEPS
        inv = 1.0 - s
        bx = inv*inv*x1 + 2*inv*s*cpx + s*s*x2
        by = inv*inv*y1 + 2*inv*s*cpy + s*s*y2
        out.append([int(bx), int(by)])
    return np.array(out, np.int32)


def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    global _prev
    _frame[0] += 1

    bg_op = op('null_bg')
    det_op = op('null_detect')
    if bg_op is None or det_op is None:
        return

    try:
        bg_arr = bg_op.numpyArray()
        det_arr = det_op.numpyArray()
    except:
        return
    if bg_arr is None or det_arr is None:
        return

    out_h, out_w = bg_arr.shape[:2]
    det_h, det_w = det_arr.shape[:2]
    if out_h < 10 or det_h < 10:
        return

    canvas = (bg_arr[:, :, :3] * 255).astype(np.uint8)
    canvas = canvas[:, :, ::-1].copy()
    gray = (det_arr[:, :, 0] * 255).astype(np.uint8)

    if _prev is None:
        _prev = gray.copy()
        out = canvas[:, :, ::-1].copy()
        rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
        rgba[:, :, :3] = out.astype(np.float32) / 255.0
        scriptOp.copyNumpyArray(rgba)
        return

    diff = cv2.absdiff(gray, _prev)
    _prev = gray.copy()

    blur = cv2.GaussianBlur(diff, (BLUR_K, BLUR_K), 0)
    _, mask = cv2.threshold(blur, MOTION_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_N)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sx = out_w / det_w
    sy = out_h / det_h
    t = absTime.seconds

    # === 輪郭追跡 ===
    fc = []
    fc_cnt = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            fc.append((M['m10']/M['m00'], M['m01']/M['m00']))
            fc_cnt.append(cnt)

    matched_ct = set()
    used_fc = set()

    for cid, ct in list(_contour_tracks.items()):
        best_d = CONTOUR_MATCH_R
        best_j = -1
        for j, (fcx, fcy) in enumerate(fc):
            if j in used_fc:
                continue
            d = math.sqrt((ct['cx']-fcx)**2 + (ct['cy']-fcy)**2)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            fx, fy = fc[best_j]
            ct['cx'] = fx
            ct['cy'] = fy
            ct['history'].append((fx, fy))
            if len(ct['history']) > HISTORY_LEN:
                ct['history'] = ct['history'][-HISTORY_LEN:]
            ct['miss'] = 0
            ct['cnt'] = fc_cnt[best_j]
            matched_ct.add(cid)
            used_fc.add(best_j)

    for j in range(len(fc)):
        if j not in used_fc:
            cid = _next_ct_id[0]
            _next_ct_id[0] += 1
            _contour_tracks[cid] = {
                'cx': fc[j][0], 'cy': fc[j][1],
                'history': [fc[j]], 'miss': 0,
                'cnt': fc_cnt[j]
            }

    dead_ct = []
    for cid in _contour_tracks:
        if cid not in matched_ct:
            _contour_tracks[cid]['miss'] += 1
            if _contour_tracks[cid]['miss'] > 5:
                dead_ct.append(cid)
    for cid in dead_ct:
        del _contour_tracks[cid]

    # === 動体フィルタ ===
    moving = []
    static_n = 0
    for cid, ct in _contour_tracks.items():
        if ct.get('miss', 0) > 0:
            continue
        if _travel(ct['history']) >= MIN_TRAVEL:
            moving.append(ct['cnt'])
        else:
            static_n += 1

    # === ノード抽出（1動体あたり3点のみ）===
    detected = []
    for cnt in moving:
        for (px, py, role) in _key_points(cnt, sx, sy, out_w, out_h):
            detected.append((px, py, role))

    # === ノード マッチング ===
    matched_nids = set()
    used_det = set()

    for nid, nd in list(_tracked.items()):
        best_d = NODE_MATCH_R
        best_j = -1
        for j, (dx, dy, role) in enumerate(detected):
            if j in used_det:
                continue
            dist = math.sqrt((nd['x']-dx)**2 + (nd['y']-dy)**2)
            if dist < best_d:
                best_d = dist
                best_j = j
        if best_j >= 0:
            dx, dy, role = detected[best_j]
            nd['vx'] = dx - nd['x']
            nd['vy'] = dy - nd['y']
            nd['x'] = dx
            nd['y'] = dy
            nd['age'] = 0
            matched_nids.add(nid)
            used_det.add(best_j)

    for j, (dx, dy, role) in enumerate(detected):
        if j not in used_det:
            nid = _next_nid[0]
            _next_nid[0] += 1
            _tracked[nid] = {
                'x': dx, 'y': dy,
                'vx': 0, 'vy': 0, 'age': 0
            }

    dead_n = []
    for nid in _tracked:
        if nid not in matched_nids:
            _tracked[nid]['age'] += 1
            if _tracked[nid]['age'] > MAX_AGE:
                dead_n.append(nid)
    for nid in dead_n:
        del _tracked[nid]
        for k in [k for k in _prev_dists if nid in k]:
            del _prev_dists[k]

    # === 重要度 ===
    importance = {}
    for nid, nd in _tracked.items():
        importance[nid] = math.sqrt(nd['vx']**2 + nd['vy']**2)

    # === エッジ（制限なし）===
    ids = list(_tracked.keys())
    n = len(ids)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            na, nb = _tracked[a], _tracked[b]
            dx = nb['x'] - na['x']
            dy = nb['y'] - na['y']
            d = math.sqrt(dx*dx + dy*dy)
            if d > EDGE_MAX_PX or d < 10:
                continue

            pair = (min(a, b), max(a, b))
            prev_d = _prev_dists.get(pair, d)
            delta = d - prev_d
            _prev_dists[pair] = d

            if delta < -DIST_CHANGE_THRESH:
                col = C_APPROACHING
            elif delta > DIST_CHANGE_THRESH:
                col = C_SEPARATING
            else:
                col = C_STABLE
            edges.append((a, b, col))

    # 接続数 → 重要度
    for (a, b, col) in edges:
        importance[a] = importance.get(a, 0) + 2.0
        importance[b] = importance.get(b, 0) + 2.0

    # === 描画: エッジ ===
    for (a, b, col) in edges:
        na, nb = _tracked[a], _tracked[b]
        pts = _bezier(na['x'], na['y'], nb['x'], nb['y'], t)
        cv2.polylines(canvas, [pts], False, col, 1, cv2.LINE_AA)

    # === 描画: ノード ===
    iv = list(importance.values())
    im = max(iv) if iv and max(iv) > 0 else 1

    for nid, nd in _tracked.items():
        px, py = int(nd['x']), int(nd['y'])
        norm = importance.get(nid, 0) / im

        if norm > 0.6:
            r = 4
            cv2.circle(canvas, (px, py), r, C_NODE_HIGH, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), r + 4, C_GLOW_RED, 1, cv2.LINE_AA)
        elif norm > 0.3:
            r = 3
            cv2.circle(canvas, (px, py), r, C_NODE_MED, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), r + 3, C_GLOW_WHITE, 1, cv2.LINE_AA)
        else:
            r = 2
            cv2.circle(canvas, (px, py), r, C_NODE_LOW, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), r + 3, C_GLOW_WHITE, 1, cv2.LINE_AA)

    # === 描画: ラベル ===
    for nid, nd in _tracked.items():
        norm = importance.get(nid, 0) / im
        if norm > 0.2:
            px, py = int(nd['x']), int(nd['y'])
            lbl = "codecore " + str(nid)
            lx = px + 7
            ly = py - 5
            if lx > out_w - 90:
                lx = px - 85
            if ly < 14:
                ly = py + 14
            cv2.putText(canvas, lbl, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        C_LABEL, 1, cv2.LINE_AA)

    if _frame[0] % 60 == 0:
        print('  [f', _frame[0], ']',
              'moving:', len(moving),
              'static:', static_n,
              'nodes:', len(_tracked),
              'edges:', len(edges))

    out_rgb = canvas[:, :, ::-1].copy()
    rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
    rgba[:, :, :3] = out_rgb.astype(np.float32) / 255.0
    scriptOp.copyNumpyArray(rgba)
'''

print('  [2] script_draw OK')

bloom1 = p.create(bloomTOP, 'bloom1')
bloom1.par.maxbloomradius = 10
bloom1.par.bloomintensity = 1.3
bloom1.par.bloomthreshold = 0.3
bloom1.inputConnectors[0].connect(script_draw.outputConnectors[0])
bloom1.nodeX = 400
bloom1.nodeY = 400

final = p.create(nullTOP, 'FINAL_OUT')
final.inputConnectors[0].connect(bloom1.outputConnectors[0])
final.nodeX = 600
final.nodeY = 400
final.viewer = True

movie_out = p.create(moviefileoutTOP, 'movie_out')
movie_out.par.file = home + '/Desktop/output_network.mp4'
movie_out.inputConnectors[0].connect(final.outputConnectors[0])
movie_out.nodeX = 600
movie_out.nodeY = 200

all_n = [
    'movie1','res_main','level_bg','null_bg',
    'mono1','res_detect','null_detect',
    'script_draw','bloom1','FINAL_OUT','movie_out'
]

ok = 0
for name in all_n:
    nd = op('/project1/' + name)
    if nd is None:
        print('  [MISSING] ' + name)
    else:
        e = nd.errors()
        if e and 'Non-Commercial' not in e:
            print('  [ERROR] ' + name + ': ' + e)
        else:
            ok += 1

print('  ' + str(ok) + '/' + str(len(all_n)) + ' nodes OK')

print('')
print('=' * 60)
print('  SETUP COMPLETE!')
print('=' * 60)
print('')
print('  FINAL_OUT → a')
print('')
print('  ■ 検出基準:')
print('    MIN_AREA=800 (大きい動体だけ)')
print('    MIN_TRAVEL=40 (しっかり動くものだけ)')
print('    1動体あたり3点のみ (重心+端点2つ)')
print('    → 人間1人 = 3ノード')
print('')
print('  ■ エッジ: 制限なし、自然に接続')
print('  ■ 色: 赤=近づく 青=離れる 白=安定')
print('')
print('  ■ 録画:')
print("    op('/project1/movie_out').par.record = True")
print("    op('/project1/movie_out').par.record = False")
print('')
print('=' * 60)
