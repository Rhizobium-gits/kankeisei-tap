# ================================================================
# TouchDesigner 2025.31760
# Human Motion Network — FINAL
# ================================================================
#
# 使い方:
#   1. File → New
#   2. Dialogs → Textport
#   3. このスクリプト全体をペースト
#   4. FINAL_OUT クリック → a で拡大
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
print('  Human Motion Network — FINAL')
print('=' * 60)
print('  Video:', VIDEO)
print('')

p = op('/project1')

for c in p.children:
    c.destroy()

# ================================================================
# 動画入力
# ================================================================

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

print('  [OK] Video pipeline')

# ================================================================
# Script TOP: 検出 + 描画
# ================================================================

script_draw = p.create(scriptTOP, 'script_draw')
script_draw.par.resolutionw = 1280
script_draw.par.resolutionh = 720
script_draw.nodeX = 200
script_draw.nodeY = 400

cb = script_draw.par.callbacks.eval()
cb.text = '''import numpy as np
import cv2
import math

# ===== パラメータ =====
MOTION_THRESH = 18
MIN_AREA = 600
SAMPLE_STEP = 25
DILATE_N = 5
BLUR_K = 11
EDGE_MAX_PX = 80
NODE_R = 3
PINK_R = 5

# 色 (BGR)
C_EDGE = (45, 25, 210)
C_NODE = (255, 240, 190)
C_PINK = (85, 45, 240)
C_GLOW = (180, 160, 140)
C_LABEL = (170, 170, 170)

_prev = None
_frame = [0]

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
    except Exception as ex:
        if _frame[0] % 60 == 1:
            print('  [err]', ex)
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
        out_f = out.astype(np.float32) / 255.0
        rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
        rgba[:, :, :3] = out_f
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
    nodes = []
    nid = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 15:
            continue

        pts = cnt.reshape(-1, 2)
        n_pts = len(pts)
        n_nodes = max(3, int(peri / SAMPLE_STEP))
        n_nodes = min(n_nodes, 40)

        for i in range(n_nodes):
            frac = i / n_nodes
            idx = int(frac * n_pts) % n_pts
            px = int(pts[idx][0] * sx)
            py = int(pts[idx][1] * sy)
            px = max(10, min(out_w - 10, px))
            py = max(10, min(out_h - 10, py))
            nid += 1
            nodes.append((px, py, nid))

    if _frame[0] % 60 == 0:
        big = sum(1 for c in contours if cv2.contourArea(c) >= MIN_AREA)
        print('  [f', _frame[0], '] large:', big, ' nodes:', len(nodes))

    # エッジ
    n = len(nodes)
    for i in range(n):
        x1, y1, id1 = nodes[i]
        for j in range(i + 1, n):
            x2, y2, id2 = nodes[j]
            dx = x2 - x1
            dy = y2 - y1
            d = math.sqrt(dx*dx + dy*dy)

            if d < EDGE_MAX_PX and d > 12:
                mx = (x1 + x2) // 2
                my = (y1 + y2) // 2
                crv = 0.1 * math.sin(t * 0.12 + (id1 + id2) * 0.3)
                cx = int(mx - dy * crv)
                cy = int(my + dx * crv)

                line_pts = np.array([[x1,y1],[cx,cy],[x2,y2]], np.int32)
                cv2.polylines(canvas, [line_pts], False, C_EDGE, 1, cv2.LINE_AA)

    # ノード + ラベル
    for (px, py, nid_val) in nodes:
        if nid_val % 5 == 0:
            cv2.circle(canvas, (px, py), PINK_R, C_PINK, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), PINK_R + 2, C_PINK, 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px, py), NODE_R, C_NODE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), NODE_R + 2, C_GLOW, 1, cv2.LINE_AA)

        if nid_val % 4 == 0:
            lbl = "codecore " + str(nid_val)
            lx = px + 7
            ly = py - 4
            if lx > out_w - 90:
                lx = px - 85
            if ly < 12:
                ly = py + 12
            cv2.putText(canvas, lbl, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                        C_LABEL, 1, cv2.LINE_AA)

    # 出力
    out_rgb = canvas[:, :, ::-1].copy()
    out_f = out_rgb.astype(np.float32) / 255.0
    rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
    rgba[:, :, :3] = out_f
    scriptOp.copyNumpyArray(rgba)
'''

print('  [OK] script_draw')

# ================================================================
# ブルーム
# ================================================================

bloom1 = p.create(bloomTOP, 'bloom1')
bloom1.par.maxbloomradius = 6
bloom1.par.bloomintensity = 1.2
bloom1.par.bloomthreshold = 0.4
bloom1.inputConnectors[0].connect(script_draw.outputConnectors[0])
bloom1.nodeX = 400
bloom1.nodeY = 400

print('  [OK] bloom1')

# ================================================================
# 出力
# ================================================================

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

print('  [OK] Output')

# ================================================================
# 検証
# ================================================================

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
print('  FINAL_OUT クリック → a で拡大')
print('')
print('  ■ 録画:')
print('    op("/project1/movie_out").par.record = True')
print('    op("/project1/movie_out").par.record = False')
print('    → ~/Desktop/output_network.mp4')
print('')
print('  ■ 調整 (script_draw callback):')
print('    MOTION_THRESH=18  小→敏感')
print('    MIN_AREA=600  小→小さい動体も')
print('    SAMPLE_STEP=25  小→ノード密')
print('    EDGE_MAX_PX=80  大→線多い')
print('    NODE_R=3  大→ノード大きく')
print('=' * 60)
