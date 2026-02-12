# ================================================================
# TouchDesigner 2025.31760
# MATRIX MOTION — FINAL
# ================================================================
#
# 動体を多言語マトリックス文字で表現
# 背景は真っ黒、動いている人間の形だけが
# 緑の文字の雨で象られる
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
print('  MATRIX MOTION — FINAL')
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
# Script TOP
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
import random

# ===== パラメータ =====
MOTION_THRESH = 15
MIN_AREA = 300
DILATE_N = 6
BLUR_K = 11
CELL_W = 14
CELL_H = 16
TRAIL_LEN = 12
CHAR_CHANGE_RATE = 0.25
MOTION_FADE = 0.85

# 多言語文字セット
CHARS = list(
    '0123456789'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '@#$%&*+=<>{}[]|~^'
    'アイウエオカキクケコサシスセソタチツテト'
    'ナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン'
    'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθ'
    'АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ'
    'กขคงจฉชซญฎฏฐดตถทธนบปพฟภมยรลว'
    '가나다라마바사아자차카타파하'
    '∑∏∫∂√∞≈≠≤≥±×÷∇∆∈∉∪∩⊂⊃'
    '你我他她它的是不了在有这中大来上个'
    '◈◇◆□■△▽○●★☆♠♣♥♦'
)

_prev = None
_frame = [0]
_columns = None
_motion_acc = None
_char_grid = None

def _init_cols(nc, nr):
    cols = []
    for c in range(nc):
        cols.append({
            'head': random.uniform(-nr, nr),
            'speed': random.uniform(0.3, 1.2),
            'active': False
        })
    return cols

def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    global _prev, _columns, _motion_acc, _char_grid
    _frame[0] += 1

    det_op = op('null_detect')
    if det_op is None:
        return

    try:
        det_arr = det_op.numpyArray()
    except:
        return

    if det_arr is None:
        return

    det_h, det_w = det_arr.shape[:2]
    if det_h < 10 or det_w < 10:
        return

    out_w = 1280
    out_h = 720
    cols = out_w // CELL_W
    rows = out_h // CELL_H

    if _columns is None or len(_columns) != cols:
        _columns = _init_cols(cols, rows)
    if _motion_acc is None or _motion_acc.shape != (rows, cols):
        _motion_acc = np.zeros((rows, cols), dtype=np.float32)
    if _char_grid is None or _char_grid.shape != (rows, cols):
        _char_grid = np.empty((rows, cols), dtype=object)
        for r in range(rows):
            for c in range(cols):
                _char_grid[r, c] = random.choice(CHARS)

    # 真っ黒キャンバス
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    gray = (det_arr[:, :, 0] * 255).astype(np.uint8)

    if _prev is None:
        _prev = gray.copy()
        rgba = np.zeros((out_h, out_w, 4), dtype=np.float32)
        rgba[:, :, 3] = 1.0
        scriptOp.copyNumpyArray(rgba)
        return

    diff = cv2.absdiff(gray, _prev)
    _prev = gray.copy()

    blur = cv2.GaussianBlur(diff, (BLUR_K, BLUR_K), 0)
    _, mask = cv2.threshold(blur, MOTION_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_N)

    mask_grid = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)
    mask_norm = mask_grid.astype(np.float32) / 255.0

    _motion_acc = _motion_acc * MOTION_FADE
    _motion_acc = np.maximum(_motion_acc, mask_norm)

    for c in range(cols):
        col = _columns[c]
        col_motion = float(np.max(_motion_acc[:, c]))
        col['active'] = col_motion > 0.15

        if col['active']:
            col['speed'] = 0.4 + col_motion * 1.0
            col['head'] += col['speed']
            if col['head'] - TRAIL_LEN > rows:
                col['head'] = random.uniform(-TRAIL_LEN, -2)
                col['speed'] = random.uniform(0.3, 1.2)
        else:
            if -TRAIL_LEN < col['head'] < rows + TRAIL_LEN:
                col['head'] += col['speed'] * 0.3
            else:
                col['head'] = random.uniform(-rows, -2)

        for r in range(rows):
            if random.random() < CHAR_CHANGE_RATE:
                _char_grid[r, c] = random.choice(CHARS)

    for c in range(cols):
        col = _columns[c]
        head = col['head']

        for r in range(rows):
            motion_val = _motion_acc[r, c]
            if motion_val < 0.05:
                continue

            dist = head - r
            if dist < 0 or dist > TRAIL_LEN:
                if motion_val > 0.3:
                    brightness = motion_val * 0.2
                else:
                    continue
            else:
                fade = 1.0 - (dist / TRAIL_LEN)
                brightness = fade * motion_val

            brightness = min(1.0, max(0.0, brightness))
            if brightness < 0.02:
                continue

            ch = _char_grid[r, c]
            px = c * CELL_W + 2
            py = r * CELL_H + CELL_H - 2

            if dist >= 0 and dist < 1.5:
                g = int(255 * brightness)
                rb = int(230 * brightness)
                color = (rb, g, rb)
            else:
                g = int(220 * brightness)
                rv = int(25 * brightness)
                bv = int(10 * brightness)
                color = (bv, g, rv)

            cv2.putText(canvas, ch, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color, 1, cv2.LINE_AA)

    if _frame[0] % 60 == 0:
        ac = sum(1 for x in _columns if x['active'])
        mc = int(np.sum(_motion_acc > 0.15))
        print('  [f', _frame[0], '] active:', ac, '/', cols, ' cells:', mc)

    out_rgb = canvas[:, :, ::-1].copy()
    rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
    rgba[:, :, :3] = out_rgb.astype(np.float32) / 255.0
    scriptOp.copyNumpyArray(rgba)
'''

print('  [OK] script_draw')

# ================================================================
# ブルーム
# ================================================================

bloom1 = p.create(bloomTOP, 'bloom1')
bloom1.par.maxbloomradius = 10
bloom1.par.bloomintensity = 1.5
bloom1.par.bloomthreshold = 0.2
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
movie_out.par.file = home + '/Desktop/output_matrix.mp4'
movie_out.inputConnectors[0].connect(final.outputConnectors[0])
movie_out.nodeX = 600
movie_out.nodeY = 200

print('  [OK] Output')

# ================================================================
# 検証
# ================================================================

all_n = [
    'movie1','mono1','res_detect','null_detect',
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
print('  MATRIX MOTION READY!')
print('=' * 60)
print('')
print('  FINAL_OUT → a で拡大')
print('')
print('  録画:')
print('    op("/project1/movie_out").par.record = True')
print('    op("/project1/movie_out").par.record = False')
print('    → ~/Desktop/output_matrix.mp4')
print('')
print('  調整 (script_draw callback):')
print('    CELL_W/H=14/16  小→密 大→粗い')
print('    TRAIL_LEN=12  大→軌跡長い')
print('    MOTION_FADE=0.85  大→残像残る')
print('    CHAR_CHANGE_RATE=0.25  大→文字変化速い')
print('    MOTION_THRESH=15  小→敏感')
print('=' * 60)
