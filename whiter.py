# ================================================================
# TouchDesigner 2025.31760
# Motion Glyph — 完全版
# ================================================================
#
# 変化量に応じた黒丸で動きを表現
#   ● = 動きの塊（塗りつぶし）
#   ○ = 重要な輪郭（縁のみ）
#   白 = 動きなし
#
# 使い方:
#   1. File → New → Textport にペースト
#   2. FINAL_OUT → a で拡大
#
# 録画:
#   op('/project1/movie_out').par.record = True
#   op('/project1/movie_out').par.record = False
#   → ~/Desktop/output_glyph.mp4
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
print('  Motion Glyph')
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

mono1 = p.create(monochromeTOP, 'mono1')
mono1.inputConnectors[0].connect(movie1.outputConnectors[0])
mono1.nodeX = -600
mono1.nodeY = 400

res_sample = p.create(resolutionTOP, 'res_sample')
res_sample.par.resolutionw = 200
res_sample.par.resolutionh = 112
res_sample.inputConnectors[0].connect(mono1.outputConnectors[0])
res_sample.nodeX = -400
res_sample.nodeY = 400

null_sample = p.create(nullTOP, 'null_sample')
null_sample.inputConnectors[0].connect(res_sample.outputConnectors[0])
null_sample.nodeX = -200
null_sample.nodeY = 400

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

# ===== パラメータ =====
GRID = 3             # グリッド間隔（小=密）
MIN_DIFF = 0.008     # 変化量の最低閾値（小=感度高い）
MAX_DOT_R = 5        # 最大ドット半径
MIN_DOT_R = 1        # 最小ドット半径
SOBEL_THRESH = 25    # エッジ閾値（小=○が増える）
NORM_SCALE = 0.25    # 正規化スケール（小=小さい変化でも反映）

COL_BG = (255, 255, 255)
COL_DOT = (25, 25, 25)

_prev = None
_frame = [0]

def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    global _prev
    _frame[0] += 1

    samp_op = op('null_sample')
    if samp_op is None:
        return

    try:
        arr = samp_op.numpyArray()
    except:
        return
    if arr is None:
        return

    sh, sw = arr.shape[:2]
    if sh < 5 or sw < 5:
        return

    gray = arr[:, :, 0]
    gray_u8 = (gray * 255).astype(np.uint8)

    if _prev is None:
        _prev = gray.copy()
        out_w, out_h = 1280, 720
        canvas = np.full((out_h, out_w, 3), COL_BG, dtype=np.uint8)
        rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
        rgba[:, :, :3] = canvas.astype(np.float32) / 255.0
        scriptOp.copyNumpyArray(rgba)
        return

    # フレーム差分
    diff = np.abs(gray.astype(np.float32) - _prev.astype(np.float32))
    _prev = gray.copy()

    # エッジ検出
    sx_sobel = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    sy_sobel = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sx_sobel**2 + sy_sobel**2)

    out_w, out_h = 1280, 720
    canvas = np.full((out_h, out_w, 3), COL_BG, dtype=np.uint8)

    scale_x = out_w / sw
    scale_y = out_h / sh

    count = 0

    for gy in range(0, sh, GRID):
        for gx in range(0, sw, GRID):
            d = diff[gy, gx]

            if d < MIN_DIFF:
                continue

            ox = int(gx * scale_x)
            oy = int(gy * scale_y)

            if ox < 3 or ox > out_w - 3 or oy < 3 or oy > out_h - 3:
                continue

            norm = min(d / NORM_SCALE, 1.0)
            r = int(MIN_DOT_R + norm * (MAX_DOT_R - MIN_DOT_R))

            seed = (gx * 31 + gy * 17 + _frame[0]) % 100
            jx = int((seed % 5 - 2) * norm)
            jy = int(((seed * 3) % 5 - 2) * norm)
            ox = max(r + 1, min(out_w - r - 1, ox + jx))
            oy = max(r + 1, min(out_h - r - 1, oy + jy))

            e = edge_mag[gy, gx]

            if e > SOBEL_THRESH and d > MIN_DIFF * 2:
                cv2.circle(canvas, (ox, oy), r, COL_DOT, 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (ox, oy), r, COL_DOT, -1, cv2.LINE_AA)

            count += 1

    if _frame[0] % 60 == 0:
        print('  [f', _frame[0], '] dots:', count)

    out_rgb = canvas[:, :, ::-1].copy()
    rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
    rgba[:, :, :3] = out_rgb.astype(np.float32) / 255.0
    scriptOp.copyNumpyArray(rgba)
'''

print('  [2] script_draw OK')

final = p.create(nullTOP, 'FINAL_OUT')
final.inputConnectors[0].connect(script_draw.outputConnectors[0])
final.nodeX = 400
final.nodeY = 400
final.viewer = True

movie_out = p.create(moviefileoutTOP, 'movie_out')
movie_out.par.file = home + '/Desktop/output_glyph.mp4'
movie_out.inputConnectors[0].connect(final.outputConnectors[0])
movie_out.nodeX = 400
movie_out.nodeY = 200

all_n = ['movie1','mono1','res_sample','null_sample',
         'script_draw','FINAL_OUT','movie_out']

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
print('  ■ 描画:')
print('    ● = 動きの塊')
print('    ○ = 重要な輪郭')
print('    白 = 動きなし')
print('')
print('  ■ 調整:')
print('    GRID=3  小→密')
print('    MIN_DIFF=0.008  小→感度高い')
print('    MAX_DOT_R=5  大→ドット大きい')
print('    SOBEL_THRESH=25  小→○が増える')
print('')
print('  ■ 録画:')
print("    op('/project1/movie_out').par.record = True")
print("    op('/project1/movie_out').par.record = False")
print('    → ~/Desktop/output_glyph.mp4')
print('')
print('=' * 60)
