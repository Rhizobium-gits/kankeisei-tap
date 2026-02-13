# ================================================================
# TouchDesigner 2025.31760
# 3-Mode Exclusive Switch + Audio Playback
# ================================================================
#
#   Mode A (静か):       Matrix    黒背景 + 緑の多言語文字
#   Mode B (中間):       Network   暗背景 + 色エッジ + ノード追跡
#   Mode C (盛り上がり): Particle  白背景 + 黒●○（動き連動）
#
#   ※ 3モードは排他的に切り替わる（同時表示なし）
#   ※ laugh_beat.mp3 をスピーカーで再生
#   ※ 録画時に音声も焼き込まれる
#
# 使い方:
#   1. File → New
#   2. Dialogs → Textport
#   3. 全体をペースト → Enter
#   4. FINAL_OUT → a で拡大表示
# ================================================================

import os

home = os.path.expanduser('~')
VIDEO = None
AUDIO = None
found_vid = []

for d in [home, home+'/Desktop', home+'/Downloads',
          home+'/Movies', home+'/Documents']:
    if os.path.isdir(d):
        try:
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if not os.path.isfile(fp):
                    continue
                fl = f.lower()
                if fl.endswith(('.mp4','.mov','.avi','.m4v','.mkv','.webm')):
                    found_vid.append(fp)
                    if f == '21.mp4' and VIDEO is None:
                        VIDEO = fp
                if fl == 'omaechare.mp3':
                    AUDIO = fp
        except:
            pass

if VIDEO is None and found_vid:
    VIDEO = found_vid[0]
if VIDEO is None:
    VIDEO = home + '/Desktop/video.mp4'
if AUDIO is None:
    AUDIO = home + '/Desktop/omaechare.mp3'

print('')
print('=' * 60)
print('  3-Mode Exclusive Switch + Audio')
print('=' * 60)
print('  Video:', VIDEO)
print('  Audio:', AUDIO)
print('  Video exists:', os.path.exists(VIDEO))
print('  Audio exists:', os.path.exists(AUDIO))
print('')

p = op('/project1')
for c in p.children:
    c.destroy()

# ================================================================
# 動画入力パイプライン
# ================================================================

movie1 = p.create(moviefileinTOP, 'movie1')
movie1.par.file = VIDEO
movie1.nodeX = -800
movie1.nodeY = 400

# メイン背景用（1280x720）
res_main = p.create(resolutionTOP, 'res_main')
res_main.par.resolutionw = 1280
res_main.par.resolutionh = 720
res_main.inputConnectors[0].connect(movie1.outputConnectors[0])
res_main.nodeX = -600
res_main.nodeY = 400

null_bg = p.create(nullTOP, 'null_bg')
null_bg.inputConnectors[0].connect(res_main.outputConnectors[0])
null_bg.nodeX = -400
null_bg.nodeY = 400

# 検出用低解像度モノクロ
mono1 = p.create(monochromeTOP, 'mono1')
mono1.inputConnectors[0].connect(movie1.outputConnectors[0])
mono1.nodeX = -600
mono1.nodeY = 200

res_det = p.create(resolutionTOP, 'res_det')
res_det.par.resolutionw = 200
res_det.par.resolutionh = 112
res_det.inputConnectors[0].connect(mono1.outputConnectors[0])
res_det.nodeX = -400
res_det.nodeY = 200

null_det = p.create(nullTOP, 'null_det')
null_det.inputConnectors[0].connect(res_det.outputConnectors[0])
null_det.nodeX = -200
null_det.nodeY = 200

# マトリックス用（閾値処理 → 高解像度）
thresh1 = p.create(thresholdTOP, 'thresh1')
thresh1.par.threshold = 0.35
thresh1.inputConnectors[0].connect(mono1.outputConnectors[0])
thresh1.nodeX = -600
thresh1.nodeY = 0

res_thr = p.create(resolutionTOP, 'res_thr')
res_thr.par.resolutionw = 80
res_thr.par.resolutionh = 45
res_thr.inputConnectors[0].connect(thresh1.outputConnectors[0])
res_thr.nodeX = -400
res_thr.nodeY = 0

null_thr = p.create(nullTOP, 'null_thr')
null_thr.inputConnectors[0].connect(res_thr.outputConnectors[0])
null_thr.nodeX = -200
null_thr.nodeY = 0

print('  [1/5] Video pipeline OK')

# ================================================================
# オーディオ入力 + 解析 + 再生
# ================================================================

audio1 = p.create(audiofileinCHOP, 'audio1')
audio1.par.file = AUDIO
audio1.par.play = True
audio1.par.volume = 1
audio1.nodeX = -800
audio1.nodeY = -200

# RMS解析
analyze1 = p.create(analyzeCHOP, 'analyze1')
analyze1.par.function = 6
analyze1.inputConnectors[0].connect(audio1.outputConnectors[0])
analyze1.nodeX = -600
analyze1.nodeY = -200

null_rms = p.create(nullCHOP, 'null_rms')
null_rms.inputConnectors[0].connect(analyze1.outputConnectors[0])
null_rms.nodeX = -400
null_rms.nodeY = -200

# 低音域 → RMSで代用（audiofilterCHOPは非対応バージョンあり）
null_bass = p.create(nullCHOP, 'null_bass')
null_bass.inputConnectors[0].connect(analyze1.outputConnectors[0])
null_bass.nodeX = -200
null_bass.nodeY = -350

# スピーカー出力（音楽再生）
audio_out = p.create(audiodeviceoutCHOP, 'audio_out')
audio_out.inputConnectors[0].connect(audio1.outputConnectors[0])
audio_out.nodeX = -600
audio_out.nodeY = -500

print('  [2/5] Audio pipeline OK (speaker output enabled)')

# ================================================================
# Script TOP: 3モード排他描画
# ================================================================

script_draw = p.create(scriptTOP, 'script_draw')
script_draw.par.resolutionw = 1280
script_draw.par.resolutionh = 720
script_draw.nodeX = 200
script_draw.nodeY = 200

cb = script_draw.par.callbacks.eval()
cb.text = '''import numpy as np
import cv2
import math
import random as _rng

# ============================================================
# パラメータ
# ============================================================

# モード切替閾値（RMSスムージング後の値）
THRESH_LOW  = 0.035   # これ以下 → Mode C: Particle
THRESH_HIGH = 0.10    # これ以上 → Mode A: Matrix / 間 → Mode B: Network
SMOOTH = 0.88

# Mode A: Matrix（緑文字）
M_CHARS = list("ABCDEFabcdef01234567{}[]<>+=/*#@$%&!?")
M_JP = list("存在関係先立生命網絡接続情報流動変化")
M_ALL = M_CHARS + M_JP
M_GRID = 16          # 文字間隔（px）
M_FONT_SCALE = 0.35
M_THICKNESS = 1

# Mode B: Network（エッジ＋ノード）
N_MOTION_THRESH = 20
N_MIN_AREA = 600
N_DILATE = 5
N_BLUR_K = 11
N_HIST_LEN = 15
N_MIN_TRAVEL = 35.0
N_CT_MATCH_R = 80
N_NODE_MATCH_R = 70
N_MAX_AGE = 3
N_EDGE_MAX = 280
N_BEZIER = 14
N_DIST_THRESH = 3.0

# Mode C: Particle/Glyph（白背景＋黒●○）
G_GRID = 3
G_MIN_DIFF = 0.008
G_MAX_R = 5
G_NORM = 0.25
G_SOBEL = 25

# 色定義
C_GREEN      = (30, 220, 50)
C_GREEN_DIM  = (15, 110, 25)
C_GREEN_GLOW = (10, 60, 15)
C_BG_BLACK   = (0, 0, 0)
C_BG_WHITE   = (255, 255, 255)
C_DOT_BK     = (25, 25, 25)
C_APPROACH   = (50, 30, 210)
C_SEPARATE   = (210, 140, 60)
C_STABLE     = (190, 185, 180)
C_NODE_HI    = (80, 50, 230)
C_NODE_MD    = (240, 230, 180)
C_NODE_LO    = (230, 230, 230)
C_GLOW_R     = (60, 35, 170)
C_GLOW_W     = (160, 155, 150)
C_LABEL      = (160, 160, 160)

# ============================================================
# 状態変数
# ============================================================
_prev = None
_frame = [0]
_level = [0.0]
_mode_str = ['?']
# Network tracking
_ct = {}
_next_ct = [1]
_tracked = {}
_next_nid = [1]
_prev_dists = {}
# Matrix: 各グリッド位置の文字をキャッシュ
_char_map = {}
_char_timer = {}


# ============================================================
# ユーティリティ
# ============================================================

def _get_audio():
    rms = 0.0
    bass = 0.0
    try:
        a = op('null_rms')
        if a and a.numChans > 0:
            rms = abs(float(a[0]))
    except:
        pass
    try:
        b = op('null_bass')
        if b and b.numChans > 0:
            bass = abs(float(b[0]))
    except:
        pass
    return rms, bass


def _travel(h):
    if len(h) < 2:
        return 0.0
    t = 0.0
    for i in range(1, len(h)):
        dx = h[i][0] - h[i-1][0]
        dy = h[i][1] - h[i-1][1]
        t += math.sqrt(dx*dx + dy*dy)
    return t


def _key_pts(cnt, sx, sy, w, h):
    pts = []
    M = cv2.moments(cnt)
    if M['m00'] <= 0:
        return pts
    rcx = M['m10'] / M['m00']
    rcy = M['m01'] / M['m00']
    cx = max(5, min(w-5, int(rcx * sx)))
    cy = max(5, min(h-5, int(rcy * sy)))
    pts.append((cx, cy))
    pp = cnt.reshape(-1, 2).astype(np.float64)
    if len(pp) < 3:
        return pts
    dd = np.sqrt((pp[:,0]-rcx)**2 + (pp[:,1]-rcy)**2)
    i1 = np.argmax(dd)
    pts.append((max(5,min(w-5,int(pp[i1][0]*sx))),
                max(5,min(h-5,int(pp[i1][1]*sy)))))
    d2 = np.sqrt((pp[:,0]-pp[i1][0])**2 + (pp[:,1]-pp[i1][1])**2)
    comb = dd + d2 * 0.5
    comb[i1] = 0
    i2 = np.argmax(comb)
    pts.append((max(5,min(w-5,int(pp[i2][0]*sx))),
                max(5,min(h-5,int(pp[i2][1]*sy)))))
    return pts


def _bezier(x1, y1, x2, y2, tv):
    dx = x2 - x1
    dy = y2 - y1
    ca = 0.3 * math.sin(tv * 0.08 + (x1+y2) * 0.01)
    cpx = (x1+x2)*0.5 - dy*ca
    cpy = (y1+y2)*0.5 + dx*ca
    out = []
    for i in range(N_BEZIER + 1):
        s = i / N_BEZIER
        inv = 1.0 - s
        out.append([int(inv*inv*x1 + 2*inv*s*cpx + s*s*x2),
                     int(inv*inv*y1 + 2*inv*s*cpy + s*s*y2)])
    return np.array(out, np.int32)


# ============================================================
# 動体追跡（Network用）
# ============================================================
def _track_objects(diff, det_w, det_h, out_w, out_h):
    diff_u8 = (diff * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(diff_u8, (N_BLUR_K, N_BLUR_K), 0)
    _, mask = cv2.threshold(blur, N_MOTION_THRESH, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=N_DILATE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    sx = out_w / det_w
    sy = out_h / det_h

    fc = []
    fc_cnt = []
    for cnt in contours:
        if cv2.contourArea(cnt) < N_MIN_AREA:
            continue
        MM = cv2.moments(cnt)
        if MM['m00'] > 0:
            fc.append((MM['m10']/MM['m00'], MM['m01']/MM['m00']))
            fc_cnt.append(cnt)

    matched_ct = set()
    used_fc = set()
    for cid, ct in list(_ct.items()):
        best_d = N_CT_MATCH_R
        best_j = -1
        for j, (fcx, fcy) in enumerate(fc):
            if j in used_fc:
                continue
            dd = math.sqrt((ct['cx']-fcx)**2 + (ct['cy']-fcy)**2)
            if dd < best_d:
                best_d = dd
                best_j = j
        if best_j >= 0:
            fx, fy = fc[best_j]
            ct['cx'] = fx
            ct['cy'] = fy
            ct['history'].append((fx, fy))
            if len(ct['history']) > N_HIST_LEN:
                ct['history'] = ct['history'][-N_HIST_LEN:]
            ct['miss'] = 0
            ct['cnt'] = fc_cnt[best_j]
            matched_ct.add(cid)
            used_fc.add(best_j)

    for j in range(len(fc)):
        if j not in used_fc:
            cid = _next_ct[0]
            _next_ct[0] += 1
            _ct[cid] = {'cx':fc[j][0], 'cy':fc[j][1],
                         'history':[fc[j]], 'miss':0, 'cnt':fc_cnt[j]}

    for cid in list(_ct.keys()):
        if cid not in matched_ct:
            _ct[cid]['miss'] = _ct[cid].get('miss', 0) + 1
            if _ct[cid]['miss'] > 5:
                del _ct[cid]

    moving = []
    for cid, ct in _ct.items():
        if ct.get('miss', 0) > 0:
            continue
        if _travel(ct['history']) >= N_MIN_TRAVEL:
            moving.append(ct['cnt'])

    detected = []
    for cnt in moving:
        for pt in _key_pts(cnt, sx, sy, out_w, out_h):
            detected.append(pt)

    matched_n = set()
    used_d = set()
    for nid, nd in list(_tracked.items()):
        best_d = N_NODE_MATCH_R
        best_j = -1
        for j, (dx, dy) in enumerate(detected):
            if j in used_d:
                continue
            dist = math.sqrt((nd['x']-dx)**2 + (nd['y']-dy)**2)
            if dist < best_d:
                best_d = dist
                best_j = j
        if best_j >= 0:
            dx, dy = detected[best_j]
            nd['vx'] = dx - nd['x']
            nd['vy'] = dy - nd['y']
            nd['x'] = dx
            nd['y'] = dy
            nd['age'] = 0
            matched_n.add(nid)
            used_d.add(best_j)

    for j, (dx, dy) in enumerate(detected):
        if j not in used_d:
            nid = _next_nid[0]
            _next_nid[0] += 1
            _tracked[nid] = {'x':dx,'y':dy,'vx':0,'vy':0,'age':0}

    for nid in [nn for nn in _tracked if nn not in matched_n]:
        _tracked[nid]['age'] += 1
        if _tracked[nid]['age'] > N_MAX_AGE:
            for k in [k for k in _prev_dists if nid in k]:
                del _prev_dists[k]
            del _tracked[nid]


# ============================================================
# Mode A: Matrix（黒背景 + 緑の多言語文字）
# ============================================================
def _draw_matrix(canvas, out_w, out_h, rms, bass):
    thr_op = op('null_thr')
    if thr_op is None:
        return
    try:
        arr = thr_op.numpyArray()
    except:
        return
    if arr is None:
        return
    th, tw = arr.shape[:2]
    if th < 2 or tw < 2:
        return

    scx = out_w / tw
    scy = out_h / th
    t = absTime.seconds
    intensity = min(rms * 8, 1.0)
    frame = _frame[0]

    for y in range(th):
        for x in range(tw):
            if arr[y, x, 0] < 0.5:
                continue
            ox = int(x * scx)
            oy = int(y * scy)

            # 有機的な揺らぎ
            jx = math.sin(t * 0.5 + ox * 0.008) * 2 * (1 + intensity)
            jy = math.cos(t * 0.4 + oy * 0.006) * 2 * (1 + intensity)
            ox = max(2, min(out_w - 10, int(ox + jx)))
            oy = max(8, min(out_h - 2, int(oy + jy)))

            # 文字を選択（位置ベースでキャッシュ、時々変化）
            key = (x, y)
            if key not in _char_map or frame % 30 == 0:
                _rng.seed(x * 31 + y * 17 + frame // 30)
                _char_map[key] = _rng.choice(M_ALL)

            ch = _char_map[key]

            # 明るさのバリエーション（距離や位置で変化）
            bright = 0.4 + 0.6 * math.sin(t * 0.3 + x * 0.2 + y * 0.15) ** 2
            bright = min(1.0, bright + intensity * 0.3)

            g = int(120 + 135 * bright)
            r = int(15 + 30 * bright)
            b = int(10 + 20 * bright)
            col = (b, g, r)  # BGR

            scale = M_FONT_SCALE * (0.8 + 0.4 * bright)

            cv2.putText(canvas, ch, (ox, oy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        col, M_THICKNESS, cv2.LINE_AA)


# ============================================================
# Mode B: Network（暗背景 + 色エッジ + ノード）
# ============================================================
def _draw_network(canvas, t):
    imp = {}
    for nid, nd in _tracked.items():
        imp[nid] = math.sqrt(nd['vx']**2 + nd['vy']**2)

    ids = list(_tracked.keys())
    n = len(ids)
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            a, b = ids[i], ids[j]
            na, nb = _tracked[a], _tracked[b]
            dx = nb['x'] - na['x']
            dy = nb['y'] - na['y']
            dd = math.sqrt(dx*dx + dy*dy)
            if dd > N_EDGE_MAX or dd < 10:
                continue
            pair = (min(a,b), max(a,b))
            prev_d = _prev_dists.get(pair, dd)
            delta = dd - prev_d
            _prev_dists[pair] = dd
            if delta < -N_DIST_THRESH:
                col = C_APPROACH
            elif delta > N_DIST_THRESH:
                col = C_SEPARATE
            else:
                col = C_STABLE
            edges.append((a, b, col))

    for (a, b, col) in edges:
        imp[a] = imp.get(a, 0) + 2.0
        imp[b] = imp.get(b, 0) + 2.0

    for (a, b, col) in edges:
        na, nb = _tracked[a], _tracked[b]
        pts = _bezier(na['x'], na['y'], nb['x'], nb['y'], t)
        cv2.polylines(canvas, [pts], False, col, 1, cv2.LINE_AA)

    iv = list(imp.values())
    im = max(iv) if iv and max(iv) > 0 else 1

    for nid, nd in _tracked.items():
        px, py = int(nd['x']), int(nd['y'])
        nm = imp.get(nid, 0) / im
        if nm > 0.6:
            cv2.circle(canvas, (px,py), 4, C_NODE_HI, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px,py), 8, C_GLOW_R, 1, cv2.LINE_AA)
        elif nm > 0.3:
            cv2.circle(canvas, (px,py), 3, C_NODE_MD, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px,py), 6, C_GLOW_W, 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px,py), 2, C_NODE_LO, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px,py), 5, C_GLOW_W, 1, cv2.LINE_AA)
        if nm > 0.2:
            lbl = "codecore " + str(nid)
            cv2.putText(canvas, lbl, (px+7, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        C_LABEL, 1, cv2.LINE_AA)


# ============================================================
# Mode C: Particle/Glyph（白背景 + 黒●○）
# ============================================================
def _draw_particle(canvas, diff, gray_u8, out_w, out_h, det_w, det_h):
    sx_s = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    sy_s = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    emag = np.sqrt(sx_s**2 + sy_s**2)

    scx = out_w / det_w
    scy = out_h / det_h
    frame = _frame[0]

    for gy in range(0, det_h, G_GRID):
        for gx in range(0, det_w, G_GRID):
            d = diff[gy, gx]
            if d < G_MIN_DIFF:
                continue
            ox = int(gx * scx)
            oy = int(gy * scy)
            if ox < 3 or ox > out_w-3 or oy < 3 or oy > out_h-3:
                continue
            norm = min(d / G_NORM, 1.0)
            r = max(1, int(1 + norm * (G_MAX_R - 1)))

            seed = (gx*31 + gy*17 + frame) % 100
            jx = int((seed % 5 - 2) * norm)
            jy = int(((seed*3) % 5 - 2) * norm)
            ox = max(r+1, min(out_w-r-1, ox+jx))
            oy = max(r+1, min(out_h-r-1, oy+jy))

            e = emag[gy, gx]
            # 動きに応じて形状を変える
            if e > G_SOBEL and d > G_MIN_DIFF * 2:
                # エッジ近傍 → 輪郭（○）
                cv2.circle(canvas, (ox, oy), r, C_DOT_BK, 1, cv2.LINE_AA)
            else:
                # 内部 → 塗りつぶし（●）
                cv2.circle(canvas, (ox, oy), r, C_DOT_BK, -1, cv2.LINE_AA)


# ============================================================
# メインループ
# ============================================================
def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    global _prev
    _frame[0] += 1

    # 検出用ピクセル取得
    det_op = op('null_det')
    bg_op = op('null_bg')
    if det_op is None or bg_op is None:
        return

    try:
        det_arr = det_op.numpyArray()
        bg_arr = bg_op.numpyArray()
    except:
        return
    if det_arr is None or bg_arr is None:
        return

    out_h, out_w = bg_arr.shape[:2]
    det_h, det_w = det_arr.shape[:2]
    if out_h < 10 or det_h < 10:
        return

    gray = det_arr[:, :, 0]
    gray_u8 = (gray * 255).astype(np.uint8)

    if _prev is None:
        _prev = gray.copy()
        canvas = np.full((out_h, out_w, 3), C_BG_BLACK, dtype=np.uint8)
        rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
        scriptOp.copyNumpyArray(rgba)
        return

    # フレーム差分
    diff = np.abs(gray.astype(np.float32) - _prev.astype(np.float32))
    _prev = gray.copy()

    # オーディオ
    rms, bass = _get_audio()
    _level[0] = _level[0] * SMOOTH + rms * (1.0 - SMOOTH)
    level = _level[0]

    t = absTime.seconds

    # 動体追跡（Network用、常に更新）
    _track_objects(diff, det_w, det_h, out_w, out_h)

    # ========================================
    # 排他的モード選択
    # ========================================
    if level < THRESH_LOW:
        # === Mode C: Particle/Glyph（静か） ===
        _mode_str[0] = 'C:Particle'
        canvas = np.full((out_h, out_w, 3), C_BG_WHITE, dtype=np.uint8)
        _draw_particle(canvas, diff, gray_u8, out_w, out_h, det_w, det_h)

    elif level < THRESH_HIGH:
        # === Mode B: Network（中間） ===
        _mode_str[0] = 'B:Network'
        bg_dark = (bg_arr[:, :, :3] * 255 * 0.3).astype(np.uint8)
        canvas = bg_dark[:, :, ::-1].copy()
        _draw_network(canvas, t)

    else:
        # === Mode A: Matrix（盛り上がり） ===
        _mode_str[0] = 'A:Matrix'
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        _draw_matrix(canvas, out_w, out_h, rms, bass)

    # デバッグ出力
    if _frame[0] % 90 == 0:
        print('  [f', _frame[0], ']',
              'mode:', _mode_str[0],
              'rms:', round(rms, 4),
              'smooth:', round(level, 4),
              'nodes:', len(_tracked))

    # BGR→RGB→RGBA出力
    out_rgb = canvas[:, :, ::-1].copy()
    rgba = np.ones((out_h, out_w, 4), dtype=np.float32)
    rgba[:, :, :3] = out_rgb.astype(np.float32) / 255.0
    scriptOp.copyNumpyArray(rgba)
'''

print('  [3/5] Script TOP (3 modes) OK')

# ================================================================
# ブルーム + 出力
# ================================================================

bloom1 = p.create(bloomTOP, 'bloom1')
bloom1.par.maxbloomradius = 12
bloom1.par.bloomintensity = 1.8
bloom1.par.bloomthreshold = 0.25
bloom1.inputConnectors[0].connect(script_draw.outputConnectors[0])
bloom1.nodeX = 400
bloom1.nodeY = 200

final = p.create(nullTOP, 'FINAL_OUT')
final.inputConnectors[0].connect(bloom1.outputConnectors[0])
final.nodeX = 600
final.nodeY = 200
final.viewer = True

print('  [4/5] Bloom + FINAL_OUT OK')

# ================================================================
# 動画出力（音声付き）
# ================================================================

movie_out = p.create(moviefileoutTOP, 'movie_out')
movie_out.par.file = home + '/Desktop/output_3mode.mp4'
movie_out.inputConnectors[0].connect(final.outputConnectors[0])
movie_out.nodeX = 600
movie_out.nodeY = 0

# 音声を動画に焼き込む
audio_param_set = False
for pname in ['audiochop', 'audioinput', 'choppath']:
    try:
        setattr(movie_out.par, pname, '/project1/audio1')
        audio_param_set = True
        print('  [5/5] movie_out audio param:', pname, '= /project1/audio1')
        break
    except:
        pass

if not audio_param_set:
    print('  [5/5] movie_out created (set Audio CHOP to audio1 manually)')

# ================================================================
# 検証
# ================================================================

all_nodes = [
    'movie1','res_main','null_bg','mono1','res_det','null_det',
    'thresh1','res_thr','null_thr',
    'audio1','analyze1','null_rms',
    'null_bass','audio_out',
    'script_draw','bloom1','FINAL_OUT','movie_out'
]

ok = 0
for name in all_nodes:
    nd = op('/project1/' + name)
    if nd is None:
        print('  [MISSING] ' + name)
    else:
        e = nd.errors()
        if e and 'Non-Commercial' not in e:
            print('  [ERROR] ' + name + ': ' + e)
        else:
            ok += 1

print('')
print('=' * 60)
print('  SETUP COMPLETE!  ' + str(ok) + '/' + str(len(all_nodes)) + ' nodes OK')
print('=' * 60)
print('')
print('  FINAL_OUT → a キーで全画面')
print('')
print('  ■ 3モード（音量で排他切替）:')
print('    静か       → C: Particle  白背景 + 黒●○')
print('    中間       → B: Network   暗背景 + 色エッジ + ノード')
print('    盛り上がり → A: Matrix    黒背景 + 緑の多言語文字')
print('')
print('  ■ 音楽:')
print('    スピーカーから自動再生（audio_out）')
print('    録画時にも音声が焼き込まれる（movie_out）')
print('')
print('  ■ 調整:')
print('    THRESH_LOW=0.035   小→Particleが減る')
print('    THRESH_HIGH=0.10   小→Matrixが増える')
print('    SMOOTH=0.88        大→切替が滑らか')
print('')
print('  ■ 録画（音声付き）:')
print("    op('/project1/movie_out').par.record = True")
print("    op('/project1/movie_out').par.record = False")
print('    → ~/Desktop/output_3mode.mp4')
print('')
