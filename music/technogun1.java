// ================================================================
//  起承転結 — 太鼓と笛からクラブへ
//  130 BPM / D陰音階 (D-E-F-A-Bb) / 5セクション / 128小節
// ================================================================

setcps(130/60/4)

// ─── 太鼓 (spacedrum_bd で太鼓感を出す) ─────────────────────
const taiko    = sound("spacedrum_bd").n("<0 2 4 1 5 3>")
                  .euclid(3,8).gain(0.92).room(0.45)
const taikoBig = sound("spacedrum_bd").n("<6 7 8 9>")
                  .euclid(1,4).gain(1.0).room(0.65)
const taikoHi  = sound("spacedrum_ht").n("<0 1 2 3 4>")
                  .euclid(5,12).gain(0.65).room(0.3)
const taikoMid = sound("spacedrum_mt").n("<0 1>")
                  .euclid(7,16).gain(0.5)

// ─── 篠笛/尺八風 (triangle波 + 遅いattack) ───────────────────
const flute = note("d5 ~ ~ ~ eb5 ~ f5 ~ ~ a5 ~ ~ bb5 ~ ~ ~ a5 ~ f5 ~ ~ e5 ~ ~ d5 ~ ~ ~ ~ ~ ~ ~")
  .sound("triangle").slow(2)
  .attack(0.18).release(0.55)
  .gain(0.52).room(0.88).lpf(3200)
  .delay(0.12).delaytime(0.375).delayfeedback(0.25)

const flute2 = note("<d5 ~ f5 ~ a5 ~ f5 ~> <bb5 ~ a5 ~ f5 ~ e5 ~>")
  .sound("triangle").slow(4)
  .attack(0.15).release(0.5)
  .gain(0.55).room(0.9).lpf(3000)

// ─── ミリダンガム (民族打楽器テクスチャ) ─────────────────────
const ta   = sound("mridangam_ta").n("<0 1 2 0 3>").euclid(5,16).gain(0.55)
const na   = sound("mridangam_na").n("<0 2 4 1>").euclid(7,12).gain(0.45)
const thom = sound("mridangam_thom").n("<0 1 2>").euclid(4,9).gain(0.48)

// ─── TR-909 テクノ打楽器 ─────────────────────────────────────
const bd = sound("tr909_bd").euclid(5,16).gain(0.9)
const sd = sound("tr909_sd").n("<0 1 2 3>").euclid(3,16).gain(0.65)
const hh = sound("tr909_hh").euclid(7,16).gain(0.25)
const oh = sound("tr909_oh").n("<0 1 2>").euclid(3,16).gain(0.3)
const cr = sound("tr909_cr").n("<0 1 2 3>").euclid(2,16).gain(0.2)

// ─── スペースドラム テクスチャ ────────────────────────────────
const space = sound("spacedrum_hh").euclid(5,12).gain(0.15).hpf(5000)

// ─── D陰音階アシッドベース ────────────────────────────────────
const bass = note("<d1 f1 bb0 c1>").slow(2)
  .sound("sawtooth").gain(0.85)
  .lpf("350 900 400 1400 600 1800 300 700").lpq(14)
  .attack(0.005).release(0.14)

const sub = note("<d0 f0 bb-1 c0>").slow(2)
  .sound("sine").gain(0.6).lpf(80)

// ─── リードシンセ (民謡スケールでクラブを貫く) ───────────────
const lead = note("<d5 ~ f5 ~ a5 ~ bb5 ~ a5 ~ f5 ~ e5 ~ d5 ~>")
  .sound("sawtooth").slow(2)
  .gain(0.48).lpf(3200).room(0.5)
  .delay(0.18).delaytime(0.375).delayfeedback(0.35)

// ─── パッド ──────────────────────────────────────────────────
const pad = note("<[d3,f3,a3] [bb2,d3,f3] [c3,eb3,g3] [a2,c3,e3]>")
  .sound("supersaw").slow(4)
  .gain(0.2).room(0.88).lpf(800)

// ─── 起: 太鼓と笛だけ [16小節] ──────────────────────────────
const ki = stack(
  taikoBig.gain(0.95),
  taiko.gain(0.82),
  taikoHi.gain(0.58),
  na.gain(0.38),
  flute.gain(0.52),
  note("d1 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~").slow(16)
    .sound("sine").gain(0.15).lpf(60)
)

// ─── 承①: テクノ侵入 [16小節] ──────────────────────────────
const sho1 = stack(
  taikoBig.gain(0.88),
  taiko.gain(0.75),
  taikoHi.gain(0.52),
  ta.gain(0.48), na.gain(0.38),
  flute2.gain(0.48),
  bd.gain(0.45),
  bass.gain(0.35).lpf(500),
  sub.gain(0.3),
  pad.gain(0.12).lpf(550)
)

// ─── 承②: 融合加速 [32小節] ────────────────────────────────
const sho2 = stack(
  taikoBig.gain(0.72),
  taiko.gain(0.62),
  taikoHi.gain(0.48),
  ta.gain(0.52), na.gain(0.42),
  flute2.gain(0.4),
  bd.gain(0.75),
  sd.gain(0.52),
  hh,
  bass.gain(0.7),
  sub.gain(0.5),
  pad.gain(0.18).lpf(950)
)

// ─── 転: 太鼓×クラブ全開 [48小節] ──────────────────────────
const ten = stack(
  taikoBig.gain(0.82),
  taiko.gain(0.68),
  taikoHi.gain(0.55),
  taikoMid.gain(0.45),
  ta.gain(0.58), na.gain(0.48), thom.gain(0.5),
  bd.gain(0.95),
  sd.gain(0.7),
  hh.gain(0.3),
  oh.gain(0.28),
  cr.gain(0.18),
  space,
  bass.gain(0.95),
  sub.gain(0.68),
  pad.gain(0.35).lpf(2000),
  lead.gain(0.48)
)

// ─── 結: 余韻 [16小節] ──────────────────────────────────────
const ketsu = stack(
  taiko.gain(0.55),
  taikoBig.gain(0.62),
  ta.gain(0.4),
  flute2.gain(0.38),
  bd.gain(0.4),
  bass.gain(0.28).lpf(300),
  sub.gain(0.4),
  pad.gain(0.5).lpf(550).room(0.98)
)

// ─── アレンジ [128小節 ≈ 4分] ───────────────────────────────
timeCat(
  [16, ki],
  [16, sho1],
  [32, sho2],
  [48, ten],
  [16, ketsu]
)
