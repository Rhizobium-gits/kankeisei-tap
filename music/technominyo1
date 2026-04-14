// ================================================================
//  起承転結 民謡×テクノ — Mridangam × TR-909
//  130 BPM / D陰音階 (D-E-F-A-Bb) / timeCat arrangement
// ================================================================

setcps(130/60/4)

// ─── 民族打楽器: Mridangam ────────────────────────────────────
const ta   = sound("mridangam_ta").n("<0 1 2 0 3>").euclid(5,16).gain(0.62)
const na   = sound("mridangam_na").n("<0 2 4 1 3>").euclid(7,12).gain(0.5)
const dhin = sound("mridangam_dhin").n("<0 1 2>").euclid(3,8).gain(0.58)
const thom = sound("mridangam_thom").n("<0 2 1>").euclid(4,9).gain(0.52)
const ka   = sound("mridangam_ka").n("<0 1 3 2>").euclid(9,16).gain(0.4)

// ─── テクノ打楽器: TR-909 ─────────────────────────────────────
const bd = sound("tr909_bd").euclid(5,16).gain(0.9)
const sd = sound("tr909_sd").euclid(3,16).gain(0.65)
const hh = sound("tr909_hh").euclid(7,16).gain(0.25)
const oh = sound("tr909_oh").n("<0 1 2>").euclid(3,16).gain(0.3)
const cr = sound("tr909_cr").n("<0 1 2 3>").euclid(2,16).gain(0.2)

// ─── スペースドラム (大気的テクスチャ) ──────────────────────
const space = sound("spacedrum_hh").euclid(5,12).gain(0.18).hpf(5000)

// ─── D陰音階アシッドベース ────────────────────────────────────
const bass = note("<d1 f1 bb0 c1>").slow(2)
  .sound("sawtooth").gain(0.85)
  .lpf("350 900 400 1400 600 1800 300 700").lpq(14)
  .attack(0.005).release(0.14)

const sub = note("<d0 f0 bb-1 c0>").slow(2)
  .sound("sine").gain(0.6).lpf(80)

// ─── 民謡風メロディ ──────────────────────────────────────────
const folkMel = note("d4 ~ ~ e4 f4 ~ a4 ~ ~ bb4 ~ ~ a4 ~ f4 ~")
  .sound("sawtooth").gain(0.42).lpf(2500).room(0.45)
  .delay(0.15).delaytime(0.375).delayfeedback(0.3)

// ─── 対旋律 (上声部) ─────────────────────────────────────────
const counter = note("<d5 ~ f5 ~ a5 ~ bb5 ~ a5 ~ f5 ~ e5 ~ d5 ~>")
  .sound("sine").slow(2)
  .gain(0.28).room(0.78)

// ─── ダークパッド ─────────────────────────────────────────────
const pad = note("<[d3,f3,a3] [c3,eb3,g3] [bb2,d3,f3] [a2,c3,e3]>")
  .sound("supersaw").slow(4)
  .gain(0.22).room(0.88).lpf(800)

// ─── 起: 民謡の目覚め [16小節] ──────────────────────────────
const ki = stack(
  ta,
  na.gain(0.42),
  thom.gain(0.44),
  note("d2 ~ ~ ~ ~ ~ ~ ~").slow(8)
    .sound("sine").gain(0.25).lpf(120),
  pad.gain(0.12).lpf(450).room(0.96)
     .delay(0.5).delaytime(0.25).delayfeedback(0.5)
)

// ─── 承: 機械との対話 [48小節] ──────────────────────────────
const sho = stack(
  bd,
  sd.gain(0.52),
  hh,
  ta, na,
  bass.gain(0.7),
  sub.gain(0.5),
  pad.gain(0.18).lpf(950),
  folkMel.slow(2).gain(0.3).lpf(2000)
)

// ─── 転: 融合、爆発 [48小節] ────────────────────────────────
const ten = stack(
  bd.gain(0.95),
  sd.gain(0.7),
  hh.gain(0.3),
  oh.gain(0.28),
  cr.gain(0.18),
  ta.gain(0.68), na.gain(0.55),
  dhin.gain(0.6), thom.gain(0.55), ka.gain(0.42),
  space,
  bass.gain(0.95),
  sub.gain(0.68),
  pad.gain(0.38).lpf(2100),
  folkMel.gain(0.48).lpf(3200),
  counter.gain(0.32)
)

// ─── 結: 民謡への回帰 [16小節] ──────────────────────────────
const ketsu = stack(
  ta.gain(0.48),
  na.gain(0.38),
  thom.gain(0.4),
  bd.gain(0.38),
  bass.gain(0.3).lpf(280),
  sub.gain(0.42),
  pad.gain(0.52).lpf(550).room(0.98),
  counter.gain(0.18)
)

// ─── アレンジ [128小節 ≈ 4分] ───────────────────────────────
timeCat(
  [16, ki],
  [48, sho],
  [48, ten],
  [16, ketsu]
)
