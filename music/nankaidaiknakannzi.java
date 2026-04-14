
  bass.gain(0.82),
  sub.gain(0.55),
  pad.gain(0.22).lpf(1200),
  stabs.gain(0.28),
  arp.gain(0.32).lpf(2500)
)

// ─── 承②: プレドロップ [16小節] ─────────────────────────────
const sho2 = stack(
  kick.gain(0.95),
  hh16.gain(0.18),
  bass.gain(0.92),
  sub.gain(0.62),
  pad.gain(0.32).lpf(2200),
  arp.gain(0.42).lpf(3200),
  stabs.gain(0.38),
  shimmer.gain(0.15)
)

// ─── 転: THE DROP ぶち上がり全開放 [64小節] ─────────────────
const ten = stack(
  kick.gain(0.96),
  snare.gain(0.78),
  hh16.gain(0.24),
  oh.gain(0.34),
  cr.gain(0.2),
  clap.gain(0.52),
  bass.gain(0.96),
  sub.gain(0.68),
  pad.gain(0.38).lpf(2500),
  arpFast.gain(0.4).lpf(4000),
  lead.gain(0.58),
  stabs.gain(0.35),
  shimmer.gain(0.22)
)

// ─── 結: アウトロ [16小節] ───────────────────────────────────
const ketsu = stack(
  kick.gain(0.72),
  snare.gain(0.55),
  hh.gain(0.18),
  bass.gain(0.6).lpf(500),
  sub.gain(0.42),
  pad.gain(0.5).lpf(1000).room(0.9),
  lead.gain(0.35).lpf(2500)
)

// ─── アレンジ [144小節 ≈ 6分45秒] ───────────────────────────
timeCat(
  [16, ki],
  [32, sho1],
  [16, sho2],
  [64, ten],
  [16, ketsu]
)
