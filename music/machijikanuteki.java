// ================================================================
//  森の中に迷って — Goldmund イメージ
//  60 BPM / Emaj7 / ambient piano × forest
//  strudel.cc にペーストして再生
// ================================================================

setcps(60/60/4)

// ─── 森の声 ──────────────────────────────────────────────────

// 葉のざわめき (tongue drum の超高域成分)
const leaves = sound("spacedrum_hh")
  .n("<0 1 2 3 4 5>").euclid(7,16)
  .gain(0.05).hpf(9000).room(0.98)

// 枝のきしみ
const branch = sound("spacedrum_ht")
  .n("<0 2 4 1 3>").euclid(3,16)
  .gain(0.1).room(0.9).lpf(2000)

// 遠くの水の滴 (tongue drum の倍音)
const water = sound("spacedrum_oh")
  .n("<0 1 2>").euclid(2,11)
  .gain(0.12).room(0.99).hpf(800)

// 森の低い鼓動 (2小節に1回)
const pulse = sound("spacedrum_bd")
  .n("<3 7 1 5>").euclid(1,8)
  .gain(0.18).room(0.88).lpf(400)

// ─── 大地のドローン ───────────────────────────────────────────
const earth = note("e1")
  .sound("sine").gain(0.06).lpf(80).room(0.99)

// ─── ピアノ ──────────────────────────────────────────────────

// コード進行: Emaj7 → Amaj7 → F#m7 → B7sus4 (各4小節)
const chords = note(
  "<[e3,g#3,b3,d#4] [a3,c#4,e4,g#4] [f#3,a3,c#4,e4] [b2,e3,f#3,a3]>"
).sound("superpiano").slow(4)
  .gain(0.22).room(0.99)

// ベース (各4小節)
const bass = note("<e2 a2 f#2 b2>")
  .sound("superpiano").slow(4)
  .gain(0.3).room(0.99)

// 主旋律: 1音 = 1小節、8小節で1フレーズ
const melody = note(
  "e4 ~ ~ ~ ~ ~ ~ ~ g#4 ~ ~ ~ ~ ~ ~ ~ " +
  "f#4 ~ ~ ~ ~ ~ ~ ~ e4  ~ ~ ~ ~ ~ ~ ~ " +
  "d#4 ~ ~ ~ ~ ~ ~ ~ c#4 ~ ~ ~ ~ ~ ~ ~ " +
  "b3  ~ ~ ~ ~ ~ ~ ~ e4  ~ ~ ~ ~ ~ ~ ~"
).sound("superpiano").slow(8)
  .gain(0.52).room(0.97)

// 内声: 問いかけと応答
const inner = note("<~ ~ ~ ~ b4 ~ ~ a4 ~ ~ g#4 ~ ~ ~ ~ ~>")
  .sound("superpiano").slow(2)
  .gain(0.3).room(0.98)

// 星のような単音 (14小節ごとに1音、極めて稀)
const stars = note("<~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ b5 ~ ~>")
  .sound("superpiano").slow(4)
  .gain(0.18).room(0.99)

// ─── 重ねる ──────────────────────────────────────────────────
stack(
  earth,
  leaves,
  branch,
  water,
  pulse,
  chords,
  bass,
  melody,
  inner,
  stars
)
