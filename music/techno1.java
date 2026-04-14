setcps(128/60/4)

const pad = note("<[a3,c4,e4] [f3,a3,c4] [c4,e4,g4] [e3,g3,b3]>")
  .sound("supersaw").slow(2)
  .gain(0.28).room(0.88).lpf(900)
  .delay(0.3).delaytime(0.375)

const bass = note("<a1 f1 c2 e1>").slow(2)
  .sound("sawtooth")
  .gain(0.85).lpf(380).attack(0.008).release(0.2)

const sub = note("<a0 f0 c1 e0>").slow(2)
  .sound("sine").gain(0.55).lpf(100)

const arp = note("a4 c5 e5 g5 a5 g5 e5 c5")
  .sound("sawtooth").gain(0.42).lpf(2600).room(0.35)

const lead = note("<a4 ~ c5 ~ e5 ~ g5 ~> <a5 ~ g5 ~ e5 ~ c5 ~>")
  .sound("sawtooth").slow(4)
  .gain(0.55).lpf(3200).room(0.55)
  .delay(0.18).delaytime(0.375).delayfeedback(0.35)

const counter = note("e4 ~ a4 ~ c5 ~ e5 ~")
  .sound("sine").slow(2)
  .gain(0.22).room(0.88)

const bd   = sound("bd*4").gain(0.88)
const sd   = sound("~ sd ~ sd").gain(0.62)
const hh   = sound("hh*8").gain(0.22)
const oh   = sound("~ ~ ~ oh ~ ~ ~ oh").gain(0.32)
const clap = sound("~ ~ cp ~ ~ ~ cp ~").gain(0.28)

const ki = stack(
  pad.gain(0.22).lpf(650).room(0.96)
     .delay(0.5).delaytime(0.375).delayfeedback(0.45),
  note("<a1 f1 c2 e1>").slow(2)
    .sound("sawtooth").gain(0.1).lpf(180),
  note("~ ~ ~ ~ a5 ~ ~ ~").slow(4)
    .sound("sine").gain(0.08).room(0.99)
)

const sho = stack(
  bd.gain(0.75),
  sd.gain(0.52),
  hh,
  bass.gain(0.68),
  sub.gain(0.42),
  pad.gain(0.26).lpf(1100),
  arp.slow(2).gain(0.28).lpf(2000)
)

const ten = stack(
  bd.gain(0.96),
  sd.gain(0.72),
  hh.gain(0.32),
  oh.gain(0.34),
  clap.gain(0.26),
  bass.gain(0.96),
  sub.gain(0.65),
  pad.gain(0.48).lpf(2200),
  arp.gain(0.52).lpf(3400),
  lead.gain(0.58),
  counter.gain(0.24)
)

const ketsu = stack(
  bd.gain(0.45),
  hh.gain(0.1),
  bass.gain(0.32).lpf(320),
  sub.gain(0.28),
  pad.gain(0.52).lpf(680).room(0.97),
  note("a5 ~ ~ ~ ~ ~ ~ ~").slow(8)
    .sound("sine").gain(0.12).room(0.99)
)

timeCat(
  [16, ki],
  [48, sho],
  [48, ten],
  [16, ketsu]
)
