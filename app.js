(function () {
  "use strict";

  const DEG = Math.PI / 180;
  const CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  const CHAIN_PALETTE = [
    0x3498db, 0xe74c3c, 0x2ecc71, 0xf1c40f,
    0x9b59b6, 0x1abc9c, 0xe67e22, 0xecf0f1,
    0x2980b9, 0xc0392b, 0x27ae60, 0xf39c12,
    0x8e44ad, 0x16a085, 0xd35400, 0xbdc3c7
  ];

  function lerpColor(c1, c2, t) {
    const r1 = (c1 >> 16) & 0xff, g1 = (c1 >> 8) & 0xff, b1 = c1 & 0xff;
    const r2 = (c2 >> 16) & 0xff, g2 = (c2 >> 8) & 0xff, b2 = c2 & 0xff;
    const r = Math.round(r1 + (r2 - r1) * t);
    const g = Math.round(g1 + (g2 - g1) * t);
    const b = Math.round(b1 + (b2 - b1) * t);
    return (r << 16) | (g << 8) | b;
  }

  function registerFixedChainTheme(plugin) {
    const provider = {
      name: "fixed-chain",
      label: "Chain Gradient Colors",
      category: "Chain",
      factory: function (ctx, props) {
        return {
          factory: provider.factory,
          granularity: "group",
          color: function (location) {
            // Read chain ID → chain index, B-factor → gradient fraction
            function colorFromUnitElement(unit, elementIdx) {
              const chainIdx = unit.chainIndex[elementIdx];
              const chainId = unit.model.atomicHierarchy.chains.auth_asym_id.value(chainIdx);
              const c = CHAIN_IDS.indexOf(chainId);
              if (c < 0) return 0x888888;
              // Palette bright (0-7) → dark (8-15) pairs
              const bright = CHAIN_PALETTE[c % 8];
              const dark = CHAIN_PALETTE[(c % 8) + 8];
              // Read B-factor for gradient position (0-99 → 0-1)
              var t = 0.5;
              try {
                var atomIdx = unit.elements[elementIdx];
                var bfac = unit.model.atomicConformation.B_iso_or_equiv.value(atomIdx);
                if (Number.isFinite(bfac)) t = bfac / 99;
              } catch (_) {}
              t = Math.max(0, Math.min(1, t));
              return lerpColor(bright, dark, t);
            }

            if (location && location.kind === "element-location") {
              try {
                return colorFromUnitElement(location.unit, location.element);
              } catch (e) { /* fall through */ }
            }
            if (location && location.kind === "bond-location") {
              try {
                return colorFromUnitElement(location.aUnit, location.aIndex);
              } catch (e) { /* fall through */ }
            }
            return 0x888888;
          },
          props: props,
          description: "Per-chain gradient colors (N→C)"
        };
      },
      getParams: function () { return {}; },
      defaultValues: {},
      isApplicable: function () { return true; }
    };
    const reg = plugin.representation.structure.themes.colorThemeRegistry;
    try { reg.add(provider); } catch (e) {
      try { reg.add("fixed-chain", provider); } catch (e2) {
        console.warn("Could not register fixed-chain theme:", e2);
      }
    }
  }

  // ─── MARTINI 2.2 Configuration ───────────────────────────────────

  const SIM = {
    fragmentCount: 6,
    minResidues: 15,
    maxResidues: 25,
    frameMs: 24,
    substepsPerFrame: 4,
    renderIntervalMs: 40,
    minRenderIntervalMs: 20,
    maxRenderIntervalMs: 100,
    totalCycles: 6000,
    initialSpread: 6.0,    // nm
    dt: 0.005,             // ps
    gamma: 1.0,            // /ps
    kB: 0.00831446,        // kJ/(mol·K)
    tempStart: 400,        // K
    tempEnd: 300,           // K
    boxHalf: 8.0,          // nm
    wallK: 500,
    wallShell: 1.0,        // nm
    ljCutoff: 1.2,         // nm
    dielectric: 15,
    neighborCellSize: 1.3, // nm
    bondConstraintIters: 8,
    maxForce: 10000,       // kJ/(mol·nm)
    maxSpeed: 8.0,         // nm/ps
    maxStep: 0.04          // nm
  };

  // ─── MARTINI bead types and sigmas ───────────────────────────────

  // Regular beads: sigma = 0.47 nm, Small (S-prefix) beads: sigma = 0.43 nm
  // Level IX uses sigma = 0.62 nm (cross-size interaction)
  const BEAD_TYPES = [
    "P5","P4","P3","P2","P1",
    "Nda","Nd","Na","N0",
    "C5","C4","C3","C2","C1",
    "Qda","Qd","Qa","Q0",
    "SP5","SP4","SP3","SP2","SP1",
    "SNda","SNd","SNa","SN0",
    "SC5","SC4","SC3","SC2","SC1",
    "SQda","SQd","SQa","SQ0"
  ];
  const BEAD_TYPE_INDEX = {};
  for (let i = 0; i < BEAD_TYPES.length; i++) BEAD_TYPE_INDEX[BEAD_TYPES[i]] = i;

  function beadSigma(type) {
    return type.charAt(0) === "S" ? 0.43 : 0.47;
  }

  // ─── MARTINI interaction matrix (Marrink 2007 Table 2) ──────────
  // Epsilon levels in kJ/mol (× 0.88 for Dry MARTINI)
  const DRY_SCALE = 0.88;
  const EPS_LEVELS = {
    O:   5.6  * DRY_SCALE,
    I:   5.0  * DRY_SCALE,
    II:  4.5  * DRY_SCALE,
    III: 4.0  * DRY_SCALE,
    IV:  3.5  * DRY_SCALE,
    V:   3.1  * DRY_SCALE,
    VI:  2.7  * DRY_SCALE,
    VII: 2.3  * DRY_SCALE,
    VIII:2.0  * DRY_SCALE,
    IX:  2.0  * DRY_SCALE
  };

  // Reduced 9-category interaction mapping: P, Nda, Nd, Na, N0, C, Qda, Qd, Qa, Q0
  // Index: 0=P, 1=Nda, 2=Nd, 3=Na, 4=N0, 5=C, 6=Qda, 7=Qd, 8=Qa, 9=Q0
  // From Marrink 2007 Table 2 (simplified — P types averaged, C types averaged)
  const INTERACTION_LEVELS = [
    //  P     Nda   Nd    Na    N0    C     Qda   Qd    Qa    Q0
    [ "O",   "IV", "IV", "IV", "III","IV", "I",  "I",  "I",  "I"  ],  // P
    [ "IV",  "III","IV", "IV", "III","IV", "III", "III","III","IV" ],  // Nda
    [ "IV",  "IV", "III","IV", "III","IV", "II",  "II", "IV", "IV" ],  // Nd
    [ "IV",  "IV", "IV", "III","III","IV", "II",  "IV", "II", "IV" ],  // Na
    [ "III", "III","III","III","II", "III","IV",  "IV", "IV", "IV" ],  // N0
    [ "IV",  "IV", "IV", "IV", "III","II", "IV",  "IV", "IV", "IV" ],  // C
    [ "I",   "III","II", "II", "IV", "IV", "O",   "I",  "I",  "I"  ],  // Qda
    [ "I",   "III","II", "IV", "IV", "IV", "I",   "O",  "V",  "I"  ],  // Qd
    [ "I",   "III","IV", "II", "IV", "IV", "I",   "V",  "O",  "I"  ],  // Qa
    [ "I",   "IV", "IV", "IV", "IV", "IV", "I",   "I",  "I",  "O"  ]   // Q0
  ];

  function beadCategory(type) {
    const base = type.replace(/^S/, "");
    if (base.charAt(0) === "P") return 0;
    if (base === "Nda") return 1;
    if (base === "Nd") return 2;
    if (base === "Na") return 3;
    if (base === "N0") return 4;
    if (base.charAt(0) === "C") return 5;
    if (base === "Qda") return 6;
    if (base === "Qd") return 7;
    if (base === "Qa") return 8;
    if (base === "Q0") return 9;
    return 4; // default N0
  }

  function pairEpsilon(typeA, typeB) {
    const catA = beadCategory(typeA);
    const catB = beadCategory(typeB);
    const level = INTERACTION_LEVELS[catA][catB];
    return EPS_LEVELS[level];
  }

  function pairSigma(typeA, typeB) {
    return 0.5 * (beadSigma(typeA) + beadSigma(typeB));
  }

  // Charges for charged bead types
  function beadCharge(type) {
    const base = type.replace(/^S/, "");
    if (base === "Qd" || base === "Qda") return 1.0;
    if (base === "Qa") return -1.0;
    return 0;
  }

  // ─── MARTINI residue definitions ─────────────────────────────────
  // BB type is null — resolved at runtime from secondary structure
  // Bond K in kJ/(mol·nm²), angle K in kJ/mol, dihedral K in kJ/mol

  const MARTINI_RESIDUES = {
    GLY: {
      beads: [{ name: "BB", type: null }],
      bonds: [], angles: []
    },
    ALA: {
      beads: [{ name: "BB", type: null }],
      bonds: [], angles: []
    },
    VAL: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "C2" }],
      bonds: [{ a: 0, b: 1, r0: 0.265, K: 7500 }],
      angles: []
    },
    LEU: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "C1" }],
      bonds: [{ a: 0, b: 1, r0: 0.265, K: 7500 }],
      angles: []
    },
    ILE: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "C1" }],
      bonds: [{ a: 0, b: 1, r0: 0.265, K: 7500 }],
      angles: []
    },
    PRO: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "C2" }],
      bonds: [{ a: 0, b: 1, r0: 0.265, K: 7500 }],
      angles: []
    },
    PHE: {
      beads: [
        { name: "BB", type: null },
        { name: "SC1", type: "SC5" },
        { name: "SC2", type: "SC5" },
        { name: "SC3", type: "SC5" }
      ],
      bonds: [
        { a: 0, b: 1, r0: 0.325, K: 7500 },
        { a: 1, b: 2, r0: 0.270, K: 7500, constraint: true },
        { a: 1, b: 3, r0: 0.270, K: 7500, constraint: true },
        { a: 2, b: 3, r0: 0.270, K: 7500, constraint: true }
      ],
      angles: [
        { a: 0, b: 1, c: 2, theta0: 150 * DEG, K: 50 },
        { a: 0, b: 1, c: 3, theta0: 150 * DEG, K: 50 }
      ]
    },
    TYR: {
      beads: [
        { name: "BB", type: null },
        { name: "SC1", type: "SC4" },
        { name: "SC2", type: "SC4" },
        { name: "SC3", type: "SP1" }
      ],
      bonds: [
        { a: 0, b: 1, r0: 0.325, K: 5000 },
        { a: 1, b: 2, r0: 0.270, K: 7500, constraint: true },
        { a: 1, b: 3, r0: 0.270, K: 7500, constraint: true },
        { a: 2, b: 3, r0: 0.270, K: 7500, constraint: true }
      ],
      angles: [
        { a: 0, b: 1, c: 2, theta0: 150 * DEG, K: 50 },
        { a: 0, b: 1, c: 3, theta0: 150 * DEG, K: 50 }
      ]
    },
    TRP: {
      beads: [
        { name: "BB", type: null },
        { name: "SC1", type: "SC4" },
        { name: "SC2", type: "SNd" },
        { name: "SC3", type: "SC5" },
        { name: "SC4", type: "SC5" }
      ],
      bonds: [
        { a: 0, b: 1, r0: 0.300, K: 5000 },
        { a: 1, b: 2, r0: 0.270, K: 7500, constraint: true },
        { a: 2, b: 3, r0: 0.270, K: 7500, constraint: true },
        { a: 3, b: 4, r0: 0.270, K: 7500, constraint: true },
        { a: 1, b: 3, r0: 0.270, K: 7500, constraint: true },
        { a: 2, b: 4, r0: 0.270, K: 7500, constraint: true }
      ],
      angles: [
        { a: 0, b: 1, c: 2, theta0: 210 * DEG, K: 50 },
        { a: 0, b: 1, c: 3, theta0: 90 * DEG, K: 50 }
      ]
    },
    SER: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "P1" }],
      bonds: [{ a: 0, b: 1, r0: 0.250, K: 7500 }],
      angles: []
    },
    THR: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "P1" }],
      bonds: [{ a: 0, b: 1, r0: 0.260, K: 7500 }],
      angles: []
    },
    CYS: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "C5" }],
      bonds: [{ a: 0, b: 1, r0: 0.240, K: 7500 }],
      angles: []
    },
    MET: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "C5" }],
      bonds: [{ a: 0, b: 1, r0: 0.310, K: 2800 }],
      angles: []
    },
    ASP: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "Qa" }],
      bonds: [{ a: 0, b: 1, r0: 0.280, K: 7500 }],
      angles: []
    },
    GLU: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "Qa" }],
      bonds: [{ a: 0, b: 1, r0: 0.310, K: 7500 }],
      angles: []
    },
    ASN: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "P5" }],
      bonds: [{ a: 0, b: 1, r0: 0.280, K: 5000 }],
      angles: []
    },
    GLN: {
      beads: [{ name: "BB", type: null }, { name: "SC1", type: "P4" }],
      bonds: [{ a: 0, b: 1, r0: 0.310, K: 5000 }],
      angles: []
    },
    LYS: {
      beads: [
        { name: "BB", type: null },
        { name: "SC1", type: "C3" },
        { name: "SC2", type: "Qd" }
      ],
      bonds: [
        { a: 0, b: 1, r0: 0.250, K: 7500 },
        { a: 1, b: 2, r0: 0.300, K: 7500 }
      ],
      angles: [
        { a: 0, b: 1, c: 2, theta0: 150 * DEG, K: 20 }
      ]
    },
    ARG: {
      beads: [
        { name: "BB", type: null },
        { name: "SC1", type: "N0" },
        { name: "SC2", type: "Qd" }
      ],
      bonds: [
        { a: 0, b: 1, r0: 0.250, K: 7500 },
        { a: 1, b: 2, r0: 0.350, K: 6200 }
      ],
      angles: [
        { a: 0, b: 1, c: 2, theta0: 150 * DEG, K: 15 }
      ]
    },
    HIS: {
      beads: [
        { name: "BB", type: null },
        { name: "SC1", type: "SC4" },
        { name: "SC2", type: "SP1" },
        { name: "SC3", type: "SP1" }
      ],
      bonds: [
        { a: 0, b: 1, r0: 0.325, K: 7500 },
        { a: 1, b: 2, r0: 0.270, K: 7500, constraint: true },
        { a: 1, b: 3, r0: 0.270, K: 7500, constraint: true },
        { a: 2, b: 3, r0: 0.270, K: 7500, constraint: true }
      ],
      angles: [
        { a: 0, b: 1, c: 2, theta0: 150 * DEG, K: 50 },
        { a: 0, b: 1, c: 3, theta0: 150 * DEG, K: 50 }
      ]
    }
  };

  const ALL_AA = Object.keys(MARTINI_RESIDUES);
  // Weighted AA sampling (roughly reflects natural abundance)
  const AA_WEIGHTS = {
    ALA: 8, ARG: 5, ASN: 4, ASP: 5, CYS: 2, GLN: 4, GLU: 7,
    GLY: 7, HIS: 2, ILE: 5, LEU: 9, LYS: 6, MET: 2, PHE: 4,
    PRO: 5, SER: 7, THR: 5, TRP: 1, TYR: 3, VAL: 7
  };
  const AA_CUMULATIVE = [];
  (function () {
    let sum = 0;
    for (const aa of ALL_AA) {
      sum += (AA_WEIGHTS[aa] || 3);
      AA_CUMULATIVE.push({ aa, cum: sum });
    }
    for (const entry of AA_CUMULATIVE) entry.cum /= sum;
  })();

  // ─── SS-dependent backbone bonded parameters (MARTINI 2.2) ──────
  // BB type by SS: helix = N0, beta = Nda, coil = P5
  const BB_PARAMS = {
    helix: {
      bbType: "N0",
      bond: { r0: 0.310, K: 0, constraint: true },  // constraint
      angle: { theta0: 96 * DEG, K: 700 },
      dihedral: { n: 1, phi0: -120 * DEG, K: 400 }
    },
    beta: {
      bbType: "Nda",
      bond: { r0: 0.350, K: 1250 },
      angle: { theta0: 134 * DEG, K: 25 },
      dihedral: { n: 1, phi0: 120 * DEG, K: 10 }
    },
    coil: {
      bbType: "P5",
      bond: { r0: 0.350, K: 1250 },
      angle: { theta0: 127 * DEG, K: 25 },
      dihedral: null
    }
  };

  // ─── DOM + State ─────────────────────────────────────────────────

  const el = {
    status: document.getElementById("status"),
    toggle: document.getElementById("toggle"),
    reset: document.getElementById("reset")
  };

  function pickAccentColor() {
    const color = CHAIN_PALETTE[Math.floor(Math.random() * CHAIN_PALETTE.length)];
    const r = (color >> 16) & 0xff;
    const g = (color >> 8) & 0xff;
    const b = color & 0xff;
    const hex = "#" + color.toString(16).padStart(6, "0");
    document.documentElement.style.setProperty("--accent", hex);
    document.documentElement.style.setProperty("--accent-rgb", `${r}, ${g}, ${b}`);
  }

  const state = {
    viewer: null,
    fragments: [],
    activeStructures: [],
    activeDataRef: null,
    cycle: 0,
    seed: 0,
    running: true,
    busy: false,
    drawBusy: false,
    drawQueued: false,
    drawCount: 0,
    lastDrawMs: 0,
    dynamicRenderIntervalMs: SIM.renderIntervalMs,
    cps: 0,
    cpsWindowStartMs: 0,
    cpsWindowStartCycle: 0,
    rng: Math.random,
    perf: {
      computeMs: 0,
      drawMs: 0,
      buildPdbMs: 0,
      renderFps: 0
    },
    lastMetrics: {
      totalBeads: 0,
      clashes: 0,
      contacts: 0,
      totalPairs: 1
    }
  };

  function setStatus(message) {
    if (el.status) el.status.textContent = message;
  }

  // ─── RNG + Math Utilities ────────────────────────────────────────

  function createRng(seed) {
    let t = seed >>> 0;
    return function () {
      t += 0x6D2B79F5;
      let x = Math.imul(t ^ (t >>> 15), t | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function setRandomSeed(seed) {
    state.rng = createRng(seed >>> 0);
  }

  function random() {
    return state.rng();
  }

  function rand(min, max) {
    return min + random() * (max - min);
  }

  function randInt(min, max) {
    return Math.floor(rand(min, max + 1));
  }

  function clamp(x, lo, hi) {
    return Math.min(hi, Math.max(lo, x));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function smoothstep(t) {
    const x = clamp(t, 0, 1);
    return x * x * (3 - 2 * x);
  }

  function ewma(prev, sample, alpha) {
    if (prev <= 1e-9) return sample;
    return prev + alpha * (sample - prev);
  }

  function gauss() {
    const u1 = Math.max(1e-12, random());
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function v(x, y, z) {
    return [x, y, z];
  }

  function add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  }

  function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  function scale(a, s) {
    return [a[0] * s, a[1] * s, a[2] * s];
  }

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
  }

  function norm(a) {
    return Math.sqrt(dot(a, a));
  }

  function normalize(a) {
    const n = norm(a);
    return n > 1e-10 ? scale(a, 1 / n) : [1, 0, 0];
  }

  function distanceSq(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  }

  function randomUnitVec() {
    const u = random();
    const z = 2 * random() - 1;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const t = 2 * Math.PI * u;
    return [r * Math.cos(t), r * Math.sin(t), z];
  }

  function progress() {
    return clamp(state.cycle / SIM.totalCycles, 0, 1);
  }

  function currentTemp(p) {
    return lerp(SIM.tempStart, SIM.tempEnd, smoothstep(p));
  }

  function updateCycleRate() {
    const now = performance.now();
    if (state.cpsWindowStartMs <= 0) {
      state.cpsWindowStartMs = now;
      state.cpsWindowStartCycle = state.cycle;
      return;
    }
    const dtMs = now - state.cpsWindowStartMs;
    if (dtMs < 500) return;
    const dCycle = state.cycle - state.cpsWindowStartCycle;
    const raw = dCycle / (dtMs / 1000);
    state.cps = state.cps <= 1e-6 ? raw : lerp(state.cps, raw, 0.35);
    state.cpsWindowStartMs = now;
    state.cpsWindowStartCycle = state.cycle;
  }

  function capVectorInPlace(vec, maxNorm) {
    const m2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    if (m2 <= maxNorm * maxNorm) return;
    const inv = maxNorm / Math.sqrt(m2);
    vec[0] *= inv;
    vec[1] *= inv;
    vec[2] *= inv;
  }

  // ─── Fragment Construction ───────────────────────────────────────

  function pickRandomAA() {
    const r = random();
    for (const entry of AA_CUMULATIVE) {
      if (r <= entry.cum) return entry.aa;
    }
    return AA_CUMULATIVE[AA_CUMULATIVE.length - 1].aa;
  }

  function seedSecondaryTypes(nRes) {
    const ss = new Array(nRes);
    for (let i = 0; i < nRes; i++) ss[i] = "coil";

    const motifs = randInt(2, Math.max(3, Math.floor(nRes / 4)));
    const helixChance = 0.65;
    const betaChance = 0.20;
    for (let m = 0; m < motifs; m++) {
      const toss = random();
      const type = toss < helixChance ? "helix" : toss < helixChance + betaChance ? "beta" : "coil";
      const span = type === "helix" ? randInt(4, 8) : type === "beta" ? randInt(3, 5) : randInt(2, 4);
      const start = randInt(0, Math.max(0, nRes - span));
      for (let i = start; i < Math.min(nRes, start + span); i++) ss[i] = type;
    }
    return ss;
  }

  function resolveBBType(ss) {
    return BB_PARAMS[ss].bbType;
  }

  function makeFragment(id, existingCenters) {
    const nRes = randInt(SIM.minResidues, SIM.maxResidues);
    const chain = CHAIN_IDS[id % CHAIN_IDS.length];

    // Pick residue types
    const resTypes = new Array(nRes);
    for (let i = 0; i < nRes; i++) resTypes[i] = pickRandomAA();

    // Assign secondary structure
    const ss = seedSecondaryTypes(nRes);

    // Place fragment center
    const center = sampleCenter(existingCenters, SIM.initialSpread);

    // Build beads
    const beads = [];
    const bbIdx = new Array(nRes);
    const bonds = [];
    const angles = [];
    const dihedrals = [];
    const exclusionSet = new Set();

    function addExclusion(i, j) {
      const lo = Math.min(i, j);
      const hi = Math.max(i, j);
      exclusionSet.add(lo * 10000 + hi);
    }

    // Place backbone beads along a chain
    for (let r = 0; r < nRes; r++) {
      const bbType = resolveBBType(ss[r]);
      const bbBead = {
        pos: [0, 0, 0], vel: [0, 0, 0], force: [0, 0, 0],
        type: bbType, name: "BB", resIdx: r, isBB: true
      };
      bbIdx[r] = beads.length;
      beads.push(bbBead);

      // Place sidechain beads
      const resDef = MARTINI_RESIDUES[resTypes[r]];
      const scOffset = beads.length - 1; // BB index
      for (let b = 1; b < resDef.beads.length; b++) {
        const def = resDef.beads[b];
        beads.push({
          pos: [0, 0, 0], vel: [0, 0, 0], force: [0, 0, 0],
          type: def.type, name: def.name, resIdx: r, isBB: false
        });
      }

      // Add intra-residue bonds (SC bonds)
      const resBeadStart = bbIdx[r];
      for (const bondDef of resDef.bonds) {
        const bi = resBeadStart + bondDef.a;
        const bj = resBeadStart + bondDef.b;
        bonds.push({
          i: bi, j: bj,
          r0: bondDef.r0, K: bondDef.K,
          isConstraint: !!bondDef.constraint
        });
        addExclusion(bi, bj);
      }

      // Add intra-residue angles
      for (const angDef of resDef.angles) {
        angles.push({
          i: resBeadStart + angDef.a,
          j: resBeadStart + angDef.b,
          k: resBeadStart + angDef.c,
          theta0: angDef.theta0, K: angDef.K
        });
        // Add 1-3 exclusions for angle terms
        addExclusion(resBeadStart + angDef.a, resBeadStart + angDef.c);
      }
    }

    // Add backbone bonds, angles, dihedrals
    for (let r = 0; r < nRes - 1; r++) {
      const bbI = bbIdx[r];
      const bbJ = bbIdx[r + 1];
      // Use SS of first residue for bond params
      const ssType = ss[r];
      const bbp = BB_PARAMS[ssType];
      bonds.push({
        i: bbI, j: bbJ,
        r0: bbp.bond.r0, K: bbp.bond.K,
        isConstraint: !!bbp.bond.constraint
      });
      addExclusion(bbI, bbJ);
    }

    for (let r = 0; r < nRes - 2; r++) {
      const bbI = bbIdx[r];
      const bbJ = bbIdx[r + 1];
      const bbK = bbIdx[r + 2];
      const ssType = ss[r + 1];
      const bbp = BB_PARAMS[ssType];
      angles.push({
        i: bbI, j: bbJ, k: bbK,
        theta0: bbp.angle.theta0, K: bbp.angle.K
      });
      addExclusion(bbI, bbK);
    }

    for (let r = 0; r < nRes - 3; r++) {
      const ssType = ss[r + 1];
      const bbp = BB_PARAMS[ssType];
      if (!bbp.dihedral) continue;
      dihedrals.push({
        i: bbIdx[r], j: bbIdx[r + 1], k: bbIdx[r + 2], l: bbIdx[r + 3],
        n: bbp.dihedral.n, phi0: bbp.dihedral.phi0, K: bbp.dihedral.K
      });
      // 1-4 exclusion
      addExclusion(bbIdx[r], bbIdx[r + 3]);
    }

    // Also exclude BB-SC within same and adjacent residues
    for (let r = 0; r < nRes; r++) {
      const resDef = MARTINI_RESIDUES[resTypes[r]];
      const resStart = bbIdx[r];
      // Exclude all beads within same residue from each other
      for (let a = 0; a < resDef.beads.length; a++) {
        for (let b = a + 1; b < resDef.beads.length; b++) {
          addExclusion(resStart + a, resStart + b);
        }
      }
      // Exclude SC beads with adjacent residue BBs
      if (r > 0) {
        for (let b = 1; b < resDef.beads.length; b++) {
          addExclusion(resStart + b, bbIdx[r - 1]);
        }
      }
      if (r < nRes - 1) {
        for (let b = 1; b < resDef.beads.length; b++) {
          addExclusion(resStart + b, bbIdx[r + 1]);
        }
      }
    }

    // Now position all beads
    // Place backbone as a worm-like chain
    placeBackbone(beads, bbIdx, nRes, ss, center);
    // Place sidechains
    placeSidechains(beads, bbIdx, nRes, resTypes);

    return {
      id, chain, nRes,
      resTypes, ss,
      beads, bbIdx,
      bonds, angles, dihedrals,
      exclusions: exclusionSet
    };
  }

  function sampleCenter(existing, spread) {
    const minSep2 = 3.0 * 3.0; // nm
    let candidate = scale(randomUnitVec(), rand(spread * 0.3, spread));
    for (let attempt = 0; attempt < 80; attempt++) {
      candidate = scale(randomUnitVec(), rand(spread * 0.3, spread));
      let ok = true;
      for (let i = 0; i < existing.length; i++) {
        if (distanceSq(candidate, existing[i]) < minSep2) {
          ok = false;
          break;
        }
      }
      if (ok) return candidate;
    }
    return candidate;
  }

  function placeBackbone(beads, bbIdx, nRes, ss, center) {
    if (nRes === 0) return;

    // Place first BB bead at center
    const p0 = beads[bbIdx[0]].pos;
    p0[0] = center[0]; p0[1] = center[1]; p0[2] = center[2];

    if (nRes === 1) return;

    // Build backbone as a chain
    const dir = randomUnitVec();
    const bbp = BB_PARAMS[ss[0]];
    const p1 = beads[bbIdx[1]].pos;
    p1[0] = p0[0] + dir[0] * bbp.bond.r0;
    p1[1] = p0[1] + dir[1] * bbp.bond.r0;
    p1[2] = p0[2] + dir[2] * bbp.bond.r0;

    for (let r = 2; r < nRes; r++) {
      const ssType = ss[r - 1];
      const params = BB_PARAMS[ssType];
      const bondLen = params.bond.r0;
      const theta = params.angle.theta0;

      const prev2 = beads[bbIdx[r - 2]].pos;
      const prev1 = beads[bbIdx[r - 1]].pos;

      // Place using ideal angle + random dihedral
      const bc = normalize(sub(prev1, prev2));
      let perp = cross(bc, [0, 1, 0]);
      if (norm(perp) < 1e-6) perp = cross(bc, [1, 0, 0]);
      perp = normalize(perp);
      const perp2 = normalize(cross(bc, perp));

      const dih = rand(0, 2 * Math.PI);
      const sinT = Math.sin(Math.PI - theta);
      const cosT = Math.cos(Math.PI - theta);

      const newDir = add(
        scale(bc, cosT),
        add(
          scale(perp, sinT * Math.cos(dih)),
          scale(perp2, sinT * Math.sin(dih))
        )
      );

      const pos = beads[bbIdx[r]].pos;
      pos[0] = prev1[0] + newDir[0] * bondLen;
      pos[1] = prev1[1] + newDir[1] * bondLen;
      pos[2] = prev1[2] + newDir[2] * bondLen;
    }
  }

  function placeSidechains(beads, bbIdx, nRes, resTypes) {
    for (let r = 0; r < nRes; r++) {
      const resDef = MARTINI_RESIDUES[resTypes[r]];
      if (resDef.beads.length <= 1) continue;

      const bbPos = beads[bbIdx[r]].pos;

      // Get backbone tangent
      let tangent;
      if (nRes === 1) {
        tangent = randomUnitVec();
      } else if (r === 0) {
        tangent = normalize(sub(beads[bbIdx[1]].pos, bbPos));
      } else if (r === nRes - 1) {
        tangent = normalize(sub(bbPos, beads[bbIdx[r - 1]].pos));
      } else {
        tangent = normalize(sub(beads[bbIdx[r + 1]].pos, beads[bbIdx[r - 1]].pos));
      }

      // Perpendicular direction, alternating sides
      let perp = cross(tangent, [0, 1, 0]);
      if (norm(perp) < 1e-6) perp = cross(tangent, [1, 0, 0]);
      perp = normalize(perp);
      const flip = (r % 2 === 0) ? 1 : -1;
      perp = scale(perp, flip);

      // Place SC1 at bond length from BB
      const resStart = bbIdx[r];
      if (resDef.beads.length >= 2) {
        const sc1Bond = resDef.bonds.find(function (b) { return b.a === 0 && b.b === 1; });
        const sc1Dist = sc1Bond ? sc1Bond.r0 : 0.30;
        const sc1Pos = beads[resStart + 1].pos;
        sc1Pos[0] = bbPos[0] + perp[0] * sc1Dist;
        sc1Pos[1] = bbPos[1] + perp[1] * sc1Dist;
        sc1Pos[2] = bbPos[2] + perp[2] * sc1Dist;
      }

      // Place subsequent SC beads along the chain
      for (let b = 2; b < resDef.beads.length; b++) {
        // Find bond connecting this bead to a previous one
        const bondDef = resDef.bonds.find(function (bd) { return bd.b === b; });
        if (!bondDef) {
          // No direct bond, try to find one where a < b
          const altBond = resDef.bonds.find(function (bd) { return bd.a === b || bd.b === b; });
          const parentIdx = altBond ? (altBond.a === b ? altBond.b : altBond.a) : b - 1;
          const parentPos = beads[resStart + parentIdx].pos;
          const dist = altBond ? altBond.r0 : 0.27;
          const jitter = randomUnitVec();
          const pos = beads[resStart + b].pos;
          pos[0] = parentPos[0] + jitter[0] * dist;
          pos[1] = parentPos[1] + jitter[1] * dist;
          pos[2] = parentPos[2] + jitter[2] * dist;
        } else {
          const parentPos = beads[resStart + bondDef.a].pos;
          const dist = bondDef.r0;
          // Place roughly continuing from parent direction
          const jitter = normalize(add(perp, scale(randomUnitVec(), 0.5)));
          const pos = beads[resStart + b].pos;
          pos[0] = parentPos[0] + jitter[0] * dist;
          pos[1] = parentPos[1] + jitter[1] * dist;
          pos[2] = parentPos[2] + jitter[2] * dist;
        }
      }
    }
  }

  // ─── Force Computation ───────────────────────────────────────────

  function clearForces() {
    for (const frag of state.fragments) {
      for (let i = 0; i < frag.beads.length; i++) {
        frag.beads[i].force[0] = 0;
        frag.beads[i].force[1] = 0;
        frag.beads[i].force[2] = 0;
      }
    }
  }

  function applySpringBead(beadA, beadB, K, r0) {
    const dx = beadB.pos[0] - beadA.pos[0];
    const dy = beadB.pos[1] - beadA.pos[1];
    const dz = beadB.pos[2] - beadA.pos[2];
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (r < 1e-10) return;
    const dr = r - r0;
    const fmag = K * dr / r;
    beadA.force[0] += fmag * dx;
    beadA.force[1] += fmag * dy;
    beadA.force[2] += fmag * dz;
    beadB.force[0] -= fmag * dx;
    beadB.force[1] -= fmag * dy;
    beadB.force[2] -= fmag * dz;
  }

  function applyAngleBead(beadI, beadJ, beadK, K, theta0) {
    const rij = sub(beadI.pos, beadJ.pos);
    const rkj = sub(beadK.pos, beadJ.pos);
    const rij_len = norm(rij);
    const rkj_len = norm(rkj);
    if (rij_len < 1e-10 || rkj_len < 1e-10) return;

    const cosTheta = clamp(dot(rij, rkj) / (rij_len * rkj_len), -1, 1);
    const theta = Math.acos(cosTheta);
    const dTheta = theta - theta0;
    const sinTheta = Math.sqrt(Math.max(1e-12, 1 - cosTheta * cosTheta));
    if (sinTheta < 1e-10) return;

    const dEdTheta = K * dTheta;
    const st = dEdTheta / sinTheta;
    const inv_rij = 1 / rij_len;
    const inv_rkj = 1 / rkj_len;

    for (let d = 0; d < 3; d++) {
      const fi = st * (rkj[d] * inv_rkj - rij[d] * inv_rij * cosTheta) * inv_rij;
      const fk = st * (rij[d] * inv_rij - rkj[d] * inv_rkj * cosTheta) * inv_rkj;
      beadI.force[d] += fi;
      beadK.force[d] += fk;
      beadJ.force[d] -= fi + fk;
    }
  }

  function applyProperDihedral(beadI, beadJ, beadK, beadL, n, phi0, Kdih) {
    const b1 = sub(beadJ.pos, beadI.pos);
    const b2 = sub(beadK.pos, beadJ.pos);
    const b3 = sub(beadL.pos, beadK.pos);

    const n1 = cross(b1, b2);
    const n2 = cross(b2, b3);
    const n1_len = norm(n1);
    const n2_len = norm(n2);
    if (n1_len < 1e-10 || n2_len < 1e-10) return;

    const b2_len = norm(b2);
    if (b2_len < 1e-10) return;

    const m1 = cross(n1, normalize(b2));
    const cosPhi = dot(n1, n2) / (n1_len * n2_len);
    const sinPhi = dot(m1, n2) / (n1_len * n2_len);
    const phi = Math.atan2(sinPhi, cosPhi);

    // V = K * [1 + cos(n*phi - phi0)]
    // dV/dphi = -K * n * sin(n*phi - phi0)
    const dEdPhi = -Kdih * n * Math.sin(n * phi - phi0);

    const n1_sq = dot(n1, n1);
    const n2_sq = dot(n2, n2);
    if (n1_sq < 1e-14 || n2_sq < 1e-14) return;

    const fi = scale(n1, -dEdPhi * b2_len / n1_sq);
    const fl = scale(n2, dEdPhi * b2_len / n2_sq);

    const dj_b1 = dot(b1, b2) / (b2_len * b2_len);
    const dk_b3 = dot(b3, b2) / (b2_len * b2_len);

    const fj = sub(scale(fi, dj_b1 - 1), scale(fl, dk_b3));
    const fk = sub(scale(fl, dk_b3 - 1), scale(fi, dj_b1));

    for (let d = 0; d < 3; d++) {
      beadI.force[d] += fi[d];
      beadJ.force[d] += fj[d];
      beadK.force[d] += fk[d];
      beadL.force[d] += fl[d];
    }
  }

  // WCA radius: sigma * 2^(1/6) — the LJ minimum
  const TWO_SIXTH = Math.pow(2, 1.0 / 6.0);

  function applyLJ(beadA, beadB, sigma, eps, dVec, r2, repulsiveOnly) {
    const r = Math.sqrt(r2);
    // WCA: truncate at r_min for repulsive-only mode (intramolecular)
    const rMin = sigma * TWO_SIXTH;
    const cutoff = repulsiveOnly ? rMin : SIM.ljCutoff;
    if (r >= cutoff) return;

    // Soft-core: when r < 0.6*sigma, cap force linearly to avoid singularity
    const softR = 0.6 * sigma;
    if (r < softR) {
      // Linear repulsion: force at softR extrapolated linearly
      const srS = sigma / softR;
      const srS6 = srS * srS * srS * srS * srS * srS;
      const srS12 = srS6 * srS6;
      const fAtSoft = 24 * eps * (2 * srS12 - srS6) / (softR * softR);
      const fmag = fAtSoft;  // constant cap
      const invR = r > 1e-10 ? 1 / r : 1 / 1e-10;
      beadA.force[0] -= fmag * dVec[0];
      beadA.force[1] -= fmag * dVec[1];
      beadA.force[2] -= fmag * dVec[2];
      beadB.force[0] += fmag * dVec[0];
      beadB.force[1] += fmag * dVec[1];
      beadB.force[2] += fmag * dVec[2];
      return;
    }

    const invR = 1 / r;
    const sr = sigma * invR;
    const sr6 = sr * sr * sr * sr * sr * sr;
    const sr12 = sr6 * sr6;

    // Force = -dU/dr * (dVec/r) => F_A = -fmag*dVec, F_B = +fmag*dVec
    const fmag = 24 * eps * (2 * sr12 - sr6) * invR * invR;

    beadA.force[0] -= fmag * dVec[0];
    beadA.force[1] -= fmag * dVec[1];
    beadA.force[2] -= fmag * dVec[2];
    beadB.force[0] += fmag * dVec[0];
    beadB.force[1] += fmag * dVec[1];
    beadB.force[2] += fmag * dVec[2];
  }

  function applyCoulomb(beadA, beadB, qA, qB, dVec, r2) {
    const r = Math.sqrt(r2);
    const cutoff = SIM.ljCutoff;
    if (r >= cutoff || r < 1e-10) return;

    // Screened Coulomb: U = f * qA * qB / (eps_r * r)
    // f = 138.935 kJ·nm/(mol·e²) (Coulomb constant)
    const f = 138.935;
    const eps_r = SIM.dielectric;
    const invR = 1 / r;
    const fmag = f * qA * qB / (eps_r * r2);

    beadA.force[0] -= fmag * dVec[0];
    beadA.force[1] -= fmag * dVec[1];
    beadA.force[2] -= fmag * dVec[2];
    beadB.force[0] += fmag * dVec[0];
    beadB.force[1] += fmag * dVec[1];
    beadB.force[2] += fmag * dVec[2];
  }

  function computeForces(p) {
    const t0 = performance.now();
    clearForces();

    // 1. Bonded interactions
    for (const frag of state.fragments) {
      // Springs (harmonic bonds, non-constraint)
      for (const bond of frag.bonds) {
        if (bond.isConstraint) continue;
        applySpringBead(frag.beads[bond.i], frag.beads[bond.j], bond.K, bond.r0);
      }

      // Angle bends
      for (const ang of frag.angles) {
        applyAngleBead(frag.beads[ang.i], frag.beads[ang.j], frag.beads[ang.k], ang.K, ang.theta0);
      }

      // Proper dihedrals
      for (const dih of frag.dihedrals) {
        applyProperDihedral(
          frag.beads[dih.i], frag.beads[dih.j],
          frag.beads[dih.k], frag.beads[dih.l],
          dih.n, dih.phi0, dih.K
        );
      }
    }

    // 2. Non-bonded: spatial hash grid over ALL beads
    const cellSize = SIM.neighborCellSize;
    const invCellSize = 1 / cellSize;
    const ljCut2 = SIM.ljCutoff * SIM.ljCutoff;

    const allBeads = [];
    const grid = new Map();

    for (let fi = 0; fi < state.fragments.length; fi++) {
      const frag = state.fragments[fi];
      for (let bi = 0; bi < frag.beads.length; bi++) {
        const bead = frag.beads[bi];
        const px = bead.pos[0];
        const py = bead.pos[1];
        const pz = bead.pos[2];
        const cx = Math.floor(px * invCellSize);
        const cy = Math.floor(py * invCellSize);
        const cz = Math.floor(pz * invCellSize);
        const entry = { bead, fi, bi, cx, cy, cz };
        const idx = allBeads.length;
        allBeads.push(entry);
        const key = cx * 73856093 ^ cy * 19349669 ^ cz * 83492791;
        let cell = grid.get(key);
        if (!cell) {
          cell = [];
          grid.set(key, cell);
        }
        cell.push(idx);
      }
    }

    let clashes = 0;
    let contacts = 0;
    let pairs = 0;
    const clashCut2 = 0.3 * 0.3; // nm

    for (let ai = 0; ai < allBeads.length; ai++) {
      const a = allBeads[ai];
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const key = (a.cx + dx) * 73856093 ^ (a.cy + dy) * 19349669 ^ (a.cz + dz) * 83492791;
            const cell = grid.get(key);
            if (!cell) continue;
            for (let c = 0; c < cell.length; c++) {
              const bi = cell[c];
              if (bi <= ai) continue;
              const b = allBeads[bi];

              // Skip exclusion pairs
              if (a.fi === b.fi) {
                const frag = state.fragments[a.fi];
                const lo = Math.min(a.bi, b.bi);
                const hi = Math.max(a.bi, b.bi);
                if (frag.exclusions.has(lo * 10000 + hi)) continue;
              }

              const dVecX = b.bead.pos[0] - a.bead.pos[0];
              const dVecY = b.bead.pos[1] - a.bead.pos[1];
              const dVecZ = b.bead.pos[2] - a.bead.pos[2];
              const r2 = dVecX * dVecX + dVecY * dVecY + dVecZ * dVecZ;

              if (r2 >= ljCut2) continue;
              pairs++;

              if (r2 < clashCut2) clashes++;
              if (a.fi !== b.fi && r2 < 0.8 * 0.8) contacts++;

              const dVec = [dVecX, dVecY, dVecZ];
              const sameChain = a.fi === b.fi;

              // LJ: WCA (repulsive-only) for intramolecular, full LJ for inter
              const sigma = pairSigma(a.bead.type, b.bead.type);
              const eps = pairEpsilon(a.bead.type, b.bead.type);
              applyLJ(a.bead, b.bead, sigma, eps, dVec, r2, sameChain);

              // Screened Coulomb for charged pairs
              const qA = beadCharge(a.bead.type);
              const qB = beadCharge(b.bead.type);
              if (qA !== 0 && qB !== 0) {
                applyCoulomb(a.bead, b.bead, qA, qB, dVec, r2);
              }
            }
          }
        }
      }
    }

    // 3. Wall forces (soft boundary)
    const bh = SIM.boxHalf;
    const shell = SIM.wallShell;
    const wk = SIM.wallK;
    for (const frag of state.fragments) {
      for (const bead of frag.beads) {
        for (let d = 0; d < 3; d++) {
          const coord = bead.pos[d];
          if (coord > bh - shell) {
            const pen = coord - (bh - shell);
            bead.force[d] -= wk * pen;
          } else if (coord < -bh + shell) {
            const pen = (-bh + shell) - coord;
            bead.force[d] += wk * pen;
          }
          if (bead.pos[d] > bh) bead.pos[d] = bh;
          if (bead.pos[d] < -bh) bead.pos[d] = -bh;
        }
      }
    }

    // 4. Force capping
    for (const frag of state.fragments) {
      for (const bead of frag.beads) {
        capVectorInPlace(bead.force, SIM.maxForce);
      }
    }

    let totalBeads = 0;
    for (const frag of state.fragments) totalBeads += frag.beads.length;

    state.lastMetrics = {
      totalBeads,
      clashes,
      contacts,
      totalPairs: Math.max(1, pairs)
    };
    state.perf.computeMs = ewma(state.perf.computeMs, performance.now() - t0, 0.25);
  }

  // ─── Integrator ──────────────────────────────────────────────────

  function enforceBondConstraints(iterations) {
    for (let it = 0; it < iterations; it++) {
      for (const frag of state.fragments) {
        for (const bond of frag.bonds) {
          if (!bond.isConstraint) continue;
          const a = frag.beads[bond.i];
          const b = frag.beads[bond.j];
          const dx = b.pos[0] - a.pos[0];
          const dy = b.pos[1] - a.pos[1];
          const dz = b.pos[2] - a.pos[2];
          const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (r < 1e-10) continue;
          const corr = 0.5 * (r - bond.r0) / r;
          const cx = corr * dx;
          const cy = corr * dy;
          const cz = corr * dz;
          a.pos[0] += cx; a.pos[1] += cy; a.pos[2] += cz;
          b.pos[0] -= cx; b.pos[1] -= cy; b.pos[2] -= cz;
        }
      }
    }
  }

  function integrateSubstep(p) {
    const dt = SIM.dt;
    const gamma = SIM.gamma;
    const temp = currentTemp(p);

    const c = Math.exp(-gamma * dt);
    // sigma_noise = sqrt(2 * kB * T * gamma * dt) for forces
    // In BBK-like: noise_vel = sqrt(kB * T * (1 - c^2))
    const sigmaNoise = Math.sqrt(Math.max(0, SIM.kB * temp * (1 - c * c)));

    computeForces(p);

    for (const frag of state.fragments) {
      for (const bead of frag.beads) {
        const f = bead.force;
        const vel = bead.vel;
        const pos = bead.pos;

        // Langevin: v = c*v + dt*F + noise
        vel[0] = c * vel[0] + dt * f[0] + sigmaNoise * gauss();
        vel[1] = c * vel[1] + dt * f[1] + sigmaNoise * gauss();
        vel[2] = c * vel[2] + dt * f[2] + sigmaNoise * gauss();

        capVectorInPlace(vel, SIM.maxSpeed);

        const dx = dt * vel[0];
        const dy = dt * vel[1];
        const dz = dt * vel[2];
        const step2 = dx * dx + dy * dy + dz * dz;
        if (step2 > SIM.maxStep * SIM.maxStep) {
          const s = SIM.maxStep / Math.sqrt(step2);
          pos[0] += dx * s;
          pos[1] += dy * s;
          pos[2] += dz * s;
        } else {
          pos[0] += dx;
          pos[1] += dy;
          pos[2] += dz;
        }

        const bh = SIM.boxHalf;
        for (let d = 0; d < 3; d++) {
          if (pos[d] > bh) { pos[d] = bh; vel[d] = 0; }
          else if (pos[d] < -bh) { pos[d] = -bh; vel[d] = 0; }
        }
      }
    }

    enforceBondConstraints(SIM.bondConstraintIters);
  }

  // ─── PDB Output ──────────────────────────────────────────────────

  // Map CG bead names to pseudo-atom names for PDB rendering
  const BEAD_TO_ATOM = { BB: "CA", SC1: "CB", SC2: "CG", SC3: "CD", SC4: "CE" };
  const BEAD_TO_ELEMENT = { BB: "C", SC1: "C", SC2: "C", SC3: "C", SC4: "C" };

  function atomLine(serial, atomName, residue, chain, resSeq, x, y, z, element, bfactor) {
    const serialText = String(serial).padStart(5, " ");
    const atomText = atomName.length < 4 ? atomName.padStart(4, " ") : atomName.slice(0, 4);
    const resText = residue.padStart(3, " ");
    const seqText = String(resSeq).padStart(4, " ");
    const xx = x.toFixed(3).padStart(8, " ");
    const yy = y.toFixed(3).padStart(8, " ");
    const zz = z.toFixed(3).padStart(8, " ");
    const bf = (bfactor != null ? bfactor : 25).toFixed(2).padStart(6, " ");
    const elText = element.padStart(2, " ");
    return `ATOM  ${serialText} ${atomText} ${resText} ${chain}${seqText}    ${xx}${yy}${zz}  1.00${bf}           ${elText}`;
  }

  function buildPdb() {
    const t0 = performance.now();
    const lines = [];
    let serial = 1;

    // Compute global COM for centering (in nm)
    let sumX = 0, sumY = 0, sumZ = 0, totalBeads = 0;
    for (const frag of state.fragments) {
      for (const bead of frag.beads) {
        sumX += bead.pos[0];
        sumY += bead.pos[1];
        sumZ += bead.pos[2];
        totalBeads++;
      }
    }
    const cx = totalBeads > 0 ? sumX / totalBeads : 0;
    const cy = totalBeads > 0 ? sumY / totalBeads : 0;
    const cz = totalBeads > 0 ? sumZ / totalBeads : 0;

    // Track BB atom serial numbers per chain for CONECT records
    const bbSerials = [];

    for (const frag of state.fragments) {
      const nResMax1 = Math.max(1, frag.nRes - 1);
      const chainBBSerials = [];
      for (const bead of frag.beads) {
        const resSeq = bead.resIdx + 1;  // per-chain, 1-based
        const atomName = BEAD_TO_ATOM[bead.name] || "CA";
        const element = BEAD_TO_ELEMENT[bead.name] || "C";
        const resName = frag.resTypes[bead.resIdx];
        // B-factor encodes gradient position (0 = N-term, 99 = C-term)
        const bfactor = (bead.resIdx / nResMax1) * 99;
        if (bead.isBB) chainBBSerials.push(serial);
        // Convert nm → Å for PDB
        lines.push(atomLine(
          serial, atomName, resName, frag.chain, resSeq,
          (bead.pos[0] - cx) * 10,
          (bead.pos[1] - cy) * 10,
          (bead.pos[2] - cz) * 10,
          element, bfactor
        ));
        serial++;
      }
      bbSerials.push(chainBBSerials);
      const lastResName = frag.resTypes[frag.nRes - 1];
      lines.push(`TER   ${String(serial).padStart(5, " ")}      ${lastResName} ${frag.chain}${String(frag.nRes).padStart(4, " ")}`);
      serial++;
    }

    // CONECT records for backbone connectivity (BB-BB bonds)
    for (const chainBB of bbSerials) {
      for (let i = 0; i < chainBB.length - 1; i++) {
        const s1 = String(chainBB[i]).padStart(5, " ");
        const s2 = String(chainBB[i + 1]).padStart(5, " ");
        lines.push(`CONECT${s1}${s2}`);
      }
    }

    lines.push("END");
    state.perf.buildPdbMs = ewma(state.perf.buildPdbMs, performance.now() - t0, 0.25);
    return lines.join("\n");
  }

  // ─── Rendering ───────────────────────────────────────────────────

  async function draw() {
    const drawStart = performance.now();
    const plugin = state.viewer.plugin;
    const pdb = buildPdb();
    const previousDataRef = state.activeDataRef;

    const dataCell = await plugin.builders.data.rawData(
      { data: pdb, label: `cg-seed-${state.seed}-cycle-${state.cycle}` }
    );
    const trajectory = await plugin.builders.structure.parseTrajectory(dataCell, "pdb");
    await plugin.builders.structure.hierarchy.applyPreset(trajectory, "default");

    const structures = plugin.managers.structure.hierarchy.selection.structures;
    const latest = structures.length ? structures[structures.length - 1] : null;
    const components = latest && Array.isArray(latest.components) ? latest.components : [];

    if (components.length) {
      await plugin.managers.structure.component.removeRepresentations(components);
      for (const comp of components) {
        // Spacefill spheres with per-chain gradient coloring
        await plugin.builders.structure.representation.addRepresentation(comp.cell, {
          type: "spacefill",
          color: "fixed-chain"
        });
        // White backbone cylinders via CONECT bonds
        await plugin.builders.structure.representation.addRepresentation(comp.cell, {
          type: "ball-and-stick",
          color: "uniform",
          colorParams: { value: 0xffffff },
          typeParams: { sizeFactor: 0.24, sizeAspectRatio: 1 }
        });
      }
    }

    if (previousDataRef) {
      await plugin.state.data.build().delete(previousDataRef).commit();
    }
    state.activeDataRef = dataCell.ref;
    state.activeStructures = latest ? [latest] : [];
    state.drawCount += 1;
    const drawEnd = performance.now();
    const drawMs = drawEnd - drawStart;
    state.perf.drawMs = ewma(state.perf.drawMs, drawMs, 0.25);
    if (state.lastDrawMs > 0) {
      const interval = drawEnd - state.lastDrawMs;
      if (interval > 0) state.perf.renderFps = ewma(state.perf.renderFps, 1000 / interval, 0.15);
    }
    if (drawMs > 45) {
      state.dynamicRenderIntervalMs = Math.min(SIM.maxRenderIntervalMs, state.dynamicRenderIntervalMs * 1.15);
    } else if (drawMs < 20) {
      state.dynamicRenderIntervalMs = Math.max(SIM.minRenderIntervalMs, state.dynamicRenderIntervalMs * 0.92);
    }
  }

  async function queueDraw(force) {
    const now = performance.now();
    if (!force && now - state.lastDrawMs < state.dynamicRenderIntervalMs) return;
    if (state.drawBusy) {
      state.drawQueued = true;
      return;
    }

    state.drawBusy = true;
    try {
      do {
        state.drawQueued = false;
        await draw();
        state.lastDrawMs = performance.now();
      } while (state.drawQueued);
    } finally {
      state.drawBusy = false;
    }
  }

  // ─── Visual Style ────────────────────────────────────────────────

  async function applyIllustrativeStyle() {
    const plugin = state.viewer.plugin;
    await plugin.managers.structure.component.setOptions({
      ...plugin.managers.structure.component.state.options,
      ignoreLight: true
    });

    if (!plugin.canvas3d) return;

    const pp = plugin.canvas3d.props.postprocessing;
    plugin.canvas3d.setProps({
      renderer: {
        backgroundColor: 0x101318
      },
      camera: {
        ...plugin.canvas3d.props.camera,
        manualReset: true
      },
      cameraHelper: {
        axes: { name: "off", params: {} }
      },
      trackball: {
        ...plugin.canvas3d.props.trackball,
        rotateSpeed: 0,
        zoomSpeed: 0,
        panSpeed: 0,
        moveSpeed: 0
      },
      transparentBackground: false,
      postprocessing: {
        outline: {
          name: "on",
          params: pp.outline.name === "on"
            ? pp.outline.params
            : { scale: 1, color: 0x000000, threshold: 0.33, includeTransparent: true }
        },
        occlusion: {
          name: "on",
          params: pp.occlusion.name === "on"
            ? pp.occlusion.params
            : {
                multiScale: { name: "off", params: {} },
                radius: 5,
                bias: 0.8,
                blurKernelSize: 15,
                blurDepthBias: 0.5,
                samples: 32,
                resolutionScale: 1,
                color: 0x000000,
                transparentThreshold: 0.4
              }
        },
        shadow: { name: "off", params: {} }
      }
    });
  }

  function enforceNoAutoCenter() {
    const plugin = state.viewer && state.viewer.plugin;
    if (!plugin || !plugin.canvas3d) return;
    plugin.canvas3d.setProps({
      camera: {
        ...plugin.canvas3d.props.camera,
        manualReset: true
      }
    });
  }

  function lockFixedCameraView() {
    const plugin = state.viewer && state.viewer.plugin;
    const canvas3d = plugin && plugin.canvas3d;
    if (!canvas3d || typeof canvas3d.requestCameraReset !== "function") return;

    const dir = normalize([1, 0.22, 1]);
    const up = [0, 1, 0];
    try {
      canvas3d.requestCameraReset({
        durationMs: 0,
        snapshot: (scene, camera) => {
          const sphere = scene && scene.boundingSphereVisible ? scene.boundingSphereVisible : null;
          const center = sphere && sphere.center ? sphere.center : [0, 0, 0];
          const radius = Math.max(18, sphere && Number.isFinite(sphere.radius) ? sphere.radius : SIM.initialSpread * 10);
          if (camera && typeof camera.getInvariantFocus === "function") {
            return camera.getInvariantFocus(center, radius * 0.75, dir, up);
          }
          if (camera && typeof camera.getFocus === "function") {
            return camera.getFocus(center, radius * 0.75);
          }
          return canvas3d.camera.getSnapshot();
        }
      });
    } catch (err) {
      try {
        if (plugin.managers && plugin.managers.camera && typeof plugin.managers.camera.reset === "function") {
          const snapshot = canvas3d.camera.getFocus([0, 0, 0], SIM.initialSpread * 12);
          plugin.managers.camera.reset(snapshot, 0);
        }
      } catch (fallbackErr) {
        console.warn("Camera lock fallback failed:", fallbackErr);
      }
      console.warn("Camera lock failed:", err);
    }
  }

  function toggleIlluminationFromDefault() {
    const plugin = state.viewer.plugin;
    const canvas3d = plugin.canvas3d;
    if (!canvas3d) return;

    const current = !!canvas3d.props.illumination.enabled;
    canvas3d.setProps({
      illumination: {
        ...canvas3d.props.illumination,
        enabled: !current
      }
    });
  }

  function updateHud() {
  }

  // ─── System Setup ────────────────────────────────────────────────

  function reseedSystem(seedOverride) {
    if (Number.isFinite(seedOverride)) state.seed = seedOverride >>> 0;
    else state.seed = (state.seed + 1) >>> 0;
    setRandomSeed(state.seed);

    state.cycle = 0;
    state.cps = 0;
    state.cpsWindowStartMs = 0;
    state.cpsWindowStartCycle = 0;
    state.drawCount = 0;
    state.dynamicRenderIntervalMs = SIM.renderIntervalMs;
    state.activeStructures = [];
    state.activeDataRef = null;
    state.fragments = [];
    state.lastMetrics = {
      totalBeads: 0,
      clashes: 0,
      contacts: 0,
      totalPairs: 1
    };
    pickAccentColor();

    const centers = [];
    for (let i = 0; i < SIM.fragmentCount; i++) {
      const fragment = makeFragment(i, centers);
      const midBB = fragment.beads[fragment.bbIdx[Math.floor(fragment.nRes / 2)]];
      centers.push(midBB.pos.slice());
      state.fragments.push(fragment);
    }

    computeForces(0);
  }

  function integrateFrame() {
    const p = progress();
    for (let i = 0; i < SIM.substepsPerFrame; i++) {
      integrateSubstep(p);
    }
    state.cycle += 1;
  }

  // ─── Debug Interface ─────────────────────────────────────────────

  function runMetrics(options) {
    const opts = options || {};
    const seeds = Array.isArray(opts.seeds) && opts.seeds.length ? opts.seeds : [11, 17, 23, 29, 37];
    const warmupCycles = Math.max(0, opts.warmupCycles | 0);
    const measureCycles = Math.max(1, opts.measureCycles | 0);
    const wasRunning = state.running;

    state.running = false;
    const aggregate = {
      seeds: seeds.length,
      warmupCycles,
      measureCycles,
      meanClashes: 0,
      meanPairs: 0,
      meanContacts: 0,
      meanTotalBeads: 0,
      perSeed: []
    };

    try {
      for (let s = 0; s < seeds.length; s++) {
        const seed = seeds[s] | 0;
        reseedSystem(seed);
        for (let i = 0; i < warmupCycles; i++) integrateFrame();

        let clashesSum = 0;
        let pairsSum = 0;
        let contactsSum = 0;
        let beadsSum = 0;
        for (let i = 0; i < measureCycles; i++) {
          integrateFrame();
          clashesSum += state.lastMetrics.clashes;
          pairsSum += state.lastMetrics.totalPairs;
          contactsSum += state.lastMetrics.contacts;
          beadsSum += state.lastMetrics.totalBeads;
        }

        const result = {
          seed,
          meanClashes: clashesSum / measureCycles,
          meanPairs: pairsSum / measureCycles,
          meanContacts: contactsSum / measureCycles,
          meanTotalBeads: beadsSum / measureCycles
        };
        aggregate.perSeed.push(result);
        aggregate.meanClashes += result.meanClashes;
        aggregate.meanPairs += result.meanPairs;
        aggregate.meanContacts += result.meanContacts;
        aggregate.meanTotalBeads += result.meanTotalBeads;
      }

      const inv = 1 / seeds.length;
      aggregate.meanClashes *= inv;
      aggregate.meanPairs *= inv;
      aggregate.meanContacts *= inv;
      aggregate.meanTotalBeads *= inv;
      return aggregate;
    } finally {
      state.running = wasRunning;
      reseedSystem();
      queueDraw(true).catch(function () {});
    }
  }

  function installDebugInterface() {
    window.__simDebug = {
      runMetrics,
      snapshot: function () {
        let totalBeads = 0;
        for (const frag of state.fragments) totalBeads += frag.beads.length;
        return {
          cycle: state.cycle,
          seed: state.seed,
          running: state.running,
          totalBeads,
          metrics: state.lastMetrics,
          perf: {
            drawMs: state.perf.drawMs,
            computeMs: state.perf.computeMs,
            buildPdbMs: state.perf.buildPdbMs,
            renderFps: state.perf.renderFps
          }
        };
      }
    };
  }

  // ─── Main Loop ───────────────────────────────────────────────────

  async function loop() {
    if (state.busy) {
      window.setTimeout(loop, SIM.frameMs);
      return;
    }

    state.busy = true;
    try {
      if (state.running) {
        integrateFrame();
        updateCycleRate();
        updateHud();
        queueDraw(false).catch((drawErr) => {
          console.error(drawErr);
          setStatus(`Draw failed: ${drawErr && drawErr.message ? drawErr.message : "unknown error"}`);
        });
        const fps = Math.round(state.perf.renderFps);
        setStatus(fps > 0
          ? `Simulating protein dynamics... (${fps} FPS)`
          : "Simulating protein dynamics...");
      }
    } catch (err) {
      console.error(err);
      state.running = false;
      if (el.toggle) el.toggle.textContent = "Resume";
      setStatus(`Simulation halted: ${err && err.message ? err.message : "unknown error"}`);
    } finally {
      state.busy = false;
      window.setTimeout(loop, SIM.frameMs);
    }
  }

  // ─── Initialization ──────────────────────────────────────────────

  async function init() {
    if (!window.molstar || !window.molstar.Viewer) {
      setStatus("Mol* failed to load from CDN.");
      return;
    }

    pickAccentColor();
    setStatus("Initializing...");
    state.viewer = await window.molstar.Viewer.create("app", {
      layoutIsExpanded: true,
      layoutShowControls: false,
      layoutShowRemoteState: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      collapseLeftPanel: true,
      collapseRightPanel: true,
      viewportFocusBehavior: "disabled",
      viewportShowReset: false,
      viewportShowScreenshotControls: false,
      viewportShowExpand: false,
      viewportShowControls: false,
      viewportShowToggleFullscreen: false,
      viewportShowSettings: false,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      viewportShowTrajectoryControls: false,
      viewportBackgroundColor: "0x05070b",
      illumination: true
    });

    registerFixedChainTheme(state.viewer.plugin);
    enforceNoAutoCenter();
    toggleIlluminationFromDefault();
    await applyIllustrativeStyle();
    enforceNoAutoCenter();

    installDebugInterface();
    reseedSystem();
    await queueDraw(true);
    lockFixedCameraView();
    updateHud();

    setStatus("Simulating protein dynamics...");
    loop();
  }

  if (el.toggle) {
    el.toggle.addEventListener("click", function () {
      state.running = !state.running;
      el.toggle.textContent = state.running ? "Pause" : "Resume";
      setStatus(state.running ? "Simulating protein dynamics..." : "Paused.");
    });
  }

  if (el.reset) {
    el.reset.addEventListener("click", async function () {
      if (state.busy) return;
      state.busy = true;
      try {
        setStatus("Reseeding...");
        const oldDataRef = state.activeDataRef;
        reseedSystem();
        if (oldDataRef) {
          await state.viewer.plugin.state.data.build().delete(oldDataRef).commit();
        }
        await queueDraw(true);
        lockFixedCameraView();
        updateHud();
        setStatus("Reseeded.");
      } catch (err) {
        console.error(err);
        setStatus(`Reseed failed: ${err && err.message ? err.message : "unknown error"}`);
      } finally {
        state.busy = false;
      }
    });
  }

  window.addEventListener("keydown", function (evt) {
    if (evt.code === "Space") {
      evt.preventDefault();
      if (el.toggle) el.toggle.click();
    }
    if (evt.key.toLowerCase() === "r") {
      evt.preventDefault();
      if (el.reset) el.reset.click();
    }
  });

  init().catch((err) => {
    console.error(err);
    setStatus(`Initialization error: ${err && err.message ? err.message : "unknown error"}`);
  });
})();
