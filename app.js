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

  function registerFixedChainTheme(plugin) {
    const provider = {
      name: "fixed-chain",
      label: "Fixed Chain Colors",
      category: "Chain",
      factory: function (ctx, props) {
        return {
          factory: provider.factory,
          granularity: "instance",
          color: function (location) {
            if (location && location.kind === "element-location") {
              try {
                const unit = location.unit;
                const chainIdx = unit.chainIndex[location.element];
                const chainId = unit.model.atomicHierarchy.chains.auth_asym_id.value(chainIdx);
                const c = CHAIN_IDS.indexOf(chainId);
                if (c >= 0) return CHAIN_PALETTE[c % CHAIN_PALETTE.length];
              } catch (e) { /* fall through */ }
            }
            if (location && location.kind === "bond-location") {
              try {
                const unit = location.aUnit;
                const chainIdx = unit.chainIndex[location.aIndex];
                const chainId = unit.model.atomicHierarchy.chains.auth_asym_id.value(chainIdx);
                const c = CHAIN_IDS.indexOf(chainId);
                if (c >= 0) return CHAIN_PALETTE[c % CHAIN_PALETTE.length];
              } catch (e) { /* fall through */ }
            }
            return 0x888888;
          },
          props: props,
          description: "Fixed per-chain colors"
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

  const SIM = {
    fragmentCount: 8,
    minResidues: 18,
    maxResidues: 30,
    frameMs: 12,
    substepsPerFrame: 2,
    renderIntervalMs: 80,
    minRenderIntervalMs: 30,
    maxRenderIntervalMs: 150,
    renderAtomMode: "full",
    fullAtomRenderEvery: 8,
    totalCycles: 4000,
    initialSpread: 52,
    finalSpread: 14,
    bondLength: 3.8,
    bondK: 64,
    secK2: 8,
    secK3: 6.5,
    secK4_helix: 4.5,
    secK4_other: 0.4,
    angleK: 10,
    angleTheta0: 104.0 * DEG,
    centerK: 0.014,
    boxHalf: 45,
    wallK: 12,
    wallShell: 4,
    repEpsIntra: 1.8,
    repEpsInter: 4.1,
    repSigmaScaleIntra: 1.05,
    repSigmaScaleInter: 1.2,
    stericMinScale: 0.98,
    attrSigma: 7.6,
    attrEpsStart: 0.05,
    attrEpsEnd: 0.45,
    attrCut: 14.5,
    nonbondStrideIntra: 2,
    nonbondStrideInter: 1,
    neighborListEnabled: true,
    neighborCellSize: 18,
    clashDistance: 3.0,
    dt: 0.02,
    gamma: 0.75,
    tempStart: 8.2,
    tempEnd: 3.2,
    maxForce: 85,
    compactK: 0.04,
    compactHydroMin: 0.3,
    maxSpeed: 12,
    maxStep: 0.45,
    bondConstraintIters: 6
  };

  const SECONDARY_TARGETS = {
    helix: { d2: 5.45, d3: 5.2, d4: 6.15, dihK: 4.6, dihPhi0: 50 * DEG },
    beta: { d2: 6.2, d3: 8.2, d4: 10.0, dihK: 1.2, dihPhi0: -162 * DEG },
    coil: { d2: 6.2, d3: 8.3, d4: 10.5, dihK: 0.35, dihPhi0: 0 }
  };

  const RESIDUE_LIBRARY = [
    { code: "G", name: "GLY", size: 3.2, hydro: 0.36, charge: 0 },
    { code: "A", name: "ALA", size: 3.35, hydro: 0.62, charge: 0 },
    { code: "V", name: "VAL", size: 3.65, hydro: 0.86, charge: 0 },
    { code: "L", name: "LEU", size: 3.75, hydro: 0.9, charge: 0 },
    { code: "I", name: "ILE", size: 3.75, hydro: 0.92, charge: 0 },
    { code: "S", name: "SER", size: 3.35, hydro: 0.28, charge: 0 },
    { code: "T", name: "THR", size: 3.45, hydro: 0.34, charge: 0 },
    { code: "D", name: "ASP", size: 3.5, hydro: 0.12, charge: -1 },
    { code: "E", name: "GLU", size: 3.6, hydro: 0.14, charge: -1 },
    { code: "K", name: "LYS", size: 3.7, hydro: 0.2, charge: 1 },
    { code: "R", name: "ARG", size: 3.8, hydro: 0.18, charge: 1 }
  ];

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
    pairWindowStartMs: 0,
    pairWindowCount: 0,
    pairChecksPerSec: 0,
    rng: Math.random,
    perf: {
      computeMs: 0,
      drawMs: 0,
      buildPdbMs: 0
    },
    lastMetrics: {
      clashes: 0,
      contacts: 0,
      totalPairs: 1,
      clashRate: 0,
      meanEndToEndNorm: 0
    }
  };

  function setStatus(message) {
    if (el.status) el.status.textContent = message;
  }

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

  function updatePairRate(pairChecks) {
    const now = performance.now();
    if (state.pairWindowStartMs <= 0) {
      state.pairWindowStartMs = now;
      state.pairWindowCount = pairChecks;
      return;
    }
    state.pairWindowCount += pairChecks;
    const dtMs = now - state.pairWindowStartMs;
    if (dtMs < 500) return;
    const rate = state.pairWindowCount / (dtMs / 1000);
    state.pairChecksPerSec = ewma(state.pairChecksPerSec, rate, 0.35);
    state.pairWindowStartMs = now;
    state.pairWindowCount = 0;
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

  function randomFrame() {
    const x = randomUnitVec();
    let y = randomUnitVec();
    y = normalize(sub(y, scale(x, dot(y, x))));
    if (norm(y) < 1e-10) y = normalize(cross(x, [0, 1, 0]));
    const z = normalize(cross(x, y));
    return { x, y, z };
  }

  function placeFromThreePoints(a, b, c, length, angleRad, dihedralRad) {
    const bc = normalize(sub(c, b));
    const ba = normalize(sub(a, b));
    let n = normalize(cross(ba, bc));
    if (norm(n) < 1e-10) n = [0, 0, 1];
    const m = cross(n, bc);

    const t1 = scale(bc, -Math.cos(angleRad) * length);
    const t2 = scale(m, Math.sin(angleRad) * Math.cos(dihedralRad) * length);
    const t3 = scale(n, Math.sin(angleRad) * Math.sin(dihedralRad) * length);
    return add(c, add(t1, add(t2, t3)));
  }

  function progress() {
    return clamp(state.cycle / SIM.totalCycles, 0, 1);
  }

  function currentTemp(p) {
    return lerp(SIM.tempStart, SIM.tempEnd, smoothstep(p));
  }

  function currentAttraction(p) {
    return lerp(SIM.attrEpsStart, SIM.attrEpsEnd, smoothstep(p));
  }

  function targetRadius(p) {
    return lerp(SIM.initialSpread, SIM.finalSpread, smoothstep(p));
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

  function seedSecondaryTypes(fragment, p) {
    for (let i = 0; i < fragment.length; i++) fragment.ss[i] = "coil";

    const motifs = randInt(2, Math.max(3, Math.floor(fragment.length / 4)));
    const helixChance = 0.72 + 0.18 * p;
    const betaChance = 0.12 * (1 - 0.4 * p);
    for (let m = 0; m < motifs; m++) {
      const toss = random();
      const type = toss < helixChance ? "helix" : toss < helixChance + betaChance ? "beta" : "coil";
      const span = type === "helix" ? randInt(4, 8) : type === "beta" ? randInt(3, 5) : randInt(2, 4);
      const start = randInt(0, Math.max(0, fragment.length - span));
      for (let i = start; i < start + span; i++) fragment.ss[i] = type;
    }
  }

  function initialDihedralForSs(ssType) {
    if (ssType === "helix") return rand(46, 54) * DEG;
    if (ssType === "beta") return (random() < 0.5 ? -1 : 1) * rand(150, 168) * DEG;
    return rand(-115, 115) * DEG;
  }

  function enforceChainBondLengths(ca, bond, iterations) {
    for (let it = 0; it < iterations; it++) {
      for (let i = 0; i < ca.length - 1; i++) {
        const p0 = ca[i];
        const p1 = ca[i + 1];
        const dx = p1[0] - p0[0];
        const dy = p1[1] - p0[1];
        const dz = p1[2] - p0[2];
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-8) continue;
        const corr = 0.5 * (r - bond) / r;
        const cx = corr * dx;
        const cy = corr * dy;
        const cz = corr * dz;
        p0[0] += cx; p0[1] += cy; p0[2] += cz;
        p1[0] -= cx; p1[1] -= cy; p1[2] -= cz;
      }
    }
  }

  function compactifyInitialChain(ca, center, targetRadius) {
    const minDist = 2.95;
    const minDist2 = minDist * minDist;
    for (let it = 0; it < 4; it++) {
      let cmx = 0, cmy = 0, cmz = 0;
      for (let i = 0; i < ca.length; i++) {
        cmx += ca[i][0];
        cmy += ca[i][1];
        cmz += ca[i][2];
      }
      const inv = 1 / ca.length;
      cmx *= inv; cmy *= inv; cmz *= inv;

      for (let i = 0; i < ca.length; i++) {
        const p = ca[i];
        const toCenterX = center[0] - p[0];
        const toCenterY = center[1] - p[1];
        const toCenterZ = center[2] - p[2];
        const toCmX = cmx - p[0];
        const toCmY = cmy - p[1];
        const toCmZ = cmz - p[2];
        p[0] += 0.025 * toCenterX + 0.012 * toCmX + rand(-0.14, 0.14);
        p[1] += 0.025 * toCenterY + 0.012 * toCmY + rand(-0.14, 0.14);
        p[2] += 0.025 * toCenterZ + 0.012 * toCmZ + rand(-0.14, 0.14);

        const rx = p[0] - center[0];
        const ry = p[1] - center[1];
        const rz = p[2] - center[2];
        const r = Math.sqrt(rx * rx + ry * ry + rz * rz);
        if (r > targetRadius && r > 1e-8) {
          const s = targetRadius / r;
          p[0] = center[0] + rx * s;
          p[1] = center[1] + ry * s;
          p[2] = center[2] + rz * s;
        }
      }

      for (let i = 0; i < ca.length; i++) {
        for (let j = i + 3; j < ca.length; j++) {
          const dx = ca[j][0] - ca[i][0];
          const dy = ca[j][1] - ca[i][1];
          const dz = ca[j][2] - ca[i][2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 >= minDist2 || d2 < 1e-12) continue;
          const d = Math.sqrt(d2);
          const push = 0.5 * (minDist - d) / d;
          const px = push * dx;
          const py = push * dy;
          const pz = push * dz;
          ca[i][0] -= px; ca[i][1] -= py; ca[i][2] -= pz;
          ca[j][0] += px; ca[j][1] += py; ca[j][2] += pz;
        }
      }

      enforceChainBondLengths(ca, SIM.bondLength, 2);
    }
  }

  function buildInitialCaChain(length, center, ss) {
    const ca = new Array(length);
    const frame = randomFrame();
    const bond = SIM.bondLength;
    const theta = SIM.angleTheta0;
    const seedRadius = Math.max(13.0, Math.min(20.0, 0.34 * length * bond));

    ca[0] = add(center, scale(frame.x, rand(-1.2, 1.2)));
    if (length === 1) return ca;

    ca[1] = add(ca[0], scale(normalize(add(frame.x, scale(frame.y, rand(-0.25, 0.25)))), bond));
    if (length === 2) return ca;

    const dir2 = normalize(add(scale(frame.x, -Math.cos(theta)), scale(frame.y, Math.sin(theta) * rand(0.8, 1.2))));
    ca[2] = add(ca[1], scale(dir2, bond));

    const localSteric2 = Math.pow(Math.max(2.8, SIM.bondLength * 0.8), 2);
    for (let i = 3; i < length; i++) {
      const mode = ss[i - 1];
      let placed = null;
      for (let attempt = 0; attempt < 24; attempt++) {
        const tau = initialDihedralForSs(mode) + rand(-24, 24) * DEG;
        const candidate = placeFromThreePoints(ca[i - 3], ca[i - 2], ca[i - 1], bond, theta, tau);
        let ok = true;
        for (let j = 0; j < i - 2; j++) {
          if (distanceSq(candidate, ca[j]) < localSteric2) {
            ok = false;
            break;
          }
        }
        if (!ok) continue;
        const radial = norm(sub(candidate, center));
        if (radial > seedRadius) continue;
        placed = candidate;
        break;
      }
      if (!placed) {
        const last = ca[i - 1];
        const dir = normalize(add(randomUnitVec(), scale(normalize(sub(center, last)), 1.2)));
        placed = add(last, scale(dir, bond));
        const radial = norm(sub(placed, center));
        if (radial > seedRadius) {
          const inward = normalize(sub(center, last));
          placed = add(last, scale(inward, bond));
        }
      }
      ca[i] = placed;
    }

    compactifyInitialChain(ca, center, seedRadius);
    return ca;
  }

  function sampleCenter(existing, spread) {
    const minSep2 = 17 * 17;
    let candidate = scale(randomUnitVec(), rand(spread * 0.55, spread));
    for (let attempt = 0; attempt < 80; attempt++) {
      candidate = scale(randomUnitVec(), rand(spread * 0.55, spread));
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

  function makeFragment(i, p, existingCenters) {
    const length = randInt(SIM.minResidues, SIM.maxResidues);
    const center = sampleCenter(existingCenters, SIM.initialSpread);

    const ca = new Array(length);
    const vel = new Array(length);
    const forces = new Array(length);
    const ss = new Array(length);
    const residues = new Array(length);

    const fragment = {
      id: i,
      chain: CHAIN_IDS[i % CHAIN_IDS.length],
      length,
      ca,
      vel,
      forces,
      ss,
      residues
    };
    seedSecondaryTypes(fragment, p);

    const initialCa = buildInitialCaChain(length, center, ss);
    for (let r = 0; r < length; r++) {
      ca[r] = initialCa[r];
      vel[r] = [0, 0, 0];
      forces[r] = [0, 0, 0];
    }

    for (let r = 0; r < length; r++) {
      residues[r] = RESIDUE_LIBRARY[randInt(0, RESIDUE_LIBRARY.length - 1)];
    }
    return fragment;
  }

  function clearForces() {
    for (const frag of state.fragments) {
      for (let i = 0; i < frag.length; i++) {
        frag.forces[i][0] = 0;
        frag.forces[i][1] = 0;
        frag.forces[i][2] = 0;
      }
    }
  }

  function addForce(frag, idx, fx, fy, fz) {
    frag.forces[idx][0] += fx;
    frag.forces[idx][1] += fy;
    frag.forces[idx][2] += fz;
  }

  function capVectorInPlace(vec, maxNorm) {
    const m2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    if (m2 <= maxNorm * maxNorm) return;
    const inv = maxNorm / Math.sqrt(m2);
    vec[0] *= inv;
    vec[1] *= inv;
    vec[2] *= inv;
  }

  function applySpring(fragA, i, fragB, j, k, target) {
    const pa = fragA.ca[i];
    const pb = fragB.ca[j];
    const d = sub(pb, pa);
    const r = norm(d);
    if (r < 1e-8) return 0;
    const dr = r - target;
    const fmag = -k * dr / r;
    const fx = fmag * d[0];
    const fy = fmag * d[1];
    const fz = fmag * d[2];

    addForce(fragA, i, fx, fy, fz);
    addForce(fragB, j, -fx, -fy, -fz);
    return 0.5 * k * dr * dr;
  }

  function applyAngleBend(frag, i, j, k, K, theta0) {
    const rij = sub(frag.ca[i], frag.ca[j]);
    const rkj = sub(frag.ca[k], frag.ca[j]);
    const rij_len = norm(rij);
    const rkj_len = norm(rkj);
    if (rij_len < 1e-8 || rkj_len < 1e-8) return 0;

    const cosTheta = clamp(dot(rij, rkj) / (rij_len * rkj_len), -1, 1);
    const theta = Math.acos(cosTheta);
    const dTheta = theta - theta0;
    const sinTheta = Math.sqrt(Math.max(1e-12, 1 - cosTheta * cosTheta));

    const dEdTheta = K * dTheta;
    const st = dEdTheta / sinTheta;

    const inv_rij = 1 / rij_len;
    const inv_rkj = 1 / rkj_len;

    for (let d = 0; d < 3; d++) {
      const fi = st * (rkj[d] * inv_rkj - rij[d] * inv_rij * cosTheta) * inv_rij;
      const fk = st * (rij[d] * inv_rij - rkj[d] * inv_rkj * cosTheta) * inv_rkj;
      frag.forces[i][d] += fi;
      frag.forces[k][d] += fk;
      frag.forces[j][d] -= fi + fk;
    }

    return 0.5 * K * dTheta * dTheta;
  }

  function applyDihedral(frag, i, j, k, l, Kdih, phi0) {
    const b1 = sub(frag.ca[j], frag.ca[i]);
    const b2 = sub(frag.ca[k], frag.ca[j]);
    const b3 = sub(frag.ca[l], frag.ca[k]);

    const n1 = cross(b1, b2);
    const n2 = cross(b2, b3);
    const n1_len = norm(n1);
    const n2_len = norm(n2);
    if (n1_len < 1e-8 || n2_len < 1e-8) return 0;

    const b2_len = norm(b2);
    if (b2_len < 1e-8) return 0;

    const m1 = cross(n1, normalize(b2));
    const cosPhi = dot(n1, n2) / (n1_len * n2_len);
    const sinPhi = dot(m1, n2) / (n1_len * n2_len);
    const phi = Math.atan2(sinPhi, cosPhi);

    const dEdPhi = Kdih * Math.sin(phi - phi0);

    const n1_sq = dot(n1, n1);
    const n2_sq = dot(n2, n2);
    if (n1_sq < 1e-12 || n2_sq < 1e-12) return 0;

    const fi = scale(n1, -dEdPhi * b2_len / n1_sq);
    const fl = scale(n2, dEdPhi * b2_len / n2_sq);

    const dj_b1 = dot(b1, b2) / (b2_len * b2_len);
    const dk_b3 = dot(b3, b2) / (b2_len * b2_len);

    const fj = sub(scale(fi, dj_b1 - 1), scale(fl, dk_b3));
    const fk = sub(scale(fl, dk_b3 - 1), scale(fi, dj_b1));

    for (let d = 0; d < 3; d++) {
      frag.forces[i][d] += fi[d];
      frag.forces[j][d] += fj[d];
      frag.forces[k][d] += fk[d];
      frag.forces[l][d] += fl[d];
    }

    return Kdih * (1 - Math.cos(phi - phi0));
  }

  function applyRepulsiveLJ(fragA, i, fragB, j, d, r2, sigma, eps) {
    if (r2 < 1e-12) return 0;

    const sigma2 = sigma * sigma;
    const rc = sigma * Math.pow(2, 1 / 6);
    const rc2 = rc * rc;
    if (r2 >= rc2) return 0;

    const invR2 = 1 / r2;
    const sr2 = sigma2 * invR2;
    const sr6 = sr2 * sr2 * sr2;
    const sr12 = sr6 * sr6;
    const fmag = (24 * eps * (2 * sr12 - sr6)) * invR2;

    const fx = fmag * d[0];
    const fy = fmag * d[1];
    const fz = fmag * d[2];
    addForce(fragA, i, -fx, -fy, -fz);
    addForce(fragB, j, fx, fy, fz);

    return 4 * eps * (sr12 - sr6) + eps;
  }

  function applyAttractiveLJ(fragA, i, fragB, j, d, r2, sigma, eps, cut) {
    if (eps <= 0) return 0;
    if (r2 < 1e-12 || r2 > cut * cut) return 0;

    const r = Math.sqrt(r2);
    const wellCenter = sigma;
    const wellWidth = 2.6;
    const dr = r - wellCenter;
    const g = Math.exp(-(dr * dr) / (2 * wellWidth * wellWidth));
    const dUdr = eps * g * (dr / (wellWidth * wellWidth));
    const fmag = -dUdr / Math.max(1e-8, r);

    const fx = fmag * d[0];
    const fy = fmag * d[1];
    const fz = fmag * d[2];
    addForce(fragA, i, -fx, -fy, -fz);
    addForce(fragB, j, fx, fy, fz);

    return -eps * g;
  }

  function computeMeanEndToEndNorm() {
    if (!state.fragments.length) return 0;
    let sum = 0;
    for (const frag of state.fragments) {
      if (frag.length < 2) continue;
      const first = frag.ca[0];
      const last = frag.ca[frag.length - 1];
      const r = Math.sqrt(distanceSq(last, first));
      const maxLen = SIM.bondLength * (frag.length - 1);
      sum += maxLen > 1e-8 ? r / maxLen : 0;
    }
    return sum / state.fragments.length;
  }

  function computeForces(p) {
    const t0 = performance.now();
    clearForces();

    const secBias = 1.0;
    const attr = currentAttraction(p);
    const radiusTarget = targetRadius(p);
    let potential = 0;

    for (const frag of state.fragments) {
      for (let i = 0; i < frag.length - 1; i++) {
        potential += applySpring(frag, i, frag, i + 1, SIM.bondK, SIM.bondLength);
      }

      for (let i = 0; i < frag.length; i++) {
        const pi = frag.ca[i];
        const r = norm(pi);
        if (r > 1e-10) {
          const dr = r - radiusTarget;
          const fmag = -SIM.centerK * dr / r;
          addForce(frag, i, fmag * pi[0], fmag * pi[1], fmag * pi[2]);
          potential += 0.5 * SIM.centerK * dr * dr;
        }

        const bh = SIM.boxHalf;
        const shell = SIM.wallShell;
        const wk = SIM.wallK;
        for (let d = 0; d < 3; d++) {
          const coord = pi[d];
          if (coord > bh - shell) {
            const pen = coord - (bh - shell);
            frag.forces[i][d] -= wk * pen;
            potential += 0.5 * wk * pen * pen;
          } else if (coord < -bh + shell) {
            const pen = (-bh + shell) - coord;
            frag.forces[i][d] += wk * pen;
            potential += 0.5 * wk * pen * pen;
          }
          if (pi[d] > bh) pi[d] = bh;
          if (pi[d] < -bh) pi[d] = -bh;
        }
      }

      for (let i = 0; i < frag.length - 2; i++) {
        const target2 = SECONDARY_TARGETS[frag.ss[i]].d2;
        potential += applySpring(frag, i, frag, i + 2, SIM.secK2 * secBias, target2);
      }

      for (let i = 0; i < frag.length - 3; i++) {
        const target3 = SECONDARY_TARGETS[frag.ss[i]].d3;
        potential += applySpring(frag, i, frag, i + 3, SIM.secK3 * secBias, target3);
      }

      for (let i = 0; i < frag.length - 2; i++) {
        potential += applyAngleBend(frag, i, i + 1, i + 2, SIM.angleK, SIM.angleTheta0);
      }

      for (let i = 0; i < frag.length - 3; i++) {
        const ss = frag.ss[i];
        const targets = SECONDARY_TARGETS[ss];
        potential += applyDihedral(frag, i, i + 1, i + 2, i + 3, targets.dihK, targets.dihPhi0);
      }

      for (let i = 0; i < frag.length - 4; i++) {
        const ss = frag.ss[i];
        const k4 = ss === "helix" ? SIM.secK4_helix : SIM.secK4_other;
        const d4 = SECONDARY_TARGETS[ss].d4;
        potential += applySpring(frag, i, frag, i + 4, k4, d4);
      }

      let cmx = 0, cmy = 0, cmz = 0;
      for (let i = 0; i < frag.length; i++) {
        cmx += frag.ca[i][0];
        cmy += frag.ca[i][1];
        cmz += frag.ca[i][2];
      }
      const invN = 1 / frag.length;
      cmx *= invN; cmy *= invN; cmz *= invN;

      const kComp = SIM.compactK * (0.65 + 0.75 * smoothstep(p));
      for (let i = 0; i < frag.length; i++) {
        const hydro = frag.residues[i].hydro;
        const w = SIM.compactHydroMin + (1 - SIM.compactHydroMin) * hydro;
        const dx = frag.ca[i][0] - cmx;
        const dy = frag.ca[i][1] - cmy;
        const dz = frag.ca[i][2] - cmz;
        const fx = -kComp * w * dx;
        const fy = -kComp * w * dy;
        const fz = -kComp * w * dz;
        addForce(frag, i, fx, fy, fz);
        potential += 0.5 * kComp * w * (dx * dx + dy * dy + dz * dz);
      }
    }

    const cellSize = SIM.neighborCellSize;
    const invCellSize = 1 / cellSize;
    const beads = [];
    const grid = new Map();
    const strideIntra = Math.max(1, SIM.nonbondStrideIntra | 0);
    const strideInter = Math.max(1, SIM.nonbondStrideInter | 0);
    const useFullInter = strideInter <= 1;
    const attrCut2 = SIM.attrCut * SIM.attrCut;
    const clashCut2 = SIM.clashDistance * SIM.clashDistance;

    for (let fi = 0; fi < state.fragments.length; fi++) {
      const frag = state.fragments[fi];
      for (let i = 0; i < frag.length; i++) {
        const p0 = frag.ca[i];
        const cx = Math.floor(p0[0] * invCellSize);
        const cy = Math.floor(p0[1] * invCellSize);
        const cz = Math.floor(p0[2] * invCellSize);
        const bead = { frag, fi, i, p: p0, residue: frag.residues[i], cx, cy, cz };
        const index = beads.length;
        beads.push(bead);
        const key = `${cx}|${cy}|${cz}`;
        let cell = grid.get(key);
        if (!cell) {
          cell = [];
          grid.set(key, cell);
        }
        cell.push(index);
      }
    }

    let clashes = 0;
    let contacts = 0;
    let pairs = 0;
    let pairComputed = 0;

    for (let ai = 0; ai < beads.length; ai++) {
      const a = beads[ai];
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const key = `${a.cx + dx}|${a.cy + dy}|${a.cz + dz}`;
            const cell = grid.get(key);
            if (!cell) continue;
            for (let c = 0; c < cell.length; c++) {
              const bi = cell[c];
              if (bi <= ai) continue;
              const b = beads[bi];
              const same = a.fi === b.fi;
              if (same && Math.abs(a.i - b.i) < 3) continue;
              if (same && ((a.i % strideIntra) !== 0 || (b.i % strideIntra) !== 0)) continue;
              if (!same && !useFullInter && ((a.i % strideInter) !== 0 || (b.i % strideInter) !== 0)) continue;
              pairs += 1;

              const dVec = sub(b.p, a.p);
              const d2 = dot(dVec, dVec);
              if (d2 < clashCut2) clashes += 1;
              if (!same && d2 < 100) contacts += 1;

              const sigmaPair = 0.5 * (a.residue.size + b.residue.size);
              const hydroPair = 0.5 * (a.residue.hydro + b.residue.hydro);
              const qq = a.residue.charge * b.residue.charge;
              const electroBias = qq < 0 ? 1.25 : qq > 0 ? 0.82 : 1.0;
              const sigmaScale = same ? SIM.repSigmaScaleIntra : SIM.repSigmaScaleInter;
              const repEps = same ? SIM.repEpsIntra : SIM.repEpsInter;

              const repSigma = Math.max(2.8, sigmaPair * sigmaScale);
              const stericFloor = SIM.stericMinScale * repSigma;
              const stericFloor2 = stericFloor * stericFloor;
              if (d2 < stericFloor2) {
                const dnorm = Math.sqrt(Math.max(1e-12, d2));
                const push = 0.5 * repEps * (stericFloor - dnorm) / dnorm;
                const px = push * dVec[0];
                const py = push * dVec[1];
                const pz = push * dVec[2];
                addForce(a.frag, a.i, -px, -py, -pz);
                addForce(b.frag, b.i, px, py, pz);
                potential += 0.5 * repEps * (stericFloor - dnorm) * (stericFloor - dnorm);
              }
              const repCut = repSigma * Math.pow(2, 1 / 6);
              if (d2 < repCut * repCut) {
                potential += applyRepulsiveLJ(a.frag, a.i, b.frag, b.i, dVec, d2, repSigma, repEps);
                pairComputed += 1;
              }
              if (!same || Math.abs(a.i - b.i) > 4) {
                const seqSep = Math.abs(a.i - b.i);
                const sameChainBias = same ? (seqSep > 8 ? 1.06 : 1.0) : 0.9;
                const pairAttr = attr * (0.12 + 0.45 * hydroPair) * electroBias * sameChainBias;
                if (d2 < attrCut2) {
                  potential += applyAttractiveLJ(a.frag, a.i, b.frag, b.i, dVec, d2, SIM.attrSigma, pairAttr, SIM.attrCut);
                  pairComputed += 1;
                }
              }
            }
          }
        }
      }
    }

    for (const frag of state.fragments) {
      for (let i = 0; i < frag.length; i++) {
        capVectorInPlace(frag.forces[i], SIM.maxForce);
      }
    }

    updatePairRate(pairComputed);
    state.lastMetrics = {
      clashes,
      contacts,
      totalPairs: Math.max(1, pairs),
      clashRate: clashes / Math.max(1, pairs),
      meanEndToEndNorm: computeMeanEndToEndNorm()
    };
    state.perf.computeMs = ewma(state.perf.computeMs, performance.now() - t0, 0.25);

    return potential;
  }

  function enforceBondConstraints(iterations) {
    for (let it = 0; it < iterations; it++) {
      for (const frag of state.fragments) {
        for (let i = 0; i < frag.length - 1; i++) {
          const p0 = frag.ca[i];
          const p1 = frag.ca[i + 1];
          const d = sub(p1, p0);
          const r = norm(d);
          if (r < 1e-8) continue;
          const corr = 0.5 * (r - SIM.bondLength) / r;
          const dx = corr * d[0];
          const dy = corr * d[1];
          const dz = corr * d[2];
          p0[0] += dx;
          p0[1] += dy;
          p0[2] += dz;
          p1[0] -= dx;
          p1[1] -= dy;
          p1[2] -= dz;
        }
      }
    }
  }

  function integrateSubstep(p) {
    const dt = SIM.dt;
    const gamma = SIM.gamma;
    const temp = currentTemp(p);

    const c = Math.exp(-gamma * dt);
    const sigmaNoise = Math.sqrt(Math.max(0, temp * (1 - c * c)));

    computeForces(p);

    for (const frag of state.fragments) {
      for (let i = 0; i < frag.length; i++) {
        const f = frag.forces[i];
        const vel = frag.vel[i];
        const pos = frag.ca[i];

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

  function frameDirection(t, n, b, u, vComp, w) {
    return normalize(add(add(scale(t, u), scale(n, vComp)), scale(b, w)));
  }

  function rotateNormalBinormal(n, b, angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const nr = [
      c * n[0] + s * b[0],
      c * n[1] + s * b[1],
      c * n[2] + s * b[2]
    ];
    const br = [
      -s * n[0] + c * b[0],
      -s * n[1] + c * b[1],
      -s * n[2] + c * b[2]
    ];
    return { n: normalize(nr), b: normalize(br) };
  }

  function hash01(a, b, c) {
    let x = (a * 374761393 + b * 668265263 + c * 2246822519) | 0;
    x = (x ^ (x >>> 13)) | 0;
    x = Math.imul(x, 1274126177);
    x ^= x >>> 16;
    return (x >>> 0) / 4294967296;
  }

  function placeFromFrame(origin, t, n, b, u, vComp, w, length) {
    return add(origin, scale(frameDirection(t, n, b, u, vComp, w), length));
  }

  function buildSidechainAtoms(resName, atomMap, t, n, b, resSeq) {
    const out = [];

    function addAtom(name, el, parentName, u, vComp, w, length) {
      const parent = atomMap[parentName];
      if (!parent) return;
      const pos = placeFromFrame(parent, t, n, b, u, vComp, w, length);
      atomMap[name] = pos;
      out.push({ name, el, res: resSeq, residue: resName, p: pos });
    }

    if (resName === "GLY") return out;
    addAtom("CB", "C", "CA", 0.1, 1.0, 0.35, 1.53);

    switch (resName) {
      case "ALA":
        break;
      case "SER":
        addAtom("OG", "O", "CB", 0.2, 1.0, 0.1, 1.41);
        break;
      case "THR":
        addAtom("OG1", "O", "CB", 0.2, 1.0, 0.1, 1.41);
        addAtom("CG2", "C", "CB", -0.55, 0.85, -0.25, 1.53);
        break;
      case "VAL":
        addAtom("CG1", "C", "CB", 0.55, 0.85, 0.22, 1.53);
        addAtom("CG2", "C", "CB", -0.55, 0.85, -0.22, 1.53);
        break;
      case "LEU":
        addAtom("CG", "C", "CB", 0.25, 0.95, 0.12, 1.53);
        addAtom("CD1", "C", "CG", 0.65, 0.75, 0.2, 1.53);
        addAtom("CD2", "C", "CG", -0.65, 0.75, -0.2, 1.53);
        break;
      case "ILE":
        addAtom("CG1", "C", "CB", 0.6, 0.8, 0.22, 1.53);
        addAtom("CG2", "C", "CB", -0.6, 0.82, -0.18, 1.53);
        addAtom("CD1", "C", "CG1", 0.62, 0.78, 0.2, 1.53);
        break;
      case "ASP":
        addAtom("CG", "C", "CB", 0.3, 0.95, 0.0, 1.52);
        addAtom("OD1", "O", "CG", 0.72, 0.5, 0.1, 1.25);
        addAtom("OD2", "O", "CG", -0.72, 0.5, -0.1, 1.25);
        break;
      case "GLU":
        addAtom("CG", "C", "CB", 0.25, 0.95, 0.0, 1.52);
        addAtom("CD", "C", "CG", 0.25, 0.95, 0.05, 1.52);
        addAtom("OE1", "O", "CD", 0.72, 0.5, 0.1, 1.25);
        addAtom("OE2", "O", "CD", -0.72, 0.5, -0.1, 1.25);
        break;
      case "LYS":
        addAtom("CG", "C", "CB", 0.22, 0.95, 0.0, 1.52);
        addAtom("CD", "C", "CG", 0.22, 0.95, 0.05, 1.52);
        addAtom("CE", "C", "CD", 0.2, 0.95, -0.05, 1.52);
        addAtom("NZ", "N", "CE", 0.18, 0.95, 0.08, 1.47);
        break;
      case "ARG":
        addAtom("CG", "C", "CB", 0.22, 0.95, 0.0, 1.52);
        addAtom("CD", "C", "CG", 0.22, 0.95, 0.05, 1.52);
        addAtom("NE", "N", "CD", 0.2, 0.95, 0.0, 1.46);
        addAtom("CZ", "C", "NE", 0.2, 0.95, 0.0, 1.34);
        addAtom("NH1", "N", "CZ", 0.72, 0.5, 0.08, 1.33);
        addAtom("NH2", "N", "CZ", -0.72, 0.5, -0.08, 1.33);
        break;
      default:
        break;
    }

    return out;
  }

  function rebuildAllAtoms(fragment, includeSidechains) {
    const atoms = [];
    let cmx = 0, cmy = 0, cmz = 0;
    for (let i = 0; i < fragment.length; i++) {
      cmx += fragment.ca[i][0];
      cmy += fragment.ca[i][1];
      cmz += fragment.ca[i][2];
    }
    const inv = fragment.length > 0 ? 1 / fragment.length : 0;
    cmx *= inv; cmy *= inv; cmz *= inv;

    for (let i = 0; i < fragment.length; i++) {
      const residue = fragment.residues[i];
      const resName = residue.name;
      const res = i + 1;
      const ca = fragment.ca[i];
      const prev = fragment.ca[Math.max(0, i - 1)];
      const next = fragment.ca[Math.min(fragment.length - 1, i + 1)];

      let t = normalize(sub(next, prev));
      if (norm(t) < 1e-10) t = [1, 0, 0];

      const towardPrev = normalize(sub(prev, ca));
      const towardNext = normalize(sub(next, ca));
      let n = normalize(add(towardPrev, towardNext));
      if (norm(n) < 1e-10) {
        n = normalize(cross(t, [0, 1, 0]));
        if (norm(n) < 1e-10) n = normalize(cross(t, [1, 0, 0]));
      }
      const b = normalize(cross(t, n));
      n = normalize(cross(b, t));

      const N = add(ca, add(scale(t, -1.45), scale(n, 0.55)));
      const C = add(ca, add(scale(t, 1.52), scale(n, 0.46)));
      const Odir = normalize(add(scale(t, -0.55), scale(b, 1.0)));
      const O = add(C, scale(Odir, 1.23));

      atoms.push({ name: "N", el: "N", residue: resName, res, p: N });
      atoms.push({ name: "CA", el: "C", residue: resName, res, p: ca });
      atoms.push({ name: "C", el: "C", residue: resName, res, p: C });
      atoms.push({ name: "O", el: "O", residue: resName, res, p: O });

      if (includeSidechains) {
        const uN = normalize(sub(N, ca));
        const uC = normalize(sub(C, ca));
        let cbDir = normalize(add(
          add(scale(uN, 0.58), scale(uC, 0.57)),
          scale(cross(uN, uC), 0.54)
        ));

        // Keep sidechains from collapsing into a single inward-facing fan,
        // but avoid a hard global flip that can synchronize entire loops.
        const outDir = normalize(sub(ca, [cmx, cmy, cmz]));
        cbDir = normalize(add(scale(cbDir, 0.72), scale(outDir, 0.5)));

        let tSc = normalize(sub(C, N));
        if (norm(tSc) < 1e-10) tSc = t;
        let bSc = normalize(cross(tSc, cbDir));
        if (norm(bSc) < 1e-10) bSc = b;
        let nSc = normalize(cross(bSc, tSc));
        if (dot(nSc, cbDir) < 0) nSc = scale(nSc, -1);

        const h = hash01(fragment.id + 1, i + 1, resName.charCodeAt(0));
        const parity = (i & 1) ? 1 : -1;
        const chiLike = parity * (48 * DEG) + (h - 0.5) * (220 * DEG);
        const sideFrame = rotateNormalBinormal(nSc, bSc, chiLike);
        const atomMap = { N, CA: ca, C, O };
        const sidechain = buildSidechainAtoms(resName, atomMap, tSc, sideFrame.n, sideFrame.b, res);
        for (const atom of sidechain) atoms.push(atom);
      }
    }

    return atoms;
  }

  function atomLine(serial, atomName, residue, chain, resSeq, x, y, z, element) {
    const serialText = String(serial).padStart(5, " ");
    const atomText = atomName.length < 4 ? atomName.padStart(4, " ") : atomName.slice(0, 4);
    const resText = residue.padStart(3, " ");
    const seqText = String(resSeq).padStart(4, " ");
    const xx = x.toFixed(3).padStart(8, " ");
    const yy = y.toFixed(3).padStart(8, " ");
    const zz = z.toFixed(3).padStart(8, " ");
    const elText = element.padStart(2, " ");
    return `ATOM  ${serialText} ${atomText} ${resText} ${chain}${seqText}    ${xx}${yy}${zz}  1.00 25.00           ${elText}`;
  }

  function buildPdb(includeSidechains) {
    const t0 = performance.now();
    const lines = [];
    let serial = 1;
    let globalResSeq = 1;
    const fragmentAtoms = [];
    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;
    let atomCount = 0;

    for (const fragment of state.fragments) {
      const atoms = rebuildAllAtoms(fragment, includeSidechains);
      fragmentAtoms.push({ fragment, atoms, startResSeq: globalResSeq });
      for (const atom of atoms) {
        sumX += atom.p[0];
        sumY += atom.p[1];
        sumZ += atom.p[2];
        atomCount += 1;
      }
      globalResSeq += fragment.length;
    }

    const cx = atomCount > 0 ? sumX / atomCount : 0;
    const cy = atomCount > 0 ? sumY / atomCount : 0;
    const cz = atomCount > 0 ? sumZ / atomCount : 0;

    for (const entry of fragmentAtoms) {
      for (const atom of entry.atoms) {
        const resSeq = entry.startResSeq + atom.res - 1;
        lines.push(atomLine(
          serial,
          atom.name,
          atom.residue,
          entry.fragment.chain,
          resSeq,
          atom.p[0] - cx,
          atom.p[1] - cy,
          atom.p[2] - cz,
          atom.el
        ));
        serial += 1;
      }
      const lastRes = entry.fragment.residues[entry.fragment.length - 1];
      const endResSeq = entry.startResSeq + entry.fragment.length - 1;
      lines.push(`TER   ${String(serial).padStart(5, " ")}      ${lastRes.name} ${entry.fragment.chain}${String(endResSeq).padStart(4, " ")}`);
      serial += 1;
    }

    lines.push("END");
    state.perf.buildPdbMs = ewma(state.perf.buildPdbMs, performance.now() - t0, 0.25);
    return lines.join("\n");
  }

  async function draw(forceDetail) {
    const drawStart = performance.now();
    const plugin = state.viewer.plugin;
    const includeSidechains = SIM.renderAtomMode === "full" ||
      (SIM.renderAtomMode === "mixed" && state.drawCount % SIM.fullAtomRenderEvery === 0);
    const pdb = buildPdb(includeSidechains);
    const previousStructures = state.activeStructures.slice();

    await state.viewer.loadStructureFromData(pdb, "pdb", {
      dataLabel: `cg-seed-${state.seed}-cycle-${state.cycle}`
    });

    const structures = plugin.managers.structure.hierarchy.selection.structures;
    const latest = structures.length ? structures[structures.length - 1] : null;
    const components = latest && Array.isArray(latest.components) ? latest.components : [];

    if (components.length) {
      await plugin.managers.structure.component.removeRepresentations(components);
      for (const comp of components) {
        await plugin.builders.structure.representation.addRepresentation(comp.cell, {
          type: "spacefill",
          color: "fixed-chain"
        });
      }
    }

    if (previousStructures.length) {
      await plugin.managers.structure.hierarchy.remove(previousStructures, true);
    }
    state.activeStructures = latest ? [latest] : [];
    state.drawCount += 1;
    const drawMs = performance.now() - drawStart;
    state.perf.drawMs = ewma(state.perf.drawMs, drawMs, 0.25);
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
        await draw(force);
        state.lastDrawMs = performance.now();
      } while (state.drawQueued);
    } finally {
      state.drawBusy = false;
    }
  }

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
          const radius = Math.max(18, sphere && Number.isFinite(sphere.radius) ? sphere.radius : SIM.initialSpread);
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
          const snapshot = canvas3d.camera.getFocus([0, 0, 0], SIM.initialSpread * 1.2);
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

  function reseedSystem(seedOverride) {
    if (Number.isFinite(seedOverride)) state.seed = seedOverride >>> 0;
    else state.seed = (state.seed + 1) >>> 0;
    setRandomSeed(state.seed);

    state.cycle = 0;
    state.cps = 0;
    state.cpsWindowStartMs = 0;
    state.cpsWindowStartCycle = 0;
    state.pairWindowStartMs = 0;
    state.pairWindowCount = 0;
    state.pairChecksPerSec = 0;
    state.drawCount = 0;
    state.dynamicRenderIntervalMs = SIM.renderIntervalMs;
    state.activeStructures = [];
    state.fragments = [];
    state.lastMetrics = {
      clashes: 0,
      contacts: 0,
      totalPairs: 1,
      clashRate: 0,
      meanEndToEndNorm: 0
    };
    pickAccentColor();

    const centers = [];
    for (let i = 0; i < SIM.fragmentCount; i++) {
      const fragment = makeFragment(i, 0, centers);
      centers.push(fragment.ca[Math.floor(fragment.length / 2)]);
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
      meanClashRate: 0,
      meanClashes: 0,
      meanPairs: 0,
      meanContacts: 0,
      meanEndToEndNorm: 0,
      perSeed: []
    };

    try {
      for (let s = 0; s < seeds.length; s++) {
        const seed = seeds[s] | 0;
        reseedSystem(seed);
        for (let i = 0; i < warmupCycles; i++) integrateFrame();

        let clashRateSum = 0;
        let clashesSum = 0;
        let pairsSum = 0;
        let contactsSum = 0;
        let endToEndSum = 0;
        for (let i = 0; i < measureCycles; i++) {
          integrateFrame();
          clashRateSum += state.lastMetrics.clashRate;
          clashesSum += state.lastMetrics.clashes;
          pairsSum += state.lastMetrics.totalPairs;
          contactsSum += state.lastMetrics.contacts;
          endToEndSum += state.lastMetrics.meanEndToEndNorm;
        }

        const result = {
          seed,
          meanClashRate: clashRateSum / measureCycles,
          meanClashes: clashesSum / measureCycles,
          meanPairs: pairsSum / measureCycles,
          meanContacts: contactsSum / measureCycles,
          meanEndToEndNorm: endToEndSum / measureCycles
        };
        aggregate.perSeed.push(result);
        aggregate.meanClashRate += result.meanClashRate;
        aggregate.meanClashes += result.meanClashes;
        aggregate.meanPairs += result.meanPairs;
        aggregate.meanContacts += result.meanContacts;
        aggregate.meanEndToEndNorm += result.meanEndToEndNorm;
      }

      const inv = 1 / seeds.length;
      aggregate.meanClashRate *= inv;
      aggregate.meanClashes *= inv;
      aggregate.meanPairs *= inv;
      aggregate.meanContacts *= inv;
      aggregate.meanEndToEndNorm *= inv;
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
        return {
          cycle: state.cycle,
          seed: state.seed,
          running: state.running,
          metrics: {
            clashes: state.lastMetrics.clashes,
            contacts: state.lastMetrics.contacts,
            totalPairs: state.lastMetrics.totalPairs,
            clashRate: state.lastMetrics.clashRate,
            meanEndToEndNorm: state.lastMetrics.meanEndToEndNorm
          }
        };
      }
    };
  }

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
        setStatus("Simulating protein dynamics...");
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
        const oldStructures = state.activeStructures.slice();
        reseedSystem();
        if (oldStructures.length) {
          await state.viewer.plugin.managers.structure.hierarchy.remove(oldStructures, true);
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
