(function () {
  "use strict";

  const DEG = Math.PI / 180;
  const CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  const SIM = {
    fragmentCount: 8,
    minResidues: 18,
    maxResidues: 30,
    frameMs: 12,
    substepsPerFrame: 2,
    renderIntervalMs: 150,
    minRenderIntervalMs: 50,
    maxRenderIntervalMs: 300,
    renderAtomMode: "backbone",
    fullAtomRenderEvery: 8,
    totalCycles: 4000,
    initialSpread: 60,
    finalSpread: 14,
    bondLength: 3.8,
    bondK: 64,
    secK2: 9,
    secK3: 8,
    centerK: 0.035,
    repSigma: 3.3,
    repEps: 1.2,
    attrSigma: 8.8,
    attrEpsStart: 0.015,
    attrEpsEnd: 0.12,
    attrCut: 18,
    nonbondStride: 2,
    neighborListEnabled: true,
    neighborCellSize: 18,
    clashDistance: 3.0,
    dt: 0.026,
    gamma: 0.75,
    tempStart: 8.0,
    tempEnd: 2.8,
    maxForce: 45,
    maxSpeed: 12,
    maxStep: 0.6
  };

  const SECONDARY_TARGETS = {
    helix: { d2: 5.45, d3: 5.2 },
    beta: { d2: 6.9, d3: 10.1 },
    coil: { d2: 6.2, d3: 8.3 }
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
    cycle: document.getElementById("cycle"),
    temperature: document.getElementById("temperature"),
    cps: document.getElementById("cps"),
    drawMs: document.getElementById("drawms"),
    pairsPs: document.getElementById("pairsps"),
    acceptance: document.getElementById("acceptance"),
    status: document.getElementById("status"),
    toggle: document.getElementById("toggle"),
    reset: document.getElementById("reset")
  };

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
    perf: {
      computeMs: 0,
      drawMs: 0,
      buildPdbMs: 0
    },
    lastMetrics: {
      clashes: 0,
      contacts: 0,
      totalPairs: 1
    }
  };

  function setStatus(message) {
    if (el.status) el.status.textContent = message;
  }

  function rand(min, max) {
    return min + Math.random() * (max - min);
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
    const u1 = Math.max(1e-12, Math.random());
    const u2 = Math.random();
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
    const u = Math.random();
    const z = 2 * Math.random() - 1;
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

    const motifs = randInt(2, Math.max(2, Math.floor(fragment.length / 5)));
    const helixChance = 0.55 + 0.2 * p;
    for (let m = 0; m < motifs; m++) {
      const type = Math.random() < helixChance ? "helix" : "beta";
      const span = type === "helix" ? randInt(4, 8) : randInt(3, 6);
      const start = randInt(0, Math.max(0, fragment.length - span));
      for (let i = start; i < start + span; i++) fragment.ss[i] = type;
    }
  }

  function initialDihedralForSs(ssType) {
    if (ssType === "helix") return rand(35, 70) * DEG;
    if (ssType === "beta") return (Math.random() < 0.5 ? -1 : 1) * rand(150, 178) * DEG;
    return rand(-145, 145) * DEG;
  }

  function buildInitialCaChain(length, center, ss) {
    const ca = new Array(length);
    const frame = randomFrame();
    const bond = SIM.bondLength;
    const theta = 111.0 * DEG;

    const start = add(center, scale(frame.x, -0.5 * bond * (length - 1)));
    ca[0] = start;
    if (length === 1) return ca;

    ca[1] = add(ca[0], scale(frame.x, bond));
    if (length === 2) return ca;

    const dir2 = normalize(add(scale(frame.x, -Math.cos(theta)), scale(frame.y, Math.sin(theta))));
    ca[2] = add(ca[1], scale(dir2, bond));

    for (let i = 3; i < length; i++) {
      const mode = ss[i - 1];
      const tau = initialDihedralForSs(mode);
      ca[i] = placeFromThreePoints(ca[i - 3], ca[i - 2], ca[i - 1], bond, theta, tau);
    }
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
      chain: "A",
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
    const fmag = (24 * eps * (2 * sr12 - sr6)) * Math.sqrt(invR2) * Math.sqrt(invR2);

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
      }

      for (let i = 0; i < frag.length - 2; i++) {
        const target2 = SECONDARY_TARGETS[frag.ss[i]].d2;
        potential += applySpring(frag, i, frag, i + 2, SIM.secK2 * secBias, target2);
      }

      for (let i = 0; i < frag.length - 3; i++) {
        const target3 = SECONDARY_TARGETS[frag.ss[i]].d3;
        potential += applySpring(frag, i, frag, i + 3, SIM.secK3 * secBias, target3);
      }
    }

    const cellSize = SIM.neighborCellSize;
    const invCellSize = 1 / cellSize;
    const beads = [];
    const grid = new Map();
    const stride = Math.max(1, SIM.nonbondStride | 0);
    const attrCut2 = SIM.attrCut * SIM.attrCut;
    const clashCut2 = SIM.clashDistance * SIM.clashDistance;

    for (let fi = 0; fi < state.fragments.length; fi++) {
      const frag = state.fragments[fi];
      for (let i = 0; i < frag.length; i += stride) {
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
              pairs += 1;

              const dVec = sub(b.p, a.p);
              const d2 = dot(dVec, dVec);
              if (d2 < clashCut2) clashes += 1;
              if (!same && d2 < 100) contacts += 1;

              const sigmaPair = 0.5 * (a.residue.size + b.residue.size);
              const hydroPair = 0.5 * (a.residue.hydro + b.residue.hydro);
              const qq = a.residue.charge * b.residue.charge;
              const electroBias = qq < 0 ? 1.25 : qq > 0 ? 0.65 : 1.0;

              const repSigma = Math.max(2.8, sigmaPair);
              const repCut = repSigma * Math.pow(2, 1 / 6);
              if (d2 < repCut * repCut) {
                potential += applyRepulsiveLJ(a.frag, a.i, b.frag, b.i, dVec, d2, repSigma, SIM.repEps);
                pairComputed += 1;
              }
              if (!same || Math.abs(a.i - b.i) > 4) {
                const pairAttr = attr * (0.25 + 0.75 * hydroPair) * electroBias;
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
      totalPairs: Math.max(1, pairs)
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
      }
    }

    enforceBondConstraints(2);
  }

  function frameDirection(t, n, b, u, vComp, w) {
    return normalize(add(add(scale(t, u), scale(n, vComp)), scale(b, w)));
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
        const atomMap = { N, CA: ca, C, O };
        const sidechain = buildSidechainAtoms(resName, atomMap, t, n, b, res);
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
      forceDetail ||
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
      await plugin.managers.structure.component.addRepresentation(components, "spacefill");
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
    const p = progress();
    const temp = currentTemp(p);

    const clashFree = 1 - state.lastMetrics.clashes / state.lastMetrics.totalPairs;

    if (el.cycle) el.cycle.textContent = String(state.cycle);
    if (el.temperature) el.temperature.textContent = temp.toFixed(2);
    if (el.cps) el.cps.textContent = state.cps.toFixed(2);
    if (el.drawMs) el.drawMs.textContent = state.perf.drawMs.toFixed(2);
    if (el.pairsPs) el.pairsPs.textContent = Math.round(state.pairChecksPerSec).toLocaleString();
    if (el.acceptance) el.acceptance.textContent = `${(100 * clamp(clashFree, 0, 1)).toFixed(1)}%`;
  }

  function reseedSystem() {
    state.seed += 1;
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
        setStatus(
          `Running coarse-grained dynamics. Render ${Math.round(state.dynamicRenderIntervalMs)}ms, ` +
          `compute ${state.perf.computeMs.toFixed(1)}ms, draw ${state.perf.drawMs.toFixed(1)}ms.`
        );
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

    setStatus("Initializing Mol* viewer...");
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

    enforceNoAutoCenter();
    toggleIlluminationFromDefault();
    await applyIllustrativeStyle();
    enforceNoAutoCenter();

    reseedSystem();
    await queueDraw(true);
    lockFixedCameraView();
    updateHud();

    setStatus("Initialized coarse-grained engine.");
    loop();
  }

  if (el.toggle) {
    el.toggle.addEventListener("click", function () {
      state.running = !state.running;
      el.toggle.textContent = state.running ? "Pause" : "Resume";
      setStatus(state.running ? "Running coarse-grained Langevin + implicit-solvent dynamics." : "Paused.");
    });
  }

  if (el.reset) {
    el.reset.addEventListener("click", async function () {
      if (state.busy) return;
      state.busy = true;
      try {
        setStatus("Reseeding coarse-grained fragments...");
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
