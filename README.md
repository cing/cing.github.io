# cing.net — Protein Dynamics Homepage

A personal homepage that doubles as a real-time coarse-grained protein dynamics simulation. Eight peptide fragments fold and interact through Langevin dynamics, rendered live in the browser using [Mol\*](https://molstar.org/).

**Live:** [cing.net](https://cing.net)

## How It Works

The simulation runs entirely client-side with no backend. A single-file physics engine (`app.js`, ~1800 lines) drives the dynamics:

1. **Peptide fragments** — 8 chains of 18–30 residues each, drawn from 11 amino acid types with distinct sidechain geometries. Each chain is assigned a secondary structure target (helix, beta sheet, or coil).

2. **Langevin dynamics** — Forces include bonded springs, angle bending, dihedral constraints for secondary structure, Lennard-Jones repulsion/attraction between chains, and a centering potential. Temperature anneals over 4000 cycles to drive folding.

3. **Mol\* rendering** — Each frame, the simulation state is serialized to PDB format, loaded into Mol\*, and rendered as spacefill spheres with per-chain coloring. The camera is locked and all Mol\* UI is hidden for a clean fullscreen look.

4. **Adaptive frame rate** — Render interval adjusts dynamically based on draw time. A real-time FPS counter is shown in the status bar.

## Running Locally

No build step required — pure vanilla JS/HTML/CSS served statically.

```bash
python3 -m http.server 4173
# Open http://127.0.0.1:4173/
```

## Controls

| Input | Action |
|-------|--------|
| **Pause** button / `Space` | Toggle simulation |
| **Reseed** button / `R` | Regenerate all fragments with a new random seed |

## Benchmarking

```bash
node scripts/benchmark-node.mjs
```

Uses Chrome DevTools Protocol to run headless performance benchmarks. Configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `4173` | Local server port |
| `DURATION_MS` | `12000` | Benchmark duration |
| `SAMPLE_MS` | `500` | Sampling interval |
| `DRAW_MODE` | `full` | `full`, `mixed`, `backbone`, or `ca` |
| `SIDECHAIN` | `full` | `full` or `cb` |

## URL Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `drawMode` | `full`, `mixed`, `backbone`, `ca` | Atom detail level |
| `fullEvery` | integer | In `mixed` mode, full-detail render every N frames |
| `sidechain` | `full`, `cb` | Sidechain detail level |

## Architecture

```
index.html          CDN fallback loader (jsDelivr → unpkg → local vendor)
app.js              Simulation engine (IIFE)
├── SIM             ~40 tunable physics parameters
├── RESIDUE_LIBRARY 11 amino acid types with sidechain definitions
├── makeFragment()  Chain construction with secondary structure assignment
├── computeForces() Bond / angle / dihedral / LJ / centering forces
├── integrateSubstep()  Langevin integrator + SHAKE bond constraints
├── buildPdb()      Serialize simulation state → PDB string
├── draw()          Load PDB into Mol*, apply spacefill + chain colors
└── loop()          Main loop: integrate → draw → repeat
styles.css          HUD overlay styling
vendor/molstar.*    Local Mol* fallback
```

## Deployment

Deployed to [cing.net](https://cing.net) via GitHub Pages from the `main` branch.
