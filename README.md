# cing.net — Protein Dynamics Homepage

A personal homepage that doubles as a real-time coarse-grained protein dynamics simulation. Six peptide fragments fold and interact through Langevin dynamics using the MARTINI 2.2 forcefield, rendered live in the browser using [Mol\*](https://molstar.org/).

**Live:** [cing.net](https://cing.net)

## How It Works

The simulation runs entirely client-side with no backend. A single-file physics engine (`app.js`, ~1800 lines) drives the dynamics:

1. **Peptide fragments** — 6 chains of 15–25 residues each, drawn from all 20 standard amino acids with weighted natural abundance. Each residue is mapped to MARTINI 2.2 coarse-grained beads (backbone BB + sidechain SC1–SC4). Each chain is assigned secondary structure motifs (helix, beta sheet, or coil).

2. **MARTINI 2.2 (Dry) forcefield** — Uses real physical units (nm, kJ/mol, ps, K). Bonded interactions include harmonic springs, angle bending, and proper cosine dihedrals with SS-dependent backbone parameters (helix: constraint bonds at 0.31 nm, 96° angles, strong dihedrals; beta: 0.35 nm bonds, 134° angles; coil: 0.35 nm bonds, 127° angles). Non-bonded interactions use the full MARTINI 18×18 bead-type Lennard-Jones interaction matrix (epsilon levels O through IX, scaled ×0.88 for implicit solvent) plus screened Coulomb (ε_r=15) for charged beads. Intramolecular non-bonded pairs use WCA (repulsive-only LJ) to prevent chain collapse; intermolecular pairs use full 12-6 LJ for hydrophobic association.

3. **Langevin dynamics** — Temperature anneals from 400 K to 300 K over 6000 cycles. SHAKE-like constraints enforce helical backbone bond lengths and aromatic ring geometries. Soft-core LJ linearization prevents numerical blowup from bead overlap.

4. **Mol\* rendering** — Each frame, CG bead positions are serialized to PDB format (BB→CA, SC1→CB, SC2→CG, etc., nm→Å), loaded into Mol\*, and rendered as spacefill spheres with per-chain N→C gradient coloring. The camera is locked and all Mol\* UI is hidden for a clean fullscreen look.

5. **Adaptive frame rate** — Render interval adjusts dynamically based on draw time. A real-time FPS counter is shown in the status bar.

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

## Architecture

```
index.html          CDN fallback loader (jsDelivr → unpkg → local vendor)
app.js              Simulation engine (IIFE)
├── SIM             MARTINI-unit physics parameters (nm, kJ/mol, ps, K)
├── MARTINI_RESIDUES  All 20 amino acids with CG bead topologies
├── INTERACTION_MATRIX  18×18 MARTINI bead-type LJ epsilon levels
├── BB_PARAMS       SS-dependent backbone bonded parameters
├── makeFragment()  Chain construction with bead placement + topology
├── computeForces() Bonded + LJ (WCA intra / full inter) + Coulomb + walls
├── integrateSubstep()  Langevin integrator + SHAKE bond constraints
├── buildPdb()      Serialize CG beads → PDB string (nm → Å)
├── draw()          Load PDB into Mol*, apply spacefill + gradient colors
└── loop()          Main loop: integrate → draw → repeat
styles.css          HUD overlay styling
vendor/molstar.*    Local Mol* fallback
```

## Deployment

Deployed to [cing.net](https://cing.net) via GitHub Pages from the `main` branch.
