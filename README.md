# Mol* Fragment Folding Demo

Fullscreen browser demo that uses Mol* to render glycine-only peptide fragments while a fake Monte Carlo engine drives:

- phi/psi-biased secondary structure formation (helix/sheet/coil basins)
- rigid-body fragment collapse toward a shared assembly center
- pseudo-time evolution via repeated PDB snapshots loaded into Mol*
- Mol* illustrative rendering (`preset-structure-representation-illustrative`) with outline+occlusion
- explicit self/inter-fragment clash penalties to reduce steric overlaps

## Run

```bash
python3 -m http.server 4173
```

Open `http://127.0.0.1:4173/` in a browser.

## Controls

- `Pause/Resume` button or `Space`
- `Reseed` button or `R`
