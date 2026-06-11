# Changelog — hicache-plus-plus

All notable changes, per version. Auto-generated from git tags by
`third_party/launch_materials/gen_changelogs.sh`; do not edit by hand.

## Unreleased

- fix(doi): update Zenodo DOI to correct concept record (18a1724)
- FLUX A/B: rebuild dev grid with the vanilla column (3-arm), lead with FLUX.1-dev (real target) over Chroma (32e44f5)
- docs(readme): surface the cache-dit FLUX A/B comparison grids (bd4022d)
- evidence: FLUX.1-dev on-target A/B (TaylorSeer 1.15x edges DMD 1.10x); honest domain-split framing for the cache-dit pitch (d993ad5)
- evidence(cache-dit DMD PR): FLUX-class A/B (Chroma1-HD) — DMD 1.11x > TaylorSeer 1.05x > vanilla (432e11e)
- Ignore paper/ (manuscripts never live in the repo) (3599c2c)
- docs: family sign-fix wave -- corrected-vs-as-released re-validation tables (5d620f1)
- release: 1.2.0 -- version sync (__version__ literal + pyproject) with drift test (381754a)
- docs+paper: the truthful selection arc + complete final DiT tables everywhere (3a337f8)
- bench(dit-imagenet): bank Phase 1b (corrected hermite + holdout A/B) + post-eigencache latency re-time; RESULTS_DIT final (01021d2)
- compute_fid: parse auto/_horizon/_fix cell names (was crashing on auto_i4_horizon) (1663317)
- docs+paper: restructure around the domain-split narrative (no single basis wins) (dd4a9e3)
- test(dmd): deflake eigencache-invalidation check (append the trajectory's true next point, print the deviation) (3753429)
- bench(dit-imagenet): bank dmd_i8 + taylor_i4 FID-10k cells; Phase-1b corrected-rerun placeholder table + pre-registered analysis; horizon A/B cell promoted in resume queue (075af83)
- bench(dit-imagenet): --holdout flag + fix resume-queue guards (pure-compute resume) (5eba9b7)
- docs: README refresh for 1.2.0.dev0 (sign fix, auto holdout modes, eigencache, honest DiT status) (87e7c3d)
- perf(dmd): cache the eigendecomposition per compute window (limitation 1) (031a440)
- feat(auto): horizon-matched holdout (opt-in), decided on microbench evidence (21ec37d)
- feat(bench-dit): per-1k-image atomic checkpointing + resume (limitation 5) (66a0e6d)
- fix(bench-dit): taylor_forecast evaluates the monomial at +k, not -k (fb54509)
- fix(hermite): evaluate the Hermite basis at +k, not -k (porting bug) (8ebdab2)
- bench(dit-imagenet): bank FID-10k ladder partials + resume queue (ac640b8)
- DOI badge: static shields.io (Zenodo badge endpoint 302s through GitHub's proxy) (e1c7407)
- Zenodo DOI badge + CITATION.cff (5071b38)
- v1.1.0 (8664a52)
- docs: drop unreleased dit-plus row from the family matrix (36f2733)
- Microbench: auto backend row + drift/regime-switch scenarios (7e1c2fc)
- Holdout 'auto' backend, snapshot ownership fix, README/SEO surgery, PR drafts, preprint (e22adb3)

## v1.2.0 — 2026-06-10

- paper: rebuild PDF (11 pp, final tables) (2cd116d)
- release: 1.2.0 -- version sync (__version__ literal + pyproject) with drift test (074374b)
- docs+paper: the truthful selection arc + complete final DiT tables everywhere (68a8023)
- bench(dit-imagenet): bank Phase 1b (corrected hermite + holdout A/B) + post-eigencache latency re-time; RESULTS_DIT final (3856374)
- compute_fid: parse auto/_horizon/_fix cell names (was crashing on auto_i4_horizon) (8b767e6)
- docs+paper: restructure around the domain-split narrative (no single basis wins) (214abc6)
- test(dmd): deflake eigencache-invalidation check (append the trajectory's true next point, print the deviation) (8e6388c)
- bench(dit-imagenet): bank dmd_i8 + taylor_i4 FID-10k cells; Phase-1b corrected-rerun placeholder table + pre-registered analysis; horizon A/B cell promoted in resume queue (3adc012)
- bench(dit-imagenet): --holdout flag + fix resume-queue guards (pure-compute resume) (7bfdc75)
- docs: README refresh for 1.2.0.dev0 (sign fix, auto holdout modes, eigencache, honest DiT status) (83ebca6)
- perf(dmd): cache the eigendecomposition per compute window (limitation 1) (46b6548)
- feat(auto): horizon-matched holdout (opt-in), decided on microbench evidence (aba0462)
- feat(bench-dit): per-1k-image atomic checkpointing + resume (limitation 5) (6302547)
- fix(bench-dit): taylor_forecast evaluates the monomial at +k, not -k (3ecd13d)
- fix(hermite): evaluate the Hermite basis at +k, not -k (porting bug) (0c94ce0)
- bench(dit-imagenet): bank FID-10k ladder partials + resume queue (540f972)
- DOI badge: static shields.io (Zenodo badge endpoint 302s through GitHub's proxy) (b636add)
- Zenodo DOI badge + CITATION.cff (16dd254)

## v1.1.0 — 2026-06-10

- v1.1.0 (5c6ee1b)
- docs: drop unreleased dit-plus row from the family matrix (a815ac1)
- Microbench: auto backend row + drift/regime-switch scenarios (1ebccde)
- Holdout 'auto' backend, snapshot ownership fix, README/SEO surgery, PR drafts, preprint (9704e1a)
- bench(dit-imagenet): paired-noise FID harness improvement (92c5c78)
- chore: remove unused square logo (banner is the only displayed image) (fbacd1c)
- docs: badge row — add PyPI version + real links, drop vanity badges (fc6647e)

## v1.0.0 — 2026-06-07

- release 1.0.0: show banner on PyPI (absolute-URL README header) + version bump (a9b2d07)
- docs: landscape banner in README header + transparent (bg-removed) square logo for PyPI/icon (e787068)
- docs: add logo to README header + social banner asset (253acdf)
- docs: add icon brief for the hicache-pp logo (6a56439)
- release: PyPI packaging metadata (authors, classifiers incl. OSI MIT, project urls) for hicache-pp 0.1.0 (803db8b)
- docs(results): add TRELLIS.2-v2 row (DMD ≈ Hermite @ i2, +0.03-0.04 @ i3-i4) (4ffcb09)
- docs(readme): add 'when to use' positioning matrix (ae5feba)
- docs: add Citation section (HiCache++ + HiCache, TaylorSeer, DMD/Prony/Matrix-Pencil, Adaptive-CFG) (4adadfd)
- results: add TRELLIS v1 SS-stage comparison (DMD most lossless: 0.829 vs Hermite 0.825 @ ~2.8x) (eed8e99)
- bench_dit: clamp interval to >=1 (method=none baseline passed interval=0) (62e8672)
- add DiT-XL/2 ImageNet-256 FID benchmark harness (63e1f15)
- microbench: add rational (Pade/FoCa) baseline as a third forecast basis (87549dd)
- add controlled forecast microbenchmark; refresh A/B numbers (honest, cross-validated) (d0f2822)
- tree.py: genericize remaining SAM3D mentions in helper docstring + test comments (80c8648)
- polish for publication: generic module headers, methods-comparison table, badges, MIT LICENSE (fd35365)
- integrations: add reproducible git-apply patches for all 4 wirings (Hunyuan-2.1, mini, SAM3D, Fast-SAM3D slat) (9b7f876)
- HiCache++ 0.1.0 — exponential (DMD/Prony) velocity forecaster for diffusion caching (402d3aa)

