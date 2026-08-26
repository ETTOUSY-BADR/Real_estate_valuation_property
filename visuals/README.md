# Generated visuals

Visualization modules write publication and diagnostic figures beneath this
directory. PNG outputs are intentionally excluded from Git because they are
reproducible artifacts.

Run the relevant visualization module or phase workflow from the repository
root to regenerate them. Compiled papers kept under `docs/paper/` remain tracked
for convenient review.

Phase 4 generates seven figures under `visuals/phase4/`: the national pipeline,
territorial coverage, frozen-test performance, prediction diagnostics, regional
robustness, feature importance and comparable-sale coverage. Use
`python -m paris_avm.visualization.phase4 --allow-incomplete` to generate the
pipeline figure before the large national benchmark is available.
