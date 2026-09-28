PYTHONPATH=hepformer-paper python3 - <<'PY'
from hepformer.tracking.eval.evaluate import eval_events

fname = "/share/rcifdata/svanstroud/hepformer/hepformer/tracking/logs/HF-final-1GeV-hc0.1-eta4_20250307-T174811/ckpts/epoch=028-val_loss=1.53465__test.h5"
num_events = 100
eta_cut = 4.0
pt_cut = 1.0

tracks, parts = eval_events(fname, num_events=num_events, eta_cut=eta_cut)

# paperplots.ipynb applies this extra post-hoc cut
tracks = tracks[tracks["eta"].abs() < eta_cut]
tracks = tracks[tracks["pt"] > pt_cut]

tgts = parts[parts.reconstructable]
integrated_fr = (~tracks.eff_dm & ~tracks.duplicate).mean()

print(f"N events: {num_events}, N particles: {len(parts)}, N tracks: {len(tracks)}")
print(f"DM Integrated efficiency: {tgts.eff_dm.mean():.1%}")
print(f"DM Integrated fake rate: {integrated_fr:.1%}")
print(f"Duplicate rate: {tracks.duplicate.mean():.1%}")
PY
