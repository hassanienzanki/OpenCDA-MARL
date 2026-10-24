# RAL baseline checkpoints

Eval-only weights of the framework paper's baselines, one directory per
run, paired with the bundled recordings they were trained against
(`traffic_<vph>vph.json`). `manifest.json` lists every entry with its
profile, algorithm and SHA-256; `SHA256SUMS` covers the files here.

    MARL:
      training:
        training_mode: false
        load_checkpoint: "ral/sac_300vph"      # resolves inside the wheel

with the entry's profile, e.g. `-t marl/intersection/sac --marl`. Set
`training_mode: true` to fine-tune (the optimizer restarts: its state was
stripped to keep the wheel small).

The paper's two 1024-wide TD3-44d baselines are not distributed; their
profile (`marl/intersection/td3_44d`) ships so they can be retrained, and
`not_distributed` in the manifest names the commit holding the originals.
