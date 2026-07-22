# Local data

Generated data is not stored in Git.

```text
data/
├── raw/          OpenStreetMap PBF extracts
├── cities/       prepared city GeoPackages
├── processed/    generated tile corpora and audits
├── manifests/    training, validation and test manifests
└── runs/         optional local experiment output
```

The normal workflow reads a PBF from `raw` once, writes a reusable file to `cities`, and then rebuilds coordinate boxes into `processed` as the corpus configuration changes.
