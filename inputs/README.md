# Prospective-analysis inputs

`prospective-analysis-plan.json` is the concrete, content-addressed schema-v2
prospective plan selected only by `configs/policy_prospective.toml`.

The files under `illustrative_population/` are deterministic algebraic test
inputs generated from the repository's four-jurisdiction population fixture.
They deliberately use test publishers and `example.invalid` source URLs, have
no authentic signature, and are not substantive population evidence. They are
checked in only so plan loading, content attestation, projected-population
preflight, and configuration binding can be validated without inventing
campaign-ready provenance.

Regenerate the plan after an intentional input-contract change with:

```text
python tools/build_prospective_analysis_plan.py
```

The generator resolves and hashes the actual configuration, scenario catalogue,
model inputs, population adapter, profile lineage, metric registry, harm
weights, and prospective output profile. It does not create a cohort, execute a
scenario, or start a campaign.
