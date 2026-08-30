# Prospective-analysis inputs

`prospective-analysis-plan.json` is the concrete, content-addressed schema-v2
prospective plan selected only by `configs/policy_prospective.toml`.

`prospective-analysis-plan-amendment-v3.json` is its explicit schema-v3
successor, selected only by the fail-closed `configs/policy_campaign.toml`.
The parent file remains unchanged. The successor preserves the parent primary
estimand, population predicate, scenario direction, period semantics, metric
contract, and harm weights while binding the expanded seed, population,
monetary, uncertainty, convergence, output, flow, and execution-attestation
contracts. It remains `UNREGISTERED`, `preregistered=false`, and
`campaign_ready=false`.

`parameter-uncertainty-design-v1.json` declares deterministic seeded
Latin-hypercube draws over explicitly illustrative parameter ranges. Those
ranges are not calibrated probability distributions and do not establish
empirical parameter uncertainty.

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

Regenerate the fail-closed successor only after reviewing every changed input
identity with:

```text
python tools/build_campaign_analysis_plan.py
```

The generators resolve and hash the actual configurations, scenario catalogue,
model inputs, population adapter, profile lineage, metric registry, harm
weights, and output profiles. The successor builder also re-attests its parent,
parameter design, monetary evidence, output contract, and execution-receipt
schema. Neither builder creates a cohort, executes a scenario, or starts a
campaign.
