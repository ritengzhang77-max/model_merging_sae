# Critical14 Support Synergy

This audit reuses the existing first-token sweep over `critical12` plus one or two noncritical top-33 ranks.
It asks whether the two support ranks in the validated 14-feature handles behave additively.

## Result

- Critical12 margin: `0.000000` (`I-It`; top token remains `It`).
- Single support ranks tested: `21`; single-support passes: `0`.
- Support pairs tested: `210`; pair passes: `5`.
- Pair-margin counts: `{'0.015625': 5, '0.000000': 114, '-0.015625': 63, '-0.031250': 23, '-0.046875': 4, '-0.062500': 1}`.
- Passing-pair support-rank frequency: `{10: 2, 22: 3, 23: 2, 29: 1, 31: 2}`.

No single support rank crosses the first-token gate. Only 5/210 two-support pairs pass, and each passes at the
minimum positive observed margin (`I-It = +0.015625`). This supports a brittle pair interaction rather than an
independent support-rank story.

## Passing Support Pairs

| ranks | features | singleton margins | pair margin | interaction | signs | descriptions |
|---|---|---:|---:|---:|---|---|
| `10+22` | `8754+9149` | `-0.015625`, `0.000000` | `0.015625` | `0.031250` | donor_higher / recipient_higher | information about specific individuals, particularly athletes and their backgrounds / key phrases related to established organizations and their foundations |
| `10+31` | `8754+1813` | `-0.015625`, `0.000000` | `0.015625` | `0.031250` | donor_higher / donor_higher | information about specific individuals, particularly athletes and their backgrounds / expressive statements of excitement and emotion |
| `22+23` | `9149+7531` | `0.000000`, `-0.015625` | `0.015625` | `0.031250` | recipient_higher / donor_higher | key phrases related to established organizations and their foundations / sections of code, specifically highlighting variable declarations and control structures in programming |
| `22+29` | `9149+321` | `0.000000`, `-0.031250` | `0.015625` | `0.046875` | recipient_higher / donor_higher | key phrases related to established organizations and their foundations / phrases and connections related to reasoning and justification |
| `23+31` | `7531+1813` | `-0.015625`, `0.000000` | `0.015625` | `0.031250` | donor_higher / donor_higher | sections of code, specifically highlighting variable declarations and control structures in programming / expressive statements of excitement and emotion |

## Top Interactions

| ranks | pair margin | singleton margins | interaction | passes |
|---|---:|---:|---:|---:|
| `22+29` | `0.015625` | `0.000000`, `-0.031250` | `0.046875` | `1` |
| `23+29` | `0.000000` | `-0.015625`, `-0.031250` | `0.046875` | `0` |
| `29+30` | `0.000000` | `-0.031250`, `-0.015625` | `0.046875` | `0` |
| `10+22` | `0.015625` | `-0.015625`, `0.000000` | `0.031250` | `1` |
| `10+31` | `0.015625` | `-0.015625`, `0.000000` | `0.031250` | `1` |
| `22+23` | `0.015625` | `0.000000`, `-0.015625` | `0.031250` | `1` |
| `23+31` | `0.015625` | `-0.015625`, `0.000000` | `0.031250` | `1` |
| `3+29` | `0.000000` | `0.000000`, `-0.031250` | `0.031250` | `0` |
| `6+10` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `6+15` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `6+30` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `10+17` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `10+23` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `10+30` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `12+29` | `0.000000` | `0.000000`, `-0.031250` | `0.031250` | `0` |
| `17+23` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `23+30` | `0.000000` | `-0.015625`, `-0.015625` | `0.031250` | `0` |
| `29+31` | `0.000000` | `-0.031250`, `0.000000` | `0.031250` | `0` |
| `29+32` | `0.000000` | `-0.031250`, `0.000000` | `0.031250` | `0` |
| `4+23` | `-0.015625` | `-0.031250`, `-0.015625` | `0.031250` | `0` |
| `6+29` | `-0.015625` | `-0.015625`, `-0.031250` | `0.031250` | `0` |
| `10+29` | `-0.015625` | `-0.015625`, `-0.031250` | `0.031250` | `0` |
| `17+29` | `-0.015625` | `-0.015625`, `-0.031250` | `0.031250` | `0` |
| `19+29` | `-0.015625` | `-0.015625`, `-0.031250` | `0.031250` | `0` |
| `4+29` | `-0.031250` | `-0.031250`, `-0.031250` | `0.031250` | `0` |

## Interpretation

Pair interaction alone is not sufficient: some high-interaction pairs only tie at `I-It = 0.000000` and still fail.
The useful condition is exact signed composition plus being already at the critical12 tie. This strengthens the
first-token basin account and weakens a simple feature-wise semantic interpretation.

Artifacts:

- `support_singletons.csv`
- `support_pair_synergy.csv`
- `manifest.json`
