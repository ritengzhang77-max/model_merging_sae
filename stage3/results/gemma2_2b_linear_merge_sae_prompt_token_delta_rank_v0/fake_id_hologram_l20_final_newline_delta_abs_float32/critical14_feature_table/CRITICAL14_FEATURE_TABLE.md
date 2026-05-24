# Critical-14 Feature Table

Layer-20 final-newline `delta_add` features from the hologram prompt-token ranking.

Critical core ranks are the 12 ranks whose leave-one-out removal from top-33 fails. Support candidates are the five noncritical ranks that appear in the validated 14-feature pairs.

| rank | feature | role | support pairs | sign | harmful delta | Neuronpedia description |
|---:|---:|---|---|---|---:|---|
| 1 | 15169 | `critical_core` | `` | `donor_higher` | 12.2785 | key phrases and indicators of parental concerns and decision-making processes |
| 2 | 12963 | `critical_core` | `` | `donor_higher` | 7.3295 | requests for action or information in a procedural context |
| 7 | 14991 | `critical_core` | `` | `donor_higher` | 5.0757 | queries and requests for confirmation or clarification in discussions |
| 9 | 12704 | `critical_core` | `` | `recipient_higher` | -4.5792 | features and elements related to software and its functionality |
| 10 | 8754 | `support_candidate` | `10+22,10+31` | `donor_higher` | 4.6410 | information about specific individuals, particularly athletes and their backgrounds |
| 14 | 8775 | `critical_core` | `` | `donor_higher` | 4.0498 | requests or commands directed at others |
| 16 | 4339 | `critical_core` | `` | `donor_higher` | 3.8497 | expressions indicating agreement or ongoing actions |
| 20 | 4326 | `critical_core` | `` | `recipient_higher` | -3.7987 | indicators of procedural processes and evaluations in contexts such as health and community services |
| 22 | 9149 | `support_candidate` | `10+22,22+23,22+29` | `recipient_higher` | -3.7411 | key phrases related to established organizations and their foundations |
| 23 | 7531 | `support_candidate` | `22+23,23+31` | `donor_higher` | 3.6762 | sections of code, specifically highlighting variable declarations and control structures in programming |
| 24 | 9135 | `critical_core` | `` | `donor_higher` | 3.6578 | references to file and package management in a coding environment |
| 25 | 13622 | `critical_core` | `` | `recipient_higher` | -3.5940 | phrases related to limits and convergence in mathematical contexts |
| 27 | 12652 | `critical_core` | `` | `donor_higher` | 3.5501 | elements related to API requests and responses in code |
| 28 | 1338 | `critical_core` | `` | `donor_higher` | 3.5421 | titles and roles related to leadership in marketing |
| 29 | 321 | `support_candidate` | `22+29` | `donor_higher` | 3.5059 | phrases and connections related to reasoning and justification |
| 31 | 1813 | `support_candidate` | `10+31,23+31` | `donor_higher` | 3.4750 | expressive statements of excitement and emotion |
| 33 | 6289 | `critical_core` | `` | `recipient_higher` | -3.4215 | phrases related to organizational structure and leadership transitions |

## Validated 14-Feature Candidate Rank Sets

- critical12 + ranks `10,22`: `1,2,7,9,10,14,16,20,22,24,25,27,28,33`
- critical12 + ranks `10,31`: `1,2,7,9,10,14,16,20,24,25,27,28,31,33`
- critical12 + ranks `22,23`: `1,2,7,9,14,16,20,22,23,24,25,27,28,33`
- critical12 + ranks `22,29`: `1,2,7,9,14,16,20,22,24,25,27,28,29,33`
- critical12 + ranks `23,31`: `1,2,7,9,14,16,20,23,24,25,27,28,31,33`

## Interpretation Note

The labels are Neuronpedia autointerp hypotheses, not causal proof. They are mostly generic request/procedure/code/organization labels rather than explicit safety/refusal concepts.
