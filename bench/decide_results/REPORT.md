# Decision backends, test split (val used only to choose tau)

## embed  (extra RAM 790 MB)

| task | n | accuracy [95 % CI] | Brier | ECE | latency p50 / p95 (s) |
|---|---:|---:|---:|---:|---:|
| done | 75 | 0.667 [0.55, 0.76] | 0.443 | 0.069 | 0.218 / 0.306 |
| route | 40 | 0.850 [0.71, 0.93] | 0.341 | 0.310 | 0.052 / 0.060 |
| mixer | 40 | 0.850 [0.71, 0.93] | 0.306 | 0.229 | 0.054 / 0.065 |

`done` guard: tau = 0.666 (from 8 validation negatives that pass the objective checks)

End to end on the 75 test states (NO = the task is not done): DONE accepted by the objective checks alone: wrong 12/36, right 36/39; objective checks + p(done) >= tau: wrong 1/36, right 9/39.

| tau | wrong DONE accepted (of 36) | right DONE accepted (of 39) |
|---:|---:|---:|
| 0.3 | 12 | 36 |
| 0.5 | 7 | 29 |
| 0.7 | 0 | 5 |
| 0.8 | 0 | 0 |
| 0.9 | 0 | 0 |
| 0.95 | 0 | 0 |

| pad lines | n | AUC (all) | AUC (objective_ok) | accuracy @0.5 | false accepts @tau (objective_ok negatives) | recall @tau (objective_ok positives) | chars median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75 | 0.705 | 0.669 | 0.667 | 1/12 [0.01, 0.35] | 9/36 | 508 |
| 25 | 75 | 0.521 | 0.537 | 0.520 | 7/12 [0.32, 0.81] | 28/36 | 1262 |
| 80 | 75 | 0.521 | 0.537 | 0.520 | 7/12 [0.32, 0.81] | 28/36 | 2894 |

## logprob_qwen35_2b  (extra RAM 0 MB)

| task | n | accuracy [95 % CI] | Brier | ECE | latency p50 / p95 (s) |
|---|---:|---:|---:|---:|---:|
| done | 75 | 0.800 [0.70, 0.87] | 0.256 | 0.039 | 0.761 / 1.102 |
| route | 40 | 0.250 [0.14, 0.40] | 1.324 | 0.689 | 0.792 / 0.798 |
| mixer | 40 | 0.450 [0.31, 0.60] | 0.805 | 0.384 | 0.415 / 0.434 |

`done` guard: tau = 0.851 (from 8 validation negatives that pass the objective checks)

End to end on the 75 test states (NO = the task is not done): DONE accepted by the objective checks alone: wrong 12/36, right 36/39; objective checks + p(done) >= tau: wrong 0/36, right 14/39.

| tau | wrong DONE accepted (of 36) | right DONE accepted (of 39) |
|---:|---:|---:|
| 0.3 | 8 | 34 |
| 0.5 | 2 | 26 |
| 0.7 | 0 | 20 |
| 0.8 | 0 | 16 |
| 0.9 | 0 | 7 |
| 0.95 | 0 | 3 |

| pad lines | n | AUC (all) | AUC (objective_ok) | accuracy @0.5 | false accepts @tau (objective_ok negatives) | recall @tau (objective_ok positives) | chars median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75 | 0.925 | 0.866 | 0.800 | 0/12 [0.00, 0.24] | 14/36 | 508 |
| 25 | 75 | 0.927 | 0.868 | 0.813 | 0/12 [0.00, 0.24] | 9/36 | 1262 |
| 80 | 75 | 0.912 | 0.847 | 0.840 | 0/12 [0.00, 0.24] | 8/36 | 2894 |

## logprob_qwen35_2b_letters  (extra RAM 0 MB)

| task | n | accuracy [95 % CI] | Brier | ECE | latency p50 / p95 (s) |
|---|---:|---:|---:|---:|---:|
| route | 40 | 0.525 [0.37, 0.67] | 0.645 | 0.184 | 0.790 / 0.795 |

## logprob_qwen35_4b  (extra RAM 0 MB)

| task | n | accuracy [95 % CI] | Brier | ECE | latency p50 / p95 (s) |
|---|---:|---:|---:|---:|---:|
| done | 75 | 0.747 [0.64, 0.83] | 0.399 | 0.171 | 2.876 / 3.515 |
| route | 40 | 0.900 [0.77, 0.96] | 0.140 | 0.056 | 2.868 / 2.916 |
| mixer | 40 | 0.925 [0.80, 0.97] | 0.101 | 0.094 | 1.840 / 1.894 |

`done` guard: tau = 0.678 (from 8 validation negatives that pass the objective checks)

End to end on the 75 test states (NO = the task is not done): DONE accepted by the objective checks alone: wrong 12/36, right 36/39; objective checks + p(done) >= tau: wrong 0/36, right 13/39.

| tau | wrong DONE accepted (of 36) | right DONE accepted (of 39) |
|---:|---:|---:|
| 0.3 | 4 | 25 |
| 0.5 | 0 | 20 |
| 0.7 | 0 | 12 |
| 0.8 | 0 | 3 |
| 0.9 | 0 | 0 |
| 0.95 | 0 | 0 |

| pad lines | n | AUC (all) | AUC (objective_ok) | accuracy @0.5 | false accepts @tau (objective_ok negatives) | recall @tau (objective_ok positives) | chars median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75 | 0.912 | 0.838 | 0.747 | 0/12 [0.00, 0.24] | 13/36 | 508 |
| 25 | 75 | 0.941 | 0.894 | 0.693 | 0/12 [0.00, 0.24] | 6/36 | 1262 |
| 80 | 75 | 0.944 | 0.903 | 0.653 | 0/12 [0.00, 0.24] | 7/36 | 2894 |

