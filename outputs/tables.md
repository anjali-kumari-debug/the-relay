# Tables (generated from outputs/results.csv)

## Table 2a - Latency and offered load

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate per min | First token ms p50 / p95 / p99 | Total time ms p50 / p95 / p99 |
|---|---|---|---|---|---|---|
| steady | Stand-in | 21 | 895.0 | 1.4 | 604 / 830 / 1,043 (n=21) | 640 / 934 / 1,120 (n=21) |
| burst | Stand-in | 30 | 896.3 | 2.0 | 615 / 10,142 / 17,424 (n=14) | 20,002 / 20,003 / 20,004 (n=30) |
| sustained_overload | Stand-in | 84 | 905.8 | 5.7 | 13,717 / 19,090 / 19,415 (n=73) | 14,759 / 20,003 / 20,004 (n=84) |
| rpm_halved | Stand-in | 21 | 914.4 | 1.4 | 3,984 / 13,917 / 14,454 (n=16) | 5,980 / 20,005 / 20,006 (n=21) |
| daily_exhausted | Stand-in | 30 | 290.0 | 6.2 | not measured | 2 / 3 / 3 (n=30) |
| real_steady | Real | 10 | 270.7 | 2.2 | 663 / 721 / 735 (n=10) | 714 / 775 / 798 (n=10) |
| overload_retry_on | Stand-in | 84 | 892.8 | 5.7 | 9,739 / 18,583 / 18,924 (n=57) | 9,798 / 18,868 / 19,430 (n=84) |
| overload_retry_off | Stand-in | 84 | 891.6 | 5.7 | 642 / 1,758 / 3,598 (n=43) | 391 / 1,158 / 3,151 (n=84) |
| real_daily_exhausted | Real | 3 | 40.0 | 4.5 | not measured | 2 / 93 / 101 (n=3) |

Notes: percentiles use linear interpolation between order statistics (numpy default, "type 7"); n is the
number of rows each percentile is computed over. First token = arrival to first answer content token
(includes queueing and retrieval), over `ok` rows that produced a token. Total time = arrival to final
outcome (answer, refusal or rejection), over all rows. Duration = max(start + total) - min(start).
Offered rate per min = requests / (max(start) - min(start)) * 60, where the denominator is the span of
arrival times (a burst concentrated in one second therefore inflates it); "not measured" if all arrivals
are simultaneous. results.csv is append-only: use one run per scenario per results file (or --outputs-dir),
otherwise Duration and Offered rate span the gap between repeated runs.

## Table 2b - Outcomes, queueing and cost

| Scenario | Real or stand-in | Succeeded % | Rate-limited % | Timed out or given up % | Other errors % | Retries per request (mean) | Peak queue length | Tokens per answer (mean) | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|---|---|
| steady | Stand-in | 100.0% | 0.0% | 0.0% | 0.0% | 0.00 | 0 | 3,192 | $0.00 (free tier); paid-equivalent not measured |
| burst | Stand-in | 46.7% | 0.0% | 53.3% | 0.0% | 0.00 | 17 | 2,945 | $0.00 (free tier); paid-equivalent not measured |
| sustained_overload | Stand-in | 86.9% | 0.0% | 13.1% | 0.0% | 0.01 | 1 | 1,559 | $0.00 (free tier); paid-equivalent not measured |
| rpm_halved | Stand-in | 76.2% | 4.8% | 19.0% | 0.0% | 0.05 | 0 | 3,096 | $0.00 (free tier); paid-equivalent not measured |
| daily_exhausted | Stand-in | 0.0% | 100.0% | 0.0% | 0.0% | 0.00 | 0 | not measured | $0.00 (free tier); paid-equivalent not measured |
| real_steady | Real | 100.0% | 0.0% | 0.0% | 0.0% | 0.00 | 0 | 3,107 | $0.00 (free tier); paid-equivalent not measured |
| overload_retry_on | Stand-in | 67.9% | 31.0% | 1.2% | 0.0% | 0.40 | 1 | 2,130 | $0.00 (free tier); paid-equivalent not measured |
| overload_retry_off | Stand-in | 51.2% | 48.8% | 0.0% | 0.0% | 0.00 | 0 | 2,910 | $0.00 (free tier); paid-equivalent not measured |
| real_daily_exhausted | Real | 0.0% | 100.0% | 0.0% | 0.0% | 0.00 | 0 | not measured | $0.00 (free tier); paid-equivalent not measured |

Notes: percentages are shares of all requests sent in that scenario (they sum to 100% with "Other errors").
Peak queue length = max `queue_len_at_arrival` (queue length seen by a request on arrival). Tokens per answer =
mean tokens_in + tokens_out over `ok` rows (refusals made without a model call count as 0 tokens; tokens of
all attempts of a request are included). Cost per 1,000 answers = all tokens spent in the scenario, including
failed requests, priced per million tokens, divided by `ok` answers, times 1,000.

## Table 3 - Retrieval settings (scenario = retrieval_eval)

| Retrieval setting | Real or stand-in | Questions asked | Graded (n) | Correct and cited % | Wrong % | Correctly refused % | Wrongly refused % | Tokens per answer (mean) | First token ms p50 / p95 | Total time ms p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| A:250w/k3/rerank=no | Real | 18 | 17 | 64.7% | 11.8% | 11.8% | 11.8% | 1,655 | 591 / 1,787 (n=17) | 603 / 1,792 (n=18) |
| B:450w/k4/rerank=no | Real | 18 | 17 | 76.5% | 5.9% | 11.8% | 5.9% | 3,327 | 751 / 2,453 (n=17) | 806 / 4,097 (n=18) |
| C:450w/k4/rerank=yes | Real | 18 (19 rows) | 17 | 70.6% | 11.8% | 11.8% | 5.9% | 3,387 | 767 / 5,191 (n=17) | 791 / 11,591 (n=19) |

Notes: the four grades (correct, wrong, refused correctly, refused wrongly) are shares of the graded rows
(non-blank `answer_grade`, n shown), so they sum to 100% per setting; "Questions asked" counts distinct questions, with the number of
rows in brackets when a question was attempted more than once (a budget-rejected attempt was re-run).
Non-ok rows are never graded. Percentiles: linear interpolation (numpy "type 7"),
n in parentheses. Tokens per answer as in Table 2b. First token over `ok` rows with a token, total over all rows.
