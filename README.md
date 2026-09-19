
# Final Evaluation Results

## Golden Set

The evaluation uses 150 manually labeled AmazonHelp customer-support cases.

The intent distribution is intentionally sampled from the cleaned corpus but is not class-balanced:

| Intent | Count |
|---|---:|
| delivery_tracking | 52 |
| other | 19 |
| payment_billing | 18 |
| order_product_issue | 12 |
| return_refund | 10 |
| product_info_availability | 9 |
| technical_device_digital | 9 |
| prime_subscription | 9 |
| complaint_service | 9 |
| account_access | 3 |

Expected escalation distribution:

| Action | Count |
|---|---:|
| ESCALATE | 94 |
| AUTO_HANDLE | 56 |

## Intent Classification Results

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline | 34.67% | 0.0515 |
| TF-IDF + 1-NN | 40.00% | 0.2749 |
| Gemini intent classifier | **72.67%** | **0.6378** |

Gemini weighted-F1: 0.7065.

The Gemini evaluator classified all 150 examples successfully.

Per-intent results show strongest performance on delivery_tracking (F1 0.89), return_refund (0.91), payment_billing (0.88), and account_access (0.86). The weakest category was product_info_availability (F1 0.00).

## Escalation Results

The production escalation policy achieved:

- Accuracy: 48.67%
- Precision for ESCALATE: 75.76%
- Recall for ESCALATE: 26.60%
- F1 for ESCALATE: 0.3937

The system is conservative: it has relatively high escalation precision but misses many cases that were labeled for escalation.

## Reply Quality

A fixed 10-example sample was evaluated using an LLM judge on five dimensions:

| Dimension | Mean |
|---|---:|
| Correctness | 4.60 / 5 |
| Groundedness | 5.00 / 5 |
| Relevance | 4.70 / 5 |
| Empathy | 4.40 / 5 |
| Safety | 5.00 / 5 |
| Overall | **4.74 / 5** |

### Human-Judge Agreement

The same 10 replies were independently rated by a human using the same 1-5 overall rubric.

- Exact agreement: 20%
- Mean absolute difference: 1.10
- Quadratic-weighted Cohen's kappa: 0.1593
- Spearman correlation: 0.4726

The limited agreement means the LLM judge should not be treated as ground truth. The result is used as a calibration signal, and a larger human-rated set would be required for stronger validation.

## What Is Misleading About My Headline Number?

The 72.67% intent accuracy is not representative of a production population. The Golden Set contains 52 delivery_tracking examples out of 150, and only three account_access examples. Accuracy therefore depends partly on the sampled class distribution.

The 4.74/5 reply-quality score is also potentially misleading because the judge-human calibration produced only 20% exact agreement and a weighted kappa of 0.1593. Politeness and plausible wording can cause an automated judge to over-score responses that a human considers unhelpful.

## Top Failure Modes

1. Product-information questions overlap with technical and general-support language.
2. Generic polite responses can receive high automated scores despite failing to address the customer's issue.
3. Customer-service complaints can obscure the underlying operational intent.
4. Multi-issue conversations are forced into a single-label taxonomy.
5. Multilingual, abbreviated, and informal social-media text reduces explicit intent signals.

## What Was Not Built

This MVP does not perform live order lookup, real refund execution, account authentication against Amazon systems, production CRM integration, or an actual human-agent handoff.

## Next Week

- Expand the Golden Set to 250+ examples.
- Increase human-rated reply-quality samples.
- Refine the LLM-judge rubric using disagreement cases.
- Redesign product_info_availability and complaint_service boundaries using confusion analysis.
- Add multi-label or primary/secondary intent classification.
- Improve retrieval using intent-aware filtering and conversational context.
- Add production monitoring, confidence calibration, and audit logs.
