
**Live demo:** https://huggingface.co/spaces/kotharidipti75/policylens

# PolicyLens 🔎

**A privacy nutrition label for any policy, with your legal rights and change tracking.**

## The problem
Almost everyone accepts privacy policies without reading them. The text is long and legal, and it changes silently. Privacy laws (GDPR, CCPA, DPDP, LGPD) give people real rights, but few know what those rights are.

## What it does
1. **Analyze a policy:** paste text, get an A-F privacy grade, the risky clauses with confidence scores, and your rights under your region's law.
2. **Detect policy changes:** paste two versions and see which new risks an update introduced.

## How it works
Clause splitting, then zero-shot classification with an open-source NLI model (`nli-deberta-v3-xsmall`) running in the browser via Transformers.js, then risk scoring and region-specific rights. The text never leaves the user's device.

## Evaluation
Tested on 31 labeled clauses (`none` or one of 8 categories). Most were written by me as illustrative examples, so treat the numbers as indicative.

| Method | Accuracy |
|---|---|
| Keyword baseline | 74% |
| Zero-shot model | 81% |

The evaluation used the PyTorch version of the same model with a 0.7 threshold.

## Limitations
- Small evaluation set, mostly self-written clauses.
- Small model: it can misread context, for example flagging essential login cookies as tracking.
- Analysis is limited to the first 40 clauses to keep it fast in the browser.
- English only. The rights text is a short summary, not legal advice.

## Roadmap
Multilingual input (NLLB-200), RAG over full legal texts (bge-m3 + Qdrant), a browser extension, and scheduled policy monitoring.
