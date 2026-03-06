# Academic Writing Standard Checklist

## Purpose
This checklist maps the current draft paper to common academic standards from IEEE, ACM, and NeurIPS-style reporting practice.

## External Standards Referenced
1. IEEE Author Center: structure, clarity, reproducibility expectations.  
   https://ieeeauthorcenter.ieee.org/
2. ACM Publications policies: claims, citations, and integrity requirements.  
   https://www.acm.org/publications/policies
3. NeurIPS checklist: experimental transparency, limitations, compute/reporting rigor.  
   https://neurips.cc/Conferences/2024/PaperInformation/PaperChecklist

## Required Structure
- Problem statement and gap definition.
- Explicit method description and architecture.
- Reproducible experiment setup.
- Quantitative results with uncertainty/significance.
- Limitations and negative results.
- Source/citation traceability.

## Current Compliance Status
| Item | Status | Evidence |
|---|---|---|
| Problem and gap clarity | Pass | `docs/DRAFT_PAPER.md` Sections 1-2 |
| Method transparency | Pass | `docs/DRAFT_PAPER.md` Section 3 + `docs/ARCHITECTURE.md` |
| Reproducible setup | Pass | `docs/DRAFT_PAPER.md` Sections 4 and 8 |
| Statistical reporting | Pass | CIs and paired bootstrap deltas in proposal/publication artifacts |
| Negative-result disclosure | Pass | Serving-efficiency weakness explicitly reported |
| Proposal claim alignment | Pass | `docs/PROPOSAL_ALIGNMENT.md` + strict quality gate pass |

## Remaining Improvements for Submission-Grade Paper
1. Expand retrieval labeled query count in the publication core benchmark.
2. Add direct human-subject user study (not only judge-based raters).
3. Include a dedicated threats-to-validity subsection with sampling bias analysis.
4. Add full bibliography metadata (venue/year/DOI) in final camera-ready format.
