---
name: license-provenance
description: Establish the license and origin of a dataset or model before use. Use when adding any data or weights, or before a release.
---

# License and provenance

1. Read the license at its source: the dataset's own license file or model card. Aggregators, blog posts and newsletters only tell you where to look.
2. Record in `configs/datasets.yaml`: exact license, source URL, access type, and allowed use (`train`, `dev`, `test`, `base`, `baseline`).
3. Set `status: listed` when the source tags a license. **Only a human sets `status: confirmed`.**
4. Check for traps:
   - Derived datasets inherit terms from every source.
   - A gated repo's form may add terms beyond the tag. A human must accept them, so ask.
   - Non-commercial terms on weights (such as Bark 3.0's CC BY-NC) rule out fine-tuning from them.
   - A repack's license doesn't override the upstream license. Check both.
   - Speech is personal data. Note whether speakers consented, and flag anything unclear for legal review.
5. Attribution: CC-BY sources must be credited in the release. Keep the list in the registry current.

If sources conflict, set `status: excluded` until a human decides.
