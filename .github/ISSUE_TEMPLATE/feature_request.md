---
name: Feature request
about: Suggest an addition or a change to how something works
title: ""
labels: enhancement
assignees: ""
---

**What problem does this solve?**
The concrete situation this would help with -- not just "it would be
nice to have X," but what you were actually trying to do when you
noticed X was missing.

**Proposed approach**
If you have one in mind. It's fine to leave this blank and just describe
the problem -- but a concrete proposal (even a rough one) is what turns
a wishlist item into something reviewable.

**Alternatives considered**
Anything you tried instead, including workarounds, and why they weren't
good enough.

**Scope check**
This project's stated design constraints (see `docs/architecture.md`):
CPU-only, no PyTorch anywhere in the stack, local-first (no telemetry, no
required cloud dependency). A request that needs a GPU or a cloud API by
design is likely out of scope -- say so upfront if you're already aware
of the tension, it saves a round-trip.
