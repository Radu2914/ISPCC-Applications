Test data design note

The synthetic test cases (--test sphere/cylinder/box/cone) are unit tests, not physics validation. They confirm the formula is implemented correctly and the code runs without error. They do not confirm that the classification thresholds are physically correct for a given solver.

What is and is not validated:

Claim	Status
Formula implemented as specified	✓ Confirmed by 8/8 per shape
Π-encoding structurally appropriate for curvature-driven refinement	✓ Confirmed by probe RF importance (Π dominant)
Threshold 1.5 (Π/Β boundary) matches solver behaviour	⚠ Not yet validated — engineering rule only
Threshold 0.60 (Β/Ε boundary) matches solver behaviour	⚠ Not yet validated — engineering rule only
Test parameters chosen independently of the formula	✗ Parameters were derived from the formula — circular by construction

The last point is intentional for a unit test (parameters must be consistent with the expected labels) but means the test cannot stand alone as physical evidence. Threshold validation requires running the same zones through a reference mesher and comparing where it places refinement boundaries.

Zone 3 (cone mid) — margin note:
edge_length_min for this zone was set to 1.2mm (full score ≈ 0.87, margin 0.27 from the Β/Ε threshold). An earlier version used 0.75mm, giving score 0.603 and margin 0.003 — a near-threshold value flagged as reverse-engineered. The current value reflects a physically reasonable element size gradient from tip (h=0.1mm) to base (h=3.0mm) and gives a robust margin. If thresholds are adjusted during solver validation, zone 3 is the first case to recheck.