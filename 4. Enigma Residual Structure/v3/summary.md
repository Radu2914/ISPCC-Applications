What the probe actually did, stated without inflation:

It was given 456,976 (position, character) pairs from a known Enigma key. It was given features encoding position as π-type and character as e-type. It was asked to predict three targets. On the German frequency target it scored R² = 1.000 with e-features and -0.085 with π-features.

That is impressive, but it recovered a structural residual it was partially handed. The plugboard partner (e_plug_part) was an explicit feature in the encoding. German frequency was the target by design. The probe found that e_plug_part is the most important feature for predicting German frequency — which is correct, because PLUG[inp] directly determines the frequency value. The probe confirmed the structure rather than discovered it blind.

What it did NOT do:

It did not break Enigma. It did not recover the plugboard pairs from ciphertext alone. It did not identify which cipher characters are plugboard partners without being told the key. The plugboard partner mapping was given to the probe as a feature. In a real cryptanalytic scenario, that mapping is exactly what you're trying to find.

What it DID do that is genuinely new:

Three things that are not trivial.

First, it correctly classified the structural type of each residual without being told what type to look for. Given three targets simultaneously — German frequency, rotor position, substitution offset — it returned 225×, 322×, and near-zero separations respectively, all correct, all in the right direction. A blind structural classifier that works across cryptographic and non-cryptographic domains with that separation ratio is not a standard result.

Second, it correctly identified the importance ordering within the e-type features. e_plug_part ranked above e_char. This is the correct signal path: PLUG[inp] enters the rotors, not inp itself. The probe recovered the Enigma architecture from importance ranking without being told the signal flows plugboard-first. This is the closest thing to genuine structural discovery in the results.

Third, and most important for cryptography specifically: the substitution offset result is a correct null. The probe correctly refused to find single-axis structure where none exists by design. This is what distinguishes ISPCC from a feature engineering trick. A method that only finds structure when structure is there is more useful for cryptography than one that finds patterns everywhere, because false structure in cryptanalysis leads you to wrong key candidates. The reliable null is the property that makes this useful.

Where ISPCC could genuinely contribute to cryptography:

Not to breaking ciphers with known keys. That is already solved. The contribution is in a different direction.

Side-channel and implementation analysis. Real cipher implementations leak information through timing, power consumption, or cache behavior. These leakages have structural character — some are periodic (clock-dependent, π-type), some are bounded and self-regulating (data-dependent, e-type). The ISPCC probe applied to side-channel traces would automatically classify whether a leakage is timing-structural (π) or data-structural (e) without the analyst needing to hypothesize the source. That classification directs the attack. Current methods require human judgment to decide which hypothesis to pursue first.

Distinguisher analysis. A good cipher should produce output that is structurally indistinguishable from random. The substitution offset result showed the probe correctly identifies this indistinguishability (null result, both axes near zero). Applied to a cipher with a structural weakness — a biased output distribution — the probe would detect which axis carries the bias before a human analyst could formulate the right statistical test. This is a genuine automation of the first step of distinguisher attacks.

Known-plaintext structure recovery. The v3 result showed that given known plaintext positions and the right target formulation, the probe recovers which encoding axis carries the exploitable structure. In a real known-plaintext scenario you have the pairs but not the key. The probe would tell you whether to pursue frequency analysis (e-type target, as in Turing's case) or period analysis (π-type target, as in index of coincidence attacks) without having to run both exhaustively.

The honest summary:

ISPCC recovered the structural residual of Enigma correctly and automatically, but it was given the key to do it. What it proved is that the method correctly types the structure when the structure exists, correctly refuses to type it when it doesn't, and correctly orders the features by their architectural role in the signal path. For cryptography, that makes it a structural diagnostic tool, not a cryptanalytic one. The distinction matters. A diagnostic that tells you what kind of structure to look for, and confirms when you've found it versus when you're chasing noise, is valuable precisely because current cryptanalysis relies heavily on human judgment to make those calls. ISPCC would automate the structural classification step that currently precedes any specific attack.

That is a real contribution. It is not Turing's contribution. Turing found the structure. ISPCC would tell you, reliably and automatically, which type of structure to look for before you start.