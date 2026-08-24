PS D:\_Formal MachineLearning System\ISPCC -- TSA\7. Material Probe PMC\v1> python phosphorus.py
Downloading periodic table → elements_pubchem.csv
Saved.


  Phosphorus Material Classifier — nuclear landmark elements

  ┌─ H    Hydrogen             Z=1     [8 props]
  │  Π (cascade)     0.1800
  │  Ε (equilibrium) 0.4325
  └  Β (boundary)    0.9709

  ┌─ He   Helium               Z=2     [7 props]
  │  Π (cascade)     0.4215
  │  Ε (equilibrium) 0.4787
  └  Β (boundary)    0.5068

  ┌─ B    Boron                Z=5     [9 props]
  │  Π (cascade)     0.6595
  │  Ε (equilibrium) 0.3884
  └  Β (boundary)    0.9846

  ┌─ C    Carbon               Z=6     [9 props]
  │  Π (cascade)     0.6469
  │  Ε (equilibrium) 0.4121
  └  Β (boundary)    0.7137

  ┌─ O    Oxygen               Z=8     [9 props]
  │  Π (cascade)     0.5195
  │  Ε (equilibrium) 0.4427
  └  Β (boundary)    0.7009

  ┌─ Al   Aluminum             Z=13    [9 props]
  │  Π (cascade)     0.2008
  │  Ε (equilibrium) 0.3815
  └  Β (boundary)    0.8560

  ┌─ Fe   Iron                 Z=26    [9 props]
  │  Π (cascade)     -0.1996
  │  Ε (equilibrium) 0.3681
  └  Β (boundary)    0.9103

  ┌─ Ni   Nickel               Z=28    [9 props]
  │  Π (cascade)     -0.3398
  │  Ε (equilibrium) 0.3657
  └  Β (boundary)    0.7990

  ┌─ Zr   Zirconium            Z=40    [9 props]
  │  Π (cascade)     -0.4682
  │  Ε (equilibrium) 0.3834
  └  Β (boundary)    0.9418

  ┌─ Mo   Molybdenum           Z=42    [9 props]
  │  Π (cascade)     -0.6170
  │  Ε (equilibrium) 0.3939
  └  Β (boundary)    0.9303

  ┌─ Gd   Gadolinium           Z=64    [8 props]
  │  Π (cascade)     0.1992
  │  Ε (equilibrium) 0.3795
  └  Β (boundary)    0.9388

  ┌─ W    Tungsten             Z=74    [9 props]
  │  Π (cascade)     -0.1330
  │  Ε (equilibrium) 0.4382
  └  Β (boundary)    0.9252

  ┌─ Pb   Lead                 Z=82    [9 props]
  │  Π (cascade)     -0.6854
  │  Ε (equilibrium) 0.3771
  └  Β (boundary)    0.9250

  ┌─ U    Uranium              Z=92    [8 props]
  │  Π (cascade)     0.2041
  │  Ε (equilibrium) 0.3975
  └  Β (boundary)    0.8547


PS D:\_Formal MachineLearning System\ISPCC -- TSA\7. Material Probe PMC\v1> python phosphorus.py --query Fe

  ┌─ Fe   Iron                 Z=26    [9 props]
  │  Π (cascade)     -0.1996
  │  Ε (equilibrium) 0.3681
  └  Β (boundary)    0.9103

  Nearest 5 to Iron (Fe) in TSA space
  ──────────────────────────────────────────────────────────────────
  Sym   Name                 Z        Π        Ε        Β      Dist
  ──────────────────────────────────────────────────────────────────
  Se    Selenium            34  -0.1661  0.3886  0.9031    0.0230
  Pd    Palladium           46  -0.1602  0.3698  0.9296    0.0253
  Ho    Holmium             67  -0.2477  0.3734  0.9232    0.0289
  Mn    Manganese           25  -0.1492  0.3683  0.9299    0.0312
  Ta    Tantalum            73  -0.2086  0.4191  0.9387    0.0341
  ──────────────────────────────────────────────────────────────────

PS D:\_Formal MachineLearning System\ISPCC -- TSA\7. Material Probe PMC\v1> python phosphorus.py --props density=7.87 electronegativity=1.83

  Query fingerprint   Π= N/A    Ε=0.3640  Β= N/A

  Nearest 5 elements to property query
  ──────────────────────────────────────────────────────────────────
  Sym   Name                 Z        Π        Ε        Β      Dist
  ──────────────────────────────────────────────────────────────────
  Cu    Copper              29  -0.4762  0.3643  0.9106    0.0003
  Ni    Nickel              28  -0.3398  0.3657  0.7990    0.0017
  Ag    Silver              47  -0.0334  0.3657  0.8814    0.0018
  Lr    Lawrencium         103  0.1614  0.3663  0.6629    0.0023
  Co    Cobalt              27  -0.3499  0.3677  0.9279    0.0037
  ──────────────────────────────────────────────────────────────────
