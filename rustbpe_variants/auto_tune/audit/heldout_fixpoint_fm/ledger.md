# auto_tune held-out-Pareto FIXPOINT ledger (corpus-cached)

- held-out shards: 10 (10,000,000 chars each), avg-gated
- gate: held-out Pareto — cov≥0 AND comp≤0 AND dead≤0 AND ≥1 strict improvement (coverage-up OR coverage-flat comp/dead win) + dead-tolerance 1.0 (Δcov≥0,Δcomp<0,Δdead≤1.0, committed after free keeps) + comp-tolerance 3.0 (Δcov>0,Δcomp≤3.0,Δdead≤1.0, committed after free keeps); coverage prioritized; recheck ALL + recycle non-earners every round; eject-cooldown base 4 (k-th jail event benches a pair 4*2^(k-1) rounds; early release on a 0-keep round); transactional net-commit (trial → settle ejections → net gate; net-negative trials vetoed + jailed)
- candidates: 0

## running tally
- status **running** | rounds 255 (commits 128, cleanups 2, pairs recycled 2, vetoes 127) | pool 0 | kept **126**
- coverage **18134** (base 13944 / infl 4190)  ·  Δ vs baseline +52
- compression avg/shard **1991455**  ·  Δ vs baseline -1152.6
- dead avg/shard **1145.3**  ·  Δ vs baseline -49.5

## rounds (one commit each; pool re-evaluated vs the new reference)
| round | pool | keeps | committed | Δcov | Δcomp | Δdead |
|---|---|---|---|---|---|---|
| 1 | 18171 | 3374 | `' inv'+'ol'` | 4 | -5.5 | -0.7 |
| 2 | 18171 | 2370 | `' inc'+'re'` | 5 | -25.5 | -0.3 |
| 3 | 18162 | 3381 | `'f'+'ter'` | 5 | -11.9 | -1.5 |
| 4 | 1 | 1 | `' Man'+'ufact'` | 2 | -0.4 | 0.2 |
| 5 | 18162 | 3270 | `' pro'+'du'` | 2 | -32.1 | -1.4 |
| 6 | 18160 | 2179 | `'ah'+'u'` | 1 | -5.2 | 0.0 |
| 7 | 18158 | 2177 | `' contin'+'u'` | 1 | -5.7 | -0.1 |
| 8 | 9 | 9 | `' imm'+'edi'` | 1 | -13.7 | -1.2 |
| 9 | 8 | 8 | `'....'+'....'` | 2 | -4.9 | -1.3 |
| 10 | 7 | 7 | `' diff'+'ere'` | 1 | -15.7 | -0.6 |
| 11 | 6 | 6 | `' tra'+'um'` | 1 | -8.8 | -0.2 |
| 12 | 21 | 21 | `'pe'+'ed'` | 2 | -1.0 | 0.1 |
| 13 | 18 | 14 | `'I'+'H'` | 1 | -7.2 | 0.0 |
| 14 | 1 | 1 | `' ag'+'ricult'` | 1 | -6.9 | -0.8 |
| 15 | 1 | 1 | `' antib'+'iot'` | 1 | -4.0 | -0.2 |
| 16 | 1 | 1 | `' Ind'+'ivid'` | 0 | -5.2 | -0.5 |
| 17 | 1 | 1 | `' Pro'+'ble'` | 0 | -3.9 | -0.7 |
| 18 | 1 | 1 | `' Te'+'ac'` | 0 | -4.8 | -0.8 |
| 19 | 1 | 1 | `' ass'+'um'` | 0 | -4.2 | -0.8 |
| 20 | 1 | 1 | `' P'+'upp'` | 0 | -3.8 | -0.2 |
| 21 | 1 | 1 | `' D'+'ise'` | 0 | -3.0 | -0.5 |
| 22 | 1 | 1 | `' F'+'if'` | 0 | -2.7 | 0.0 |
| 23 | 1 | 1 | `' ant'+'ip'` | 0 | -2.3 | 0.0 |
| 24 | 1 | 1 | `' V'+'ac'` | 0 | -2.3 | 0.1 |
| 25 | 1 | 1 | `' acc'+'ur'` | 0 | -2.0 | -0.8 |
| 26 | 1 | 1 | `' m'+'ov'` | 2 | -9.6 | -0.6 |
| 27 | 1 | 1 | `'·'+'·'` | 1 | -9.6 | -1.9 |
| 28 | 1 | 1 | `' inc'+'lud'` | 2 | -9.5 | -0.6 |
| 29 | 1 | 1 | `' exper'+'im'` | 0 | -7.2 | -0.5 |
| 30 | 1 | 1 | `'ot'+'ted'` | 1 | -7.6 | -0.5 |
| 31 | 1 | 1 | `'ot'+'ropic'` | 0 | -14.6 | -0.1 |
| 32 | 1 | 1 | `'av'+'ig'` | 0 | -12.5 | 0.6 |
| 33 | 1 | 1 | `'u'+'ild'` | 0 | -11.2 | -0.3 |
| 34 | 1 | 1 | `' w'+'r'` | 0 | -12.6 | 0.2 |
| 35 | 1 | 1 | `' cit'+'iz'` | 0 | -8.1 | -0.9 |
| 36 | 1 | 1 | `' est'+'ab'` | 0 | -8.0 | -0.5 |
| 37 | 1 | 1 | `'th'+'rop'` | 0 | -7.3 | -0.2 |
| 38 | 1 | 1 | `' inst'+'it'` | 0 | -7.0 | -0.4 |
| 39 | 1 | 1 | `'S'+'F'` | 0 | -6.8 | -0.1 |
| 40 | 1 | 1 | `'om'+'ile'` | 0 | -6.8 | -0.1 |
| 41 | 1 | 1 | `' ex'+'ce'` | 0 | -6.5 | -0.7 |
| 42 | 1 | 1 | `'op'+'art'` | 0 | -5.7 | 0.0 |
| 43 | 1 | 1 | `' rep'+'res'` | 0 | -5.4 | -0.7 |
| 44 | 1 | 1 | `' t'+'iss'` | 0 | -5.3 | -1.0 |
| 45 | 1 | 1 | `' reg'+'ul'` | 0 | -5.6 | -1.2 |
| 46 | 1 | 1 | `' vol'+'un'` | 0 | -7.5 | -0.8 |
| 47 | 1 | 1 | `' encou'+'rag'` | 0 | -5.0 | -1.9 |
| 48 | 1 | 1 | `' cont'+'roll'` | 0 | -5.0 | 0.0 |
| 49 | 1 | 1 | `'ar'+'as'` — −1 ejected, net 0/-4.7/0.0 | 0 | -4.7 | 0.0 |
| 50 | 1 | 1 | `' prov'+'id'` | 0 | -3.5 | 0.3 |
| 51 | 1 | 1 | `','+'00'` — −1 ejected, net 0/-505.5/-1.8 | 1 | -506.3 | -0.4 |
| 52 | 1 | 1 | `'on'+'el'` | 1 | -1.4 | -0.3 |
| 53 | 1 | 1 | `' b'+'reat'` | 1 | -1.1 | -0.3 |
| 54 | 1 | 1 | `'el'+'ry'` | 1 | -0.6 | -0.1 |
| 55 | 1 | 1 | `'ther'+'net'` | 1 | -1.8 | -0.3 |
| 56 | 1 | 1 | `'ress'+'ions'` | 1 | -1.8 | -0.1 |
| 57 | 1 | 1 | `'ign'+'ing'` | 1 | -0.4 | 0.0 |
| 58 | 1 | 1 | `'n'+'ate'` | 1 | 1.2 | 0.0 |
| 59 | 1 | 1 | `'ang'+'ar'` | 2 | -1.5 | -0.6 |
| 60 | 18151 | 1 | `' '+'\uf0b7'` | 0 | -2.3 | -0.3 |
| 61 | 18151 | 1 | `' Ad'+'vis'` | 0 | -1.2 | 0.0 |
| 62 | 18151 | 1 | `' An'+'alog'` | 0 | -3.7 | -0.1 |
| 63 | 18151 | 1 | `' An'+'im'` | 0 | -1.0 | 0.0 |
| 64 | 18151 | 1 | `' >'+'>'` | 0 | -0.9 | 0.0 |
| 65 | 18151 | 1 | `' A'+'AA'` | 0 | -1.1 | -0.1 |
| 66 | 18151 | 1 | `' Air'+'bus'` | 0 | -2.5 | 0.0 |
| 67 | 1 | 1 | `'n'+'iv'` | 0 | -12.4 | -0.9 |
| 68 | 1 | 1 | `'por'+'ary'` | 0 | -10.5 | -0.4 |
| 69 | 1 | 1 | `'x'+'im'` | 0 | -9.2 | 0.8 |
| 70 | 1 | 1 | `'ec'+'ause'` | 0 | -6.0 | -0.4 |
| 71 | 1 | 1 | `' desc'+'rib'` | 0 | -7.7 | -0.7 |
| 72 | 1 | 1 | `' ele'+'ph'` | 0 | -6.8 | -1.5 |
| 73 | 1 | 1 | `'auc'+'oma'` | 0 | -4.0 | 0.2 |
| 74 | 1 | 1 | `' colle'+'ag'` | 0 | -6.8 | -1.0 |
| 75 | 1 | 1 | `' cl'+'us'` | 0 | -4.2 | -0.9 |
| 76 | 1 | 1 | `' d'+'imens'` | 0 | -4.5 | -0.9 |
| 77 | 1 | 1 | `' mat'+'hemat'` | 0 | -6.5 | -0.8 |
| 78 | 1 | 1 | — no commit: vetoed er+ior (net-negative) | - | - | - |
| 79 | 1 | 1 | — no commit: vetoed nder+standing (net-negative) | - | - | - |
| 80 | 1 | 1 | — no commit: vetoed ns+wer (net-negative) | - | - | - |
| 81 | 1 | 1 | — no commit: vetoed  vers+at (net-negative) | - | - | - |
| 82 | 1 | 1 | — no commit: vetoed ct+ure (net-negative) | - | - | - |
| 83 | 1 | 1 | — no commit: vetoed  comp+on (net-negative) | - | - | - |
| 84 | 1 | 1 | — no commit: vetoed cc+ording (net-negative) | - | - | - |
| 85 | 1 | 1 | — no commit: vetoed ild+ren (net-negative) | - | - | - |
| 86 | 1 | 1 | — no commit: vetoed  ve+h (net-negative) | - | - | - |
| 87 | 1 | 1 | — no commit: vetoed  un+us (net-negative) | - | - | - |
| 88 | 1 | 1 | — no commit: vetoed  res+our (net-negative) | - | - | - |
| 89 | 1 | 1 | — no commit: vetoed  di+ag (net-negative) | - | - | - |
| 90 | 1 | 1 | — no commit: vetoed  f+ert (net-negative) | - | - | - |
| 91 | 1 | 1 | — no commit: vetoed urther+more (net-negative) | - | - | - |
| 92 | 1 | 1 | — no commit: vetoed erson+al (net-negative) | - | - | - |
| 93 | 1 | 1 | — no commit: vetoed  streng+the (net-negative) | - | - | - |
| 94 | 1 | 1 | — no commit: vetoed arent+s (net-negative) | - | - | - |
| 95 | 1 | 1 | — no commit: vetoed  f+oss (net-negative) | - | - | - |
| 96 | 1 | 1 | — no commit: vetoed  v+ib (net-negative) | - | - | - |
| 97 | 1 | 1 | — no commit: vetoed  comp+at (net-negative) | - | - | - |
| 98 | 1 | 1 | — no commit: vetoed yp+es (net-negative) | - | - | - |
| 99 | 1 | 1 | — no commit: vetoed  prom+ot (net-negative) | - | - | - |
| 100 | 1 | 1 | — no commit: vetoed alth+y (net-negative) | - | - | - |
| 101 | 1 | 1 | — no commit: vetoed  en+ab (net-negative) | - | - | - |
| 102 | 1 | 1 | — no commit: vetoed  sign+ific (net-negative) | - | - | - |
| 103 | 1 | 1 | — no commit: vetoed  al+umin (net-negative) | - | - | - |
| 104 | 1 | 1 | — no commit: vetoed per+ature (net-negative) | - | - | - |
| 105 | 1 | 1 | `' sh'+'ap'` | 0 | -5.1 | 0.4 |
| 106 | 1 | 1 | `' c'+'ateg'` | 0 | -4.4 | -0.2 |
| 107 | 1 | 1 | `' diss'+'oci'` | 0 | -4.2 | 0.0 |
| 108 | 1 | 1 | `'du'+'ate'` | 0 | -4.2 | -0.4 |
| 109 | 1 | 1 | `'nov'+'ation'` | 0 | -4.2 | 0.0 |
| 110 | 1 | 1 | `' ch'+'ann'` | 0 | -4.1 | 0.1 |
| 111 | 1 | 1 | `' influ'+'en'` | 0 | -4.0 | -0.7 |
| 112 | 1 | 1 | `'y'+'pe'` | 0 | -4.3 | -0.2 |
| 113 | 1 | 1 | `'ak'+'u'` | 0 | -4.1 | 0.2 |
| 114 | 1 | 1 | `'e'+'ah'` | 0 | -4.0 | -0.9 |
| 115 | 1 | 1 | `' es'+'oph'` | 0 | -3.9 | 0.5 |
| 116 | 1 | 1 | `'N'+'OT'` | 0 | -3.9 | -0.1 |
| 117 | 1 | 1 | `' se'+'arc'` | 0 | -3.7 | -0.6 |
| 118 | 1 | 1 | `' dis'+'pl'` | 0 | -3.6 | 0.0 |
| 119 | 1 | 1 | `' pl'+'aus'` | 0 | -3.6 | 0.1 |
| 120 | 1 | 1 | `'ric'+'ulum'` | 0 | -3.5 | 0.0 |
| 121 | 1 | 1 | `' out'+'bre'` | 0 | -3.4 | -0.9 |
| 122 | 1 | 1 | `' port'+'ra'` | 0 | -3.4 | -0.1 |
| 123 | 1 | 1 | `' sub'+'sequ'` | 0 | -3.4 | -0.8 |
| 124 | 1 | 1 | — no commit: vetoed  vers+at (net-negative) | - | - | - |
| 125 | 1 | 1 | — no commit: vetoed ct+ure (net-negative) | - | - | - |
| 126 | 1 | 1 | — no commit: vetoed er+ior (net-negative) | - | - | - |
| 127 | 1 | 1 | — no commit: vetoed nder+standing (net-negative) | - | - | - |
| 128 | 1 | 1 | — no commit: vetoed ns+wer (net-negative) | - | - | - |
| 129 | 1 | 1 | `' Int'+'ellig'` | 0 | -3.2 | -0.5 |
| 130 | 1 | 1 | `' bi'+'op'` | 0 | -3.2 | -0.2 |
| 131 | 1 | 1 | `' c'+'igaret'` | 0 | -3.2 | 0.0 |
| 132 | 1 | 1 | — no commit: vetoed  comp+on (net-negative) | - | - | - |
| 133 | 1 | 1 | `' pers'+'pect'` | 0 | -3.2 | -0.8 |
| 134 | 1 | 1 | — no commit: vetoed  d+ire (net-negative) | - | - | - |
| 135 | 1 | 1 | — no commit: vetoed  em+ot (net-negative) | - | - | - |
| 136 | 1 | 1 | — no commit: vetoed  g+ir (net-negative) | - | - | - |
| 137 | 1 | 1 | — no commit: vetoed ****+**** (net-negative) | - | - | - |
| 138 | 1 | 1 | `'an'+'ish'` | 1 | -9.1 | -0.2 |
| 139 | 1 | 1 | — no commit: vetoed ====+==== (net-negative) | - | - | - |
| 140 | 1 | 1 | — no commit: vetoed pe+ed (net-negative) | - | - | - |
| 141 | 1 | 1 | — no commit: vetoed pe+ar (net-negative) | - | - | - |
| 142 | 1 | 1 | — no commit: vetoed --------+---- (net-negative) | - | - | - |
| 143 | 1 | 1 | — no commit: vetoed  re+se (net-negative) | - | - | - |
| 144 | 1 | 1 | — no commit: vetoed **+** (net-negative) | - | - | - |
| 145 | 1 | 1 | — no commit: vetoed ut+her (net-negative) | - | - | - |
| 146 | 1 | 1 | — no commit: vetoed  ra+b (net-negative) | - | - | - |
| 147 | 1 | 1 | — no commit: vetoed  ing+red (net-negative) | - | - | - |
| 148 | 1 | 1 | — no commit: vetoed ==+== (net-negative) | - | - | - |
| 149 | 1 | 1 | — no commit: vetoed ol+o (net-negative) | - | - | - |
| 150 | 1 | 1 | — no commit: vetoed  vers+at (net-negative) | - | - | - |
| 151 | 1 | 1 | — no commit: vetoed ct+ure (net-negative) | - | - | - |
| 152 | 1 | 1 | — no commit: vetoed  un+us (net-negative) | - | - | - |
| 153 | 1 | 1 | — no commit: vetoed  ve+h (net-negative) | - | - | - |
| 154 | 1 | 1 | — no commit: vetoed cc+ording (net-negative) | - | - | - |
| 155 | 1 | 1 | — no commit: vetoed ild+ren (net-negative) | - | - | - |
| 156 | 1 | 1 | — no commit: vetoed  di+ag (net-negative) | - | - | - |
| 157 | 1 | 1 | — no commit: vetoed  res+our (net-negative) | - | - | - |
| 158 | 1 | 1 | — no commit: vetoed ion+ally (net-negative) | - | - | - |
| 159 | 1 | 1 | — no commit: vetoed  f+ert (net-negative) | - | - | - |
| 160 | 1 | 1 | — no commit: vetoed urther+more (net-negative) | - | - | - |
| 161 | 1 | 1 | — no commit: vetoed erson+al (net-negative) | - | - | - |
| 162 | 1 | 1 | — no commit: vetoed ——+—— (net-negative) | - | - | - |
| 163 | 1 | 1 | — no commit: vetoed  streng+the (net-negative) | - | - | - |
| 164 | 1 | 1 | — no commit: vetoed nd+om (net-negative) | - | - | - |
| 165 | 1 | 1 | — no commit: vetoed arent+s (net-negative) | - | - | - |
| 166 | 1 | 1 | — no commit: vetoed  f+oss (net-negative) | - | - | - |
| 167 | 1 | 1 | — no commit: vetoed  v+ib (net-negative) | - | - | - |
| 168 | 1 | 1 | — no commit: vetoed ……+…… (net-negative) | - | - | - |
| 169 | 1 | 1 | — no commit: vetoed  comp+at (net-negative) | - | - | - |
| 170 | 1 | 1 | — no commit: vetoed  cry+st (net-negative) | - | - | - |
| 171 | 1 | 1 | — no commit: vetoed  prom+ot (net-negative) | - | - | - |
| 172 | 1 | 1 | — no commit: vetoed alth+y (net-negative) | - | - | - |
| 173 | 1 | 1 | — no commit: vetoed yp+es (net-negative) | - | - | - |
| 174 | 1 | 1 | — no commit: vetoed  en+ab (net-negative) | - | - | - |
| 175 | 1 | 1 | — no commit: vetoed  sign+ific (net-negative) | - | - | - |
| 176 | 1 | 1 | — no commit: vetoed  al+umin (net-negative) | - | - | - |
| 177 | 1 | 1 | — no commit: vetoed ore+r (net-negative) | - | - | - |
| 178 | 1 | 1 | — no commit: vetoed ody+nam (net-negative) | - | - | - |
| 179 | 1 | 1 | — no commit: vetoed on+in (net-negative) | - | - | - |
| 180 | 1 | 1 | — no commit: vetoed op+uses (net-negative) | - | - | - |
| 181 | 1 | 1 | — no commit: vetoed  art+is (net-negative) | - | - | - |
| 182 | 1 | 1 | — no commit: vetoed  st+ru (net-negative) | - | - | - |
| 183 | 1 | 1 | — no commit: vetoed pt+oms (net-negative) | - | - | - |
| 184 | 1 | 1 | — no commit: vetoed  L+D (net-negative) | - | - | - |
| 185 | 1 | 1 | — no commit: vetoed  capt+iv (net-negative) | - | - | - |
| 186 | 1 | 1 | — no commit: vetoed  satis+fact (net-negative) | - | - | - |
| 187 | 1 | 1 | — no commit: vetoed  te+ac (net-negative) | - | - | - |
| 188 | 1 | 1 | — no commit: vetoed ********+******** (net-negative) | - | - | - |
| 189 | 1 | 1 | — no commit: vetoed  ch+all (net-negative) | - | - | - |
| 190 | 1 | 1 | `' const'+'ra'` | 0 | -3.2 | -0.6 |
| 191 | 1 | 1 | — no commit: vetoed  pot+at (net-negative) | - | - | - |
| 192 | 1 | 1 | `'ic'+'ose'` | 0 | -3.2 | -0.2 |
| 193 | 1 | 1 | `'n'+'om'` | 0 | -3.2 | -0.4 |
| 194 | 1 | 1 | — no commit: vetoed ro+at (net-negative) | - | - | - |
| 195 | 1 | 1 | — no commit: vetoed  PC+OS (net-negative) | - | - | - |
| 196 | 1 | 1 | `' rep'+'ut'` | 0 | -3.1 | -0.4 |
| 197 | 1 | 1 | — no commit: vetoed ========+======== (net-negative) | - | - | - |
| 198 | 1 | 1 | — no commit: vetoed m+ectin (net-negative) | - | - | - |
| 199 | 1 | 1 | `'ns'+'ure'` | 1 | -3.1 | -1.1 |
| 200 | 1 | 1 | — no commit: vetoed  pred+ic (net-negative) | - | - | - |
| 201 | 1 | 1 | — no commit: vetoed  sym+met (net-negative) | - | - | - |
| 202 | 1 | 1 | — no commit: vetoed asc+us (net-negative) | - | - | - |
| 203 | 1 | 1 | — no commit: vetoed erc+ury (net-negative) | - | - | - |
| 204 | 1 | 1 | `'iber'+'ty'` | 1 | -3.0 | -0.5 |
| 205 | 1 | 1 | — no commit: vetoed ist+les (net-negative) | - | - | - |
| 206 | 1 | 1 | `'ol'+'ver'` | 0 | -3.0 | 0.2 |
| 207 | 1 | 1 | — no commit: vetoed  ag+g (net-negative) | - | - | - |
| 208 | 1 | 1 | `' exper'+'ien'` | 0 | -2.9 | -0.8 |
| 209 | 1 | 1 | — no commit: vetoed  occ+as (net-negative) | - | - | - |
| 210 | 1 | 1 | `' rein'+'for'` | 0 | -2.9 | -0.4 |
| 211 | 1 | 1 | — no commit: vetoed _+id (net-negative) | - | - | - |
| 212 | 1 | 1 | `'ab'+'it'` | 0 | -2.9 | -1.0 |
| 213 | 1 | 1 | `'ab'+'ul'` | 0 | -2.9 | 0.1 |
| 214 | 1 | 1 | `'cc'+'ess'` | 0 | -2.9 | -0.3 |
| 215 | 1 | 1 | — no commit: vetoed d+ent (net-negative) | - | - | - |
| 216 | 1 | 1 | `'meric'+'an'` | 0 | -2.9 | -0.6 |
| 217 | 1 | 1 | — no commit: vetoed ol+om (net-negative) | - | - | - |
| 218 | 1 | 1 | — no commit: vetoed  F+C (net-negative) | - | - | - |
| 219 | 1 | 1 | — no commit: vetoed  pro+ble (net-negative) | - | - | - |
| 220 | 1 | 1 | `' rest'+'ric'` | 0 | -2.8 | 0.0 |
| 221 | 1 | 1 | `'-'+'hyd'` | 0 | -2.8 | -0.1 |
| 222 | 1 | 1 | — no commit: vetoed g+red (net-negative) | - | - | - |
| 223 | 1 | 1 | — no commit: vetoed rew+s (net-negative) | - | - | - |
| 224 | 1 | 1 | `'under'+'st'` | 0 | -2.8 | -0.3 |
| 225 | 1 | 1 | — no commit: vetoed  Ar+men (net-negative) | - | - | - |
| 226 | 1 | 1 | — no commit: vetoed  M+n (net-negative) | - | - | - |
| 227 | 1 | 1 | — no commit: vetoed  int+oler (net-negative) | - | - | - |
| 228 | 1 | 1 | — no commit: vetoed  lic+ens (net-negative) | - | - | - |
| 229 | 1 | 1 | `' p'+'estic'` | 0 | -2.7 | -0.8 |
| 230 | 1 | 1 | `'.'+'in'` | 0 | -2.7 | 0.0 |
| 231 | 1 | 1 | — no commit: vetoed PV+C (net-negative) | - | - | - |
| 232 | 1 | 1 | — no commit: vetoed ang+ut (net-negative) | - | - | - |
| 233 | 1 | 1 | — no commit: vetoed per+ature (net-negative) | - | - | - |
| 234 | 1 | 1 | — no commit: vetoed pr+il (net-negative) | - | - | - |
| 235 | 1 | 1 | `'r'+'ateg'` | 0 | -2.7 | -0.8 |
| 236 | 1 | 1 | `'ros'+'so'` | 1 | -2.7 | -0.2 |
| 237 | 1 | 1 | — no commit: vetoed  Wood+pe (net-negative) | - | - | - |
| 238 | 1 | 1 | — no commit: vetoed  d+ru (net-negative) | - | - | - |
| 239 | 1 | 1 | — no commit: vetoed  flavon+oids (net-negative) | - | - | - |
| 240 | 1 | 1 | `' micro'+'cont'` | 0 | -2.6 | 0.1 |
| 241 | 1 | 1 | — no commit: vetoed -g+rid (net-negative) | - | - | - |
| 242 | 1 | 1 | `'ay'+'enne'` | 1 | -2.6 | 0.1 |
| 243 | 1 | 1 | — no commit: vetoed it+one (net-negative) | - | - | - |
| 244 | 1 | 1 | — no commit: vetoed l+ished (net-negative) | - | - | - |
| 245 | 1 | 1 | `'yth'+'ons'` | 0 | -2.6 | 0.0 |
| 246 | 1 | 1 | `' earth'+'qu'` | 0 | -2.5 | -0.8 |
| 247 | 1 | 1 | — no commit: vetoed  o+me (net-negative) | - | - | - |
| 248 | 1 | 1 | — no commit: vetoed  stand+by (net-negative) | - | - | - |
| 249 | 1 | 1 | `'-b'+'ut'` | 0 | -2.5 | -0.1 |
| 250 | 1 | 1 | `'S'+'ar'` | 0 | -2.5 | -0.1 |
| 251 | 1 | 1 | — no commit: vetoed and+in (net-negative) | - | - | - |
| 252 | 1 | 1 | — no commit: vetoed f+ur (net-negative) | - | - | - |
| 253 | 1 | 1 | — no commit: vetoed ste+ine (net-negative) | - | - | - |
| 254 | 1 | 1 | `'ur'+'is'` | 2 | -2.5 | -0.6 |
| 255 | 1 | 1 | — no commit: vetoed  comp+ut (net-negative) | - | - | - |

## round 256 IN PROGRESS — 18144/0 evaluated, 1344 keep(s) so far (best committed at round end)
| pair | Δcov | Δcomp | Δdead |
|---|---|---|---|
| `'ob+by'` | 3 | -15.1 | 0.0 |
| `'s+hip'` | 2 | -5.6 | 0.3 |
| `'en+ers'` | 2 | -2.4 | -0.9 |
| `'at+omy'` | 2 | -2.3 | 0.0 |
| `'erm+at'` | 2 | -2.1 | -0.5 |
| `'sc+ill'` | 2 | -1.3 | 0.2 |
| `'asp+berry'` | 2 | 0.7 | -0.1 |
| `'umb+ing'` | 2 | 1.0 | -0.4 |
| `'v+ard'` | 2 | 2.2 | 0.7 |
| `'ogene+ous'` | 2 | 2.3 | -0.2 |
| `'erm+is'` | 2 | 2.9 | 0.1 |
| `'pe+ed'` | 1 | -8.8 | 0.6 |
| `'**+**'` | 1 | -5.8 | -0.5 |
| `' ra+b'` | 1 | -5.6 | 0.4 |
| `'==+=='` | 1 | -5.0 | -1.1 |
| `'ol+o'` | 1 | -4.9 | -0.1 |
| `'ol+om'` | 1 | -2.9 | -0.2 |
| `'rew+s'` | 1 | -2.8 | -0.4 |
| `' v+om'` | 1 | -2.3 | -0.2 |
| `'o+am'` | 1 | -2.3 | 0.0 |
| `'ith+ium'` | 1 | -2.1 | -0.2 |
| `' out+l'` | 1 | -2.1 | 0.2 |
| `'ing+o'` | 1 | -2.0 | 0.1 |
| `'ut+her'` | 1 | -1.9 | 0.1 |
| `'w+ed'` | 1 | -1.9 | 0.2 |
| `'az+y'` | 1 | -1.7 | -0.7 |
| `'i+ants'` | 1 | -1.4 | 0.3 |
| `'bs+ite'` | 1 | -1.3 | 0.0 |
| `'is+le'` | 1 | -1.3 | 0.0 |
| `'ars+ightedness'` | 1 | -1.1 | -0.2 |
| `'ou+les'` | 1 | -1.1 | -0.1 |
| `'ax+y'` | 1 | -0.7 | 0.0 |
| `'iod+iversity'` | 1 | -0.6 | 0.1 |
| `'y+al'` | 1 | -0.4 | -1.1 |
| `'s+els'` | 1 | -0.4 | -0.1 |
| `'ct+u'` | 1 | -0.4 | 0.0 |
| `'d+omen'` | 1 | -0.4 | 0.0 |
| `'Pro+t'` | 1 | -0.4 | 0.4 |
| `'ure+au'` | 1 | -0.4 | 0.4 |
| `'ig+ers'` | 1 | -0.3 | -0.3 |
| `'ra+chn'` | 1 | -0.3 | -0.3 |
| `'pect+ing'` | 1 | -0.3 | -0.1 |
| `'ential+s'` | 1 | -0.3 | 0.0 |
| `'ph+os'` | 1 | -0.3 | 0.0 |
| `'unch+ing'` | 1 | -0.3 | 0.0 |
| `'ad+ians'` | 1 | -0.2 | -0.2 |
| `'e+pt'` | 1 | -0.2 | -0.1 |
| `'il+ant'` | 1 | -0.2 | -0.1 |
| `'it+in'` | 1 | -0.2 | 0.1 |
| `'ort+ment'` | 1 | -0.2 | 0.1 |
| `'rament+o'` | 1 | -0.2 | 0.1 |
| `'ard+ens'` | 1 | -0.2 | 0.3 |
| `'ist+ication'` | 1 | -0.1 | -0.1 |
| `'accept+able'` | 1 | -0.1 | 0.0 |
| `'at+re'` | 1 | -0.1 | 0.0 |
| `'ud+son'` | 1 | -0.1 | 0.0 |
| `'ung+ent'` | 1 | -0.1 | 0.0 |
| `'ur+red'` | 1 | -0.1 | 0.0 |
| `'ent+ation'` | 1 | -0.1 | 0.5 |
| `'ick+ness'` | 1 | 0.0 | -0.6 |
| `'asp+berries'` | 1 | 0.0 | -0.3 |
| `'ER+V'` | 1 | 0.0 | 0.0 |
| `'ard+rum'` | 1 | 0.0 | 0.0 |
| `'es+ame'` | 1 | 0.0 | 0.0 |
| `'orn+ings'` | 1 | 0.0 | 0.0 |
| `'fect+ions'` | 1 | 0.0 | 0.2 |
| `'ign+ant'` | 1 | 0.0 | 0.2 |
| `'ned+y'` | 1 | 0.0 | 0.2 |
| `'u+ine'` | 1 | 0.0 | 0.3 |
| `'ad+vant'` | 1 | 0.0 | 0.4 |
| `'ert+il'` | 1 | 0.0 | 0.6 |
| `'ight+ly'` | 1 | 0.0 | 0.9 |
| `'els+h'` | 1 | 0.1 | 0.0 |
| `'ic+ago'` | 1 | 0.1 | 0.0 |
| `'ad+ron'` | 1 | 0.1 | 0.2 |
| `'d+ess'` | 1 | 0.1 | 0.4 |
| `'ort+ium'` | 1 | 0.2 | -0.2 |
| `'ah+l'` | 1 | 0.2 | 0.0 |
| `'ar+ns'` | 1 | 0.2 | 0.0 |
| `'ceed+ings'` | 1 | 0.2 | 0.0 |
| `'ley+ball'` | 1 | 0.2 | 0.0 |
| `'te+ins'` | 1 | 0.2 | 0.0 |
| `'agnet+ism'` | 1 | 0.2 | 0.1 |
| `'ennes+see'` | 1 | 0.2 | 0.1 |
| `'es+ar'` | 1 | 0.2 | 0.1 |
| `'ess+ori'` | 1 | 0.2 | 0.1 |
| `'bit+ious'` | 1 | 0.2 | 0.2 |
| `'fect+ious'` | 1 | 0.2 | 0.2 |
| `'rades+h'` | 1 | 0.2 | 0.2 |
| `'iz+arre'` | 1 | 0.3 | 0.0 |
| `'ustom+ed'` | 1 | 0.3 | 0.1 |
| `'ind+eer'` | 1 | 0.3 | 0.2 |
| `'od+ka'` | 1 | 0.3 | 0.4 |
| `'uff+le'` | 1 | 0.4 | -0.1 |
| `'oo+oo'` | 1 | 0.4 | 0.0 |
| `'u+art'` | 1 | 0.4 | 0.0 |
| `'o+ing'` | 1 | 0.4 | 0.1 |
| `'so+ever'` | 1 | 0.4 | 0.4 |
| `'v+ests'` | 1 | 0.4 | 0.4 |
| `'ed+ay'` | 1 | 0.4 | 0.7 |
| `'wards+hip'` | 1 | 0.5 | -0.1 |
| `'iction+ary'` | 1 | 0.5 | 0.1 |
| `'on+ts'` | 1 | 0.5 | 0.1 |
| `'ruct+ions'` | 1 | 0.5 | 0.1 |
| `'ruct+or'` | 1 | 0.5 | 0.1 |
| `'en+ment'` | 1 | 0.5 | 0.2 |
| `'pend+icular'` | 1 | 0.5 | 0.3 |
| `'ert+ation'` | 1 | 0.5 | 0.4 |
| `'ask+et'` | 1 | 0.6 | 0.0 |
| `'ject+ion'` | 1 | 0.6 | 0.0 |
| `'ull+ing'` | 1 | 0.6 | 0.0 |
| `'comfort+able'` | 1 | 0.6 | 0.1 |
| `'ir+ates'` | 1 | 0.6 | 0.1 |
| `'il+ings'` | 1 | 0.6 | 0.2 |
| `'ware+ness'` | 1 | 0.6 | 0.2 |
| `'m+ets'` | 1 | 0.6 | 0.3 |
| `'or+ious'` | 1 | 0.6 | 0.4 |
| `'ver+ages'` | 1 | 0.6 | 0.4 |
| `'ient+ial'` | 1 | 0.6 | 0.6 |
| `'is+dom'` | 1 | 0.7 | -0.5 |
| `'am+en'` | 1 | 0.7 | -0.1 |
| `'pret+ation'` | 1 | 0.7 | 0.0 |
| `'an+ut'` | 1 | 0.7 | 0.1 |
| `'udd+y'` | 1 | 0.7 | 0.1 |
| `'orks+hire'` | 1 | 0.7 | 0.2 |
| `'end+ix'` | 1 | 0.7 | 0.5 |
| `'o+ard'` | 1 | 0.7 | 0.6 |
| `'ft+y'` | 1 | 0.7 | 0.7 |
| `'le+ans'` | 1 | 0.7 | 0.9 |
| `'lect+ions'` | 1 | 0.8 | -0.1 |
| `'ab+ies'` | 1 | 0.8 | 0.0 |
| `'amp+a'` | 1 | 0.8 | 0.0 |
| `'ff+ee'` | 1 | 0.8 | 0.1 |
| `'lish+ing'` | 1 | 0.8 | 0.1 |
| `'und+ry'` | 1 | 0.8 | 0.1 |
| `'zym+atic'` | 1 | 0.8 | 0.1 |
| `'dd+en'` | 1 | 0.8 | 0.6 |
| `'get+table'` | 1 | 0.8 | 0.7 |
| `'aps+es'` | 1 | 0.9 | 0.3 |
| `'arch+ment'` | 1 | 0.9 | 0.3 |
| `'ps+is'` | 1 | 0.9 | 0.4 |
| `'us+cript'` | 1 | 0.9 | 0.4 |
| `'ent+ary'` | 1 | 0.9 | 0.5 |
| `'ides+pread'` | 1 | 0.9 | 0.7 |
| `'ol+esterol'` | 1 | 1.0 | -0.9 |
| `'ever+se'` | 1 | 1.0 | 0.0 |
| `'re+ted'` | 1 | 1.0 | 0.0 |
| `'lud+ge'` | 1 | 1.0 | 0.3 |
| `'il+let'` | 1 | 1.0 | 0.5 |
| `'itzer+land'` | 1 | 1.0 | 0.6 |
| `'is+d'` | 1 | 1.0 | 0.7 |
| `'og+gy'` | 1 | 1.0 | 0.8 |
| `'iss+ipp'` | 1 | 1.1 | -0.1 |
| `'AM+E'` | 1 | 1.1 | 0.0 |
| `'s+ed'` | 1 | 1.1 | 0.1 |
| `'ect+ing'` | 1 | 1.1 | 0.2 |
| `'ES+CO'` | 1 | 1.1 | 0.7 |
| `'ion+als'` | 1 | 1.1 | 0.7 |
| `' nut+ri'` | 1 | 1.2 | -0.6 |
| `'ie+ves'` | 1 | 1.2 | 0.0 |
| `'ol+ing'` | 1 | 1.2 | 0.0 |
| `'ord+on'` | 1 | 1.2 | 0.1 |
| `'id+el'` | 1 | 1.2 | 0.2 |
| `'ud+ding'` | 1 | 1.2 | 0.2 |
| `'ra+cle'` | 1 | 1.2 | 0.4 |
| `'IS+PR'` | 1 | 1.2 | 0.5 |
| `'atter+ies'` | 1 | 1.2 | 0.5 |
| `'en+oid'` | 1 | 1.2 | 0.7 |
| `'er+ness'` | 1 | 1.3 | -0.2 |
| `'ep+hal'` | 1 | 1.3 | 0.0 |
| `'ank+a'` | 1 | 1.3 | 0.1 |
| `'est+ive'` | 1 | 1.3 | 0.2 |
| `'ing+ling'` | 1 | 1.3 | 0.2 |
| `'se+ver'` | 1 | 1.3 | 0.2 |
| `'acks+on'` | 1 | 1.3 | 0.3 |
| `'a+very'` | 1 | 1.3 | 0.6 |
| `'al+and'` | 1 | 1.3 | 0.6 |
| `'OS+E'` | 1 | 1.4 | 0.0 |
| `'ST+M'` | 1 | 1.4 | 0.0 |
| `'row+ning'` | 1 | 1.4 | 0.2 |
| `'hop+pers'` | 1 | 1.4 | 0.3 |
| `'isc+al'` | 1 | 1.4 | 0.3 |
| `'yd+ney'` | 1 | 1.4 | 0.4 |
| `'ver+age'` | 1 | 1.4 | 0.5 |
| `'ro+te'` | 1 | 1.4 | 0.6 |
| `'ount+ains'` | 1 | 1.4 | 0.8 |
| `'ct+uary'` | 1 | 1.5 | -0.4 |
| `'ric+s'` | 1 | 1.5 | -0.1 |
| `'og+an'` | 1 | 1.5 | 0.0 |
| `'pert+ure'` | 1 | 1.5 | 0.0 |
| `'ID+I'` | 1 | 1.5 | 0.2 |
| `'es+pan'` | 1 | 1.5 | 0.2 |
| `'ras+ound'` | 1 | 1.5 | 0.2 |
| `'ay+lor'` | 1 | 1.5 | 0.3 |
| `'j+amin'` | 1 | 1.5 | 0.5 |
| `'allow+een'` | 1 | 1.5 | 0.6 |
| `'sc+ar'` | 1 | 1.6 | 0.0 |
| `'ats+on'` | 1 | 1.6 | 0.7 |
| `'istic+ated'` | 1 | 1.6 | 0.9 |
| `'t+uce'` | 1 | 1.7 | -0.1 |
| `'ucalypt+us'` | 1 | 1.7 | 0.0 |
| `'ames+e'` | 1 | 1.7 | 0.5 |
| `'ier+ra'` | 1 | 1.7 | 0.6 |
| `'p+ic'` | 1 | 1.8 | -0.1 |
| `'oubted+ly'` | 1 | 1.8 | 0.0 |
| `'ad+ata'` | 1 | 1.8 | 0.8 |
| `'red+it'` | 1 | 1.9 | -0.2 |
| `'ill+ation'` | 1 | 1.9 | 0.0 |
| `'du+ino'` | 1 | 1.9 | 0.2 |
| `'ew+able'` | 1 | 1.9 | 0.2 |
| `'ab+us'` | 1 | 1.9 | 0.4 |
| `'els+ius'` | 1 | 1.9 | 0.8 |
| `'im+on'` | 1 | 2.0 | -0.4 |
| `'undred+s'` | 1 | 2.0 | 0.0 |
| `'ing+es'` | 1 | 2.0 | 0.5 |
| `'os+ion'` | 1 | 2.0 | 0.5 |
| `'ap+le'` | 1 | 2.0 | 0.6 |
| `'ah+oo'` | 1 | 2.1 | 0.0 |
| `'ament+als'` | 1 | 2.1 | 0.0 |
| `'d+ies'` | 1 | 2.1 | 0.0 |
| `'os+ers'` | 1 | 2.1 | 0.0 |
| `'tain+ment'` | 1 | 2.1 | 0.0 |
| `'ad+ows'` | 1 | 2.1 | 0.5 |
| `'ock+ets'` | 1 | 2.1 | 0.5 |
| `'w+art'` | 1 | 2.1 | 0.6 |
| `'et+ah'` | 1 | 2.1 | 0.7 |
| `'att+les'` | 1 | 2.2 | -0.8 |
| `'ome+gran'` | 1 | 2.2 | -0.1 |
| `'und+ra'` | 1 | 2.2 | 0.2 |
| `'on+ian'` | 1 | 2.3 | -0.1 |
| `'uff+er'` | 1 | 2.3 | 0.0 |
| `'d+river'` | 1 | 2.3 | 0.2 |
| `'uck+ily'` | 1 | 2.3 | 0.2 |
| `'ours+es'` | 1 | 2.3 | 0.3 |
| `'y+t'` | 1 | 2.4 | -0.2 |
| `'hes+s'` | 1 | 2.4 | 0.0 |
| `'ung+sten'` | 1 | 2.4 | 0.5 |
| `'ist+ure'` | 1 | 2.5 | 0.1 |
| `'k+ets'` | 1 | 2.5 | 0.1 |
| `'che+ster'` | 1 | 2.5 | 0.7 |
| `'aw+ning'` | 1 | 2.6 | 0.3 |
| `'be+it'` | 1 | 2.6 | 0.7 |
| `'ps+um'` | 1 | 2.7 | 0.5 |
| `'ment+ia'` | 1 | 2.7 | 0.6 |
| `'aus+s'` | 1 | 2.7 | 0.8 |
| `'ric+es'` | 1 | 2.7 | 1.0 |
| `'ar+king'` | 1 | 2.8 | -0.2 |
| `'ind+le'` | 1 | 2.8 | 0.0 |
| `'p+us'` | 1 | 2.8 | 0.1 |
| `'a+ign'` | 1 | 2.8 | 0.6 |
| `'resh+old'` | 1 | 2.8 | 0.9 |
| `'ie+w'` | 1 | 2.9 | -0.4 |
| `'au+ge'` | 1 | 2.9 | 0.1 |
| `'t+ight'` | 1 | 2.9 | 0.8 |
| `'art+ments'` | 1 | 2.9 | 1.0 |
| `'e+or'` | 1 | 3.0 | 0.3 |
| `'ri+ors'` | 1 | 3.0 | 1.0 |
| `' d+ire'` | 0 | -18.7 | 0.9 |
| `' em+ot'` | 0 | -14.0 | 0.8 |
| `' g+ir'` | 0 | -12.3 | 0.5 |
| `'nder+standing'` | 0 | -11.4 | -0.7 |
| `'****+****'` | 0 | -9.6 | -0.7 |
| `'ar+ry'` | 0 | -9.3 | -0.3 |
| `'====+===='` | 0 | -9.0 | -1.1 |
| `'ar+ge'` | 0 | -6.9 | -0.3 |
| `'pe+ar'` | 0 | -6.9 | 0.2 |
| `'--------+----'` | 0 | -6.0 | 0.2 |
| `' re+se'` | 0 | -5.8 | -0.3 |
| `' ing+red'` | 0 | -5.2 | -1.4 |
| `'cc+ording'` | 0 | -5.0 | -0.7 |
| `'agon+ist'` | 0 | -4.9 | -0.8 |
| `'er+ior'` | 0 | -4.8 | -0.8 |
| `'ns+wer'` | 0 | -4.8 | -0.8 |
| `' vers+at'` | 0 | -4.8 | -0.7 |
| `'ct+ure'` | 0 | -4.8 | -0.3 |
| `' comp+on'` | 0 | -4.7 | -0.7 |
| `'ild+ren'` | 0 | -4.7 | -0.7 |
| `' ve+h'` | 0 | -4.7 | -0.4 |
| `' un+us'` | 0 | -4.7 | 0.0 |
| `' occ+as'` | 0 | -4.7 | 0.4 |
| `' res+our'` | 0 | -4.6 | -0.1 |
| `' di+ag'` | 0 | -4.6 | 0.2 |
| `'ion+ally'` | 0 | -4.6 | 0.8 |
| `' f+ert'` | 0 | -4.5 | -0.5 |
| `'urther+more'` | 0 | -4.5 | -0.5 |
| `'——+——'` | 0 | -4.4 | -0.6 |
| `'erson+al'` | 0 | -4.4 | 0.1 |
| `' streng+the'` | 0 | -4.2 | -1.0 |
| `'nd+om'` | 0 | -4.2 | 0.2 |
| `'arent+s'` | 0 | -4.1 | -0.4 |
| `' comp+ut'` | 0 | -4.1 | -0.2 |
| `'……+……'` | 0 | -4.0 | -0.8 |
| `' f+oss'` | 0 | -4.0 | -0.4 |
| `' v+ib'` | 0 | -4.0 | 0.1 |
| `' comp+at'` | 0 | -3.9 | -0.2 |
| `'yp+es'` | 0 | -3.9 | -0.2 |
| `' prom+ot'` | 0 | -3.9 | 0.0 |
| `'alth+y'` | 0 | -3.9 | 0.1 |
| `' en+ab'` | 0 | -3.8 | -0.2 |
| `' sign+ific'` | 0 | -3.8 | -0.1 |
| `'ore+r'` | 0 | -3.7 | -0.6 |
| `' al+umin'` | 0 | -3.7 | -0.1 |
| `'on+in'` | 0 | -3.6 | -0.6 |
| `'ody+nam'` | 0 | -3.6 | -0.4 |
| `'op+uses'` | 0 | -3.6 | -0.2 |
| `' st+ru'` | 0 | -3.5 | -0.5 |
| `' art+is'` | 0 | -3.5 | -0.2 |
| `'pt+oms'` | 0 | -3.5 | 1.0 |
| `' NFT+s'` | 0 | -3.4 | -0.6 |
| `' satis+fact'` | 0 | -3.4 | -0.6 |
| `' te+ac'` | 0 | -3.4 | -0.5 |
| `'********+********'` | 0 | -3.4 | -0.5 |
| `' L+D'` | 0 | -3.4 | 0.1 |
| `' capt+iv'` | 0 | -3.4 | 0.1 |
| `' ch+all'` | 0 | -3.3 | -0.4 |
| `'ro+at'` | 0 | -3.2 | -1.0 |
| `' pot+at'` | 0 | -3.2 | -0.5 |
| `'m+ectin'` | 0 | -3.1 | -0.7 |
| `'========+========'` | 0 | -3.1 | -0.6 |
| `' PC+OS'` | 0 | -3.1 | -0.2 |
| `'ist+les'` | 0 | -3.0 | -0.3 |
| `'asc+us'` | 0 | -3.0 | -0.1 |
| `' sym+met'` | 0 | -3.0 | 0.0 |
| `'erc+ury'` | 0 | -3.0 | 0.1 |
| `' pred+ic'` | 0 | -3.0 | 0.2 |
| `' ag+g'` | 0 | -2.9 | -2.1 |
| `'_+id'` | 0 | -2.9 | -0.3 |
| `'d+ent'` | 0 | -2.9 | 0.0 |
| `' pro+ble'` | 0 | -2.8 | -0.7 |
| `'g+red'` | 0 | -2.8 | -0.7 |
| `' F+C'` | 0 | -2.8 | -0.1 |
| `'ang+ut'` | 0 | -2.7 | -0.9 |
| `'per+ature'` | 0 | -2.7 | -0.5 |
| `' M+n'` | 0 | -2.7 | 0.0 |
| `' int+oler'` | 0 | -2.7 | 0.0 |
| `'pr+il'` | 0 | -2.7 | 0.0 |
| `' Ar+men'` | 0 | -2.7 | 0.1 |
| `'PV+C'` | 0 | -2.7 | 0.1 |
| `' lic+ens'` | 0 | -2.7 | 0.5 |
| `'l+ished'` | 0 | -2.6 | -0.6 |
| `' flavon+oids'` | 0 | -2.6 | -0.3 |
| `'it+one'` | 0 | -2.6 | -0.2 |
| `' Wood+pe'` | 0 | -2.6 | -0.1 |
| `' d+ru'` | 0 | -2.6 | -0.1 |
| `'-g+rid'` | 0 | -2.6 | 0.1 |
| `' o+me'` | 0 | -2.5 | 0.0 |
| `' stand+by'` | 0 | -2.5 | 0.1 |
| `' c+ichl'` | 0 | -2.4 | -0.4 |
| `' s+alsa'` | 0 | -2.4 | -0.2 |
| `'l+abel'` | 0 | -2.4 | -0.1 |
| `' S+co'` | 0 | -2.4 | 0.0 |
| `'hy+n'` | 0 | -2.4 | 0.0 |
| `'ix+els'` | 0 | -2.4 | 0.0 |
| `'ob+ot'` | 0 | -2.4 | 0.0 |
| `'ub+arb'` | 0 | -2.4 | 0.0 |
| `' aut+of'` | 0 | -2.4 | 0.1 |
| `'L+oad'` | 0 | -2.4 | 0.1 |
| `'gu+ide'` | 0 | -2.4 | 0.1 |
| `'he+rapy'` | 0 | -2.4 | 0.2 |
| `'ST+EP'` | 0 | -2.3 | -0.1 |
| `' car+bohyd'` | 0 | -2.3 | 0.0 |
| `'Sw+imming'` | 0 | -2.3 | 0.0 |
| `' g+ou'` | 0 | -2.3 | 0.1 |
| `'ophy+tes'` | 0 | -2.3 | 0.1 |
| `'os+quit'` | 0 | -2.3 | 0.1 |
| `'M+ike'` | 0 | -2.3 | 0.2 |
| `'ed+om'` | 0 | -2.3 | 0.2 |
| `'mit+t'` | 0 | -2.3 | 0.2 |
| `' lar+v'` | 0 | -2.2 | -0.5 |
| `' som+et'` | 0 | -2.2 | -0.5 |
| `' th+romb'` | 0 | -2.2 | -0.3 |
| `'-b+and'` | 0 | -2.2 | -0.1 |
| `' Me+yer'` | 0 | -2.2 | 0.0 |
| `'ig+raphy'` | 0 | -2.2 | 0.0 |
| `'ig+ure'` | 0 | -2.2 | 0.0 |
| `' Autom+ation'` | 0 | -2.2 | 0.1 |
| `'A+W'` | 0 | -2.2 | 0.1 |
| `'M+obile'` | 0 | -2.2 | 0.1 |
| `'j+ar'` | 0 | -2.2 | 0.1 |
| `' n+erv'` | 0 | -2.2 | 0.2 |
| `'ber+y'` | 0 | -2.2 | 0.2 |
| `'pro+t'` | 0 | -2.2 | 0.3 |
| `'ount+ry'` | 0 | -2.2 | 0.5 |
| `' P+P'` | 0 | -2.1 | -1.2 |
| `' pre+val'` | 0 | -2.1 | -0.9 |
| `' we+ap'` | 0 | -2.1 | -0.9 |
| `' scen+ari'` | 0 | -2.1 | -0.8 |
| `' mis+under'` | 0 | -2.1 | -0.6 |
| `'o+osing'` | 0 | -2.1 | -0.6 |
| `' int+u'` | 0 | -2.1 | -0.5 |
| `'m+ese'` | 0 | -2.1 | -0.3 |
| `' chick+peas'` | 0 | -2.1 | -0.2 |
| `'w+or'` | 0 | -2.1 | -0.2 |
| `' C+trl'` | 0 | -2.1 | -0.1 |
| `' h+ij'` | 0 | -2.1 | 0.0 |
| `'pt+ember'` | 0 | -2.1 | 0.0 |
| `'w+d'` | 0 | -2.1 | 0.0 |
| `'pt+ide'` | 0 | -2.1 | 0.1 |
| `'s+ky'` | 0 | -2.1 | 0.1 |
| `' G+UI'` | 0 | -2.1 | 0.2 |
| `'est+ic'` | 0 | -2.1 | 0.9 |
| `' comp+an'` | 0 | -2.0 | -1.4 |
| `' sat+ell'` | 0 | -2.0 | -0.8 |
| `' sand+w'` | 0 | -2.0 | -0.3 |
| `' skate+board'` | 0 | -2.0 | -0.3 |
| `'en+b'` | 0 | -2.0 | -0.3 |
| `'o+T'` | 0 | -2.0 | -0.3 |
| `' phot+ograp'` | 0 | -2.0 | -0.1 |
| `'Ch+ocolate'` | 0 | -2.0 | -0.1 |
| `'V+EN'` | 0 | -2.0 | -0.1 |
| `' un+in'` | 0 | -2.0 | 0.0 |
| `'f+ax'` | 0 | -2.0 | 0.0 |
| `'´+s'` | 0 | -2.0 | 0.0 |
| `' V+ib'` | 0 | -2.0 | 0.1 |
| `' con+ven'` | 0 | -2.0 | 0.1 |
| `' cont+rib'` | 0 | -2.0 | 0.1 |
| `'av+id'` | 0 | -2.0 | 0.1 |
| `'c+ure'` | 0 | -2.0 | 0.1 |
| `'ps+om'` | 0 | -2.0 | 0.1 |
| `' st+ro'` | 0 | -2.0 | 0.2 |
| `'chn+ology'` | 0 | -2.0 | 0.2 |
| `'m+ol'` | 0 | -2.0 | 0.2 |
| `'v+ices'` | 0 | -2.0 | 0.2 |
| `'yd+ia'` | 0 | -2.0 | 0.2 |
| `'f+idence'` | 0 | -2.0 | 0.3 |
| `'ac+ent'` | 0 | -2.0 | 0.5 |
| `' av+oc'` | 0 | -1.9 | -0.5 |
| `'ores+is'` | 0 | -1.9 | -0.2 |
| `' Coron+avirus'` | 0 | -1.9 | -0.1 |
| `' V+inc'` | 0 | -1.9 | 0.0 |
| `' var+ieg'` | 0 | -1.9 | 0.0 |
| `'Co+al'` | 0 | -1.9 | 0.0 |
| `'cycl+op'` | 0 | -1.9 | 0.1 |
| `'etch+up'` | 0 | -1.9 | 0.1 |
| `'oc+ado'` | 0 | -1.9 | 0.1 |
| `'re+ous'` | 0 | -1.9 | 0.2 |
| `' h+olid'` | 0 | -1.8 | -0.7 |
| `' ret+ur'` | 0 | -1.8 | -0.7 |
| `'AD+VERTIS'` | 0 | -1.8 | -0.4 |
| `' my+ocard'` | 0 | -1.8 | -0.3 |
| `' mit+ig'` | 0 | -1.8 | -0.1 |
| `' rec+re'` | 0 | -1.8 | -0.1 |
| `' N+W'` | 0 | -1.8 | 0.0 |
| `' chim+ps'` | 0 | -1.8 | 0.0 |
| `'F+ashion'` | 0 | -1.8 | 0.0 |
| `'celer+ation'` | 0 | -1.8 | 0.0 |
| `' be+h'` | 0 | -1.8 | 0.1 |
| `' c+ott'` | 0 | -1.8 | 0.1 |
| `' I+CT'` | 0 | -1.8 | 0.2 |
| `'Sub+st'` | 0 | -1.8 | 0.2 |
| `'ens+ors'` | 0 | -1.8 | 0.2 |
| `'he+ra'` | 0 | -1.8 | 0.2 |
| `'uro+pe'` | 0 | -1.8 | 0.3 |
| `'od+ia'` | 0 | -1.7 | -1.4 |
| `'ur+ized'` | 0 | -1.7 | -0.6 |
| `'ht+ml'` | 0 | -1.7 | -0.4 |
| `'————+————'` | 0 | -1.7 | -0.3 |
| `'Con+crete'` | 0 | -1.7 | -0.2 |
| `'cess+ing'` | 0 | -1.7 | -0.1 |
| `'chlor+ic'` | 0 | -1.7 | -0.1 |
| `' P+LC'` | 0 | -1.7 | 0.0 |
| `' inter+p'` | 0 | -1.7 | 0.0 |
| `' me+as'` | 0 | -1.7 | 0.0 |
| `' M+erc'` | 0 | -1.7 | 0.1 |
| `'Te+achers'` | 0 | -1.7 | 0.1 |
| `'eg+al'` | 0 | -1.7 | 0.1 |
| `'enth+ic'` | 0 | -1.7 | 0.1 |
| `'red+ity'` | 0 | -1.7 | 0.1 |
| `'term+ilk'` | 0 | -1.7 | 0.1 |
| `' preschool+ers'` | 0 | -1.7 | 0.2 |
| `' sign+alling'` | 0 | -1.7 | 0.2 |
| `'-h+aired'` | 0 | -1.7 | 0.2 |
| `'Ins+ert'` | 0 | -1.7 | 0.2 |
| `'ig+u'` | 0 | -1.7 | 0.2 |
| `'oci+ety'` | 0 | -1.7 | 0.2 |
| `'ra+ment'` | 0 | -1.6 | -1.0 |
| `' Ve+h'` | 0 | -1.6 | -0.6 |
| `'…………+…………'` | 0 | -1.6 | -0.5 |
| `' ath+let'` | 0 | -1.6 | -0.4 |
| `' ex+oplan'` | 0 | -1.6 | -0.4 |
| `' ph+r'` | 0 | -1.6 | -0.4 |
| `' trek+king'` | 0 | -1.6 | -0.2 |
| `'Ans+wered'` | 0 | -1.6 | -0.1 |
| `'ient+ed'` | 0 | -1.6 | -0.1 |
| `' :+-'` | 0 | -1.6 | 0.0 |
| `' Aut+hent'` | 0 | -1.6 | 0.0 |
| `' E+OS'` | 0 | -1.6 | 0.0 |
| `' HT+TP'` | 0 | -1.6 | 0.0 |
| `'ant+hem'` | 0 | -1.6 | 0.0 |
| `'eren+cing'` | 0 | -1.6 | 0.0 |
| `'l+ion'` | 0 | -1.6 | 0.0 |
| `' Che+l'` | 0 | -1.6 | 0.1 |
| `' iP+od'` | 0 | -1.6 | 0.1 |
| `'H+G'` | 0 | -1.6 | 0.1 |
| `'in+ol'` | 0 | -1.6 | 0.1 |
| `'oc+oc'` | 0 | -1.6 | 0.1 |
| `' f+ri'` | 0 | -1.6 | 0.2 |
| `'P+aint'` | 0 | -1.6 | 0.2 |
| `'Per+iod'` | 0 | -1.6 | 0.2 |
| `'b+b'` | 0 | -1.6 | 0.2 |
| `'c+i'` | 0 | -1.6 | 0.2 |
| `'u+y'` | 0 | -1.6 | 0.2 |
| `'ear+ch'` | 0 | -1.5 | -2.4 |
| `' fram+ew'` | 0 | -1.5 | -0.7 |
| `' chall+eng'` | 0 | -1.5 | -0.5 |
| `'Intern+et'` | 0 | -1.5 | -0.5 |
| `'asc+ar'` | 0 | -1.5 | -0.4 |
| `' Ether+net'` | 0 | -1.5 | -0.3 |
| `' catast+rop'` | 0 | -1.5 | -0.2 |
| `' an+est'` | 0 | -1.5 | -0.1 |
| `' incon+ven'` | 0 | -1.5 | -0.1 |
| `' Comp+ar'` | 0 | -1.5 | 0.0 |
| `' I+B'` | 0 | -1.5 | 0.0 |
| `' dis+ag'` | 0 | -1.5 | 0.0 |
| `'Up+date'` | 0 | -1.5 | 0.0 |
| `'in+ars'` | 0 | -1.5 | 0.0 |
| `' G+RE'` | 0 | -1.5 | 0.1 |
| `' end+oc'` | 0 | -1.5 | 0.1 |
| `' has+ht'` | 0 | -1.5 | 0.1 |
| `' perm+aculture'` | 0 | -1.5 | 0.1 |
| `' util+ised'` | 0 | -1.5 | 0.1 |
| `'Ext+reme'` | 0 | -1.5 | 0.1 |
| `'M+it'` | 0 | -1.5 | 0.1 |
| `'ate+mal'` | 0 | -1.5 | 0.1 |
| `'c+rum'` | 0 | -1.5 | 0.1 |
| `'chan+ics'` | 0 | -1.5 | 0.1 |
| `'fe+it'` | 0 | -1.5 | 0.1 |
| `'op+hor'` | 0 | -1.5 | 0.1 |
| `' lit+re'` | 0 | -1.5 | 0.2 |
| `'-ch+annel'` | 0 | -1.5 | 0.2 |
| `'B+iology'` | 0 | -1.5 | 0.2 |
| `'C+enter'` | 0 | -1.5 | 0.2 |
| `'de+v'` | 0 | -1.5 | 0.2 |
| `'diff+erence'` | 0 | -1.5 | 0.2 |
| `'op+a'` | 0 | -1.5 | 0.2 |
| `'orm+ally'` | 0 | -1.5 | 0.2 |
| `'ut+ory'` | 0 | -1.5 | 0.2 |
| `' d+at'` | 0 | -1.5 | 0.3 |
| `'pt+om'` | 0 | -1.5 | 0.9 |
| `' h+tt'` | 0 | -1.4 | -0.7 |
| `'out+heast'` | 0 | -1.4 | -0.7 |
| `' P+orsche'` | 0 | -1.4 | -0.6 |
| `'plement+ing'` | 0 | -1.4 | -0.6 |
| `'erg+ed'` | 0 | -1.4 | -0.4 |
| `' gall+bladder'` | 0 | -1.4 | -0.3 |
| `'od+der'` | 0 | -1.4 | -0.3 |
| `' anti+hist'` | 0 | -1.4 | -0.2 |
| `'ant+um'` | 0 | -1.4 | -0.2 |
| `' super+f'` | 0 | -1.4 | 0.0 |
| `'--+\n'` | 0 | -1.4 | 0.0 |
| `'Dep+artment'` | 0 | -1.4 | 0.0 |
| `'M+att'` | 0 | -1.4 | 0.0 |
| `'am+il'` | 0 | -1.4 | 0.0 |
| `'l+ap'` | 0 | -1.4 | 0.0 |
| `'ok+u'` | 0 | -1.4 | 0.0 |
| `' G+D'` | 0 | -1.4 | 0.1 |
| `' I+ber'` | 0 | -1.4 | 0.1 |
| `'-t+raining'` | 0 | -1.4 | 0.1 |
| `'Cal+cium'` | 0 | -1.4 | 0.1 |
| `'Sec+urity'` | 0 | -1.4 | 0.1 |
| `' do+od'` | 0 | -1.4 | 0.2 |
| `'-t+alk'` | 0 | -1.4 | 0.2 |
| `'N+ET'` | 0 | -1.4 | 0.2 |
| `'S+ad'` | 0 | -1.4 | 0.2 |
| `'W+ait'` | 0 | -1.4 | 0.2 |
| `'m+ilk'` | 0 | -1.4 | 0.2 |
| `'med+ia'` | 0 | -1.4 | 0.2 |
| `'oll+o'` | 0 | -1.4 | 0.2 |
| `' Ind+ust'` | 0 | -1.3 | -0.5 |
| `' dat+ab'` | 0 | -1.3 | -0.5 |
| `' eukary+otes'` | 0 | -1.3 | -0.4 |
| `'he+ter'` | 0 | -1.3 | -0.3 |
| `'it+amin'` | 0 | -1.3 | -0.3 |
| `'R+uss'` | 0 | -1.3 | -0.1 |
| `'fol+io'` | 0 | -1.3 | -0.1 |
| `' per+c'` | 0 | -1.3 | 0.0 |
| `'-st+rand'` | 0 | -1.3 | 0.0 |
| `'Ess+ay'` | 0 | -1.3 | 0.0 |
| `'ac+ock'` | 0 | -1.3 | 0.0 |
| `'ll+is'` | 0 | -1.3 | 0.0 |
| `' G+Hz'` | 0 | -1.3 | 0.1 |
| `'$$+\n'` | 0 | -1.3 | 0.1 |
| `'-b+ox'` | 0 | -1.3 | 0.1 |
| `'C+hemistry'` | 0 | -1.3 | 0.1 |
| `'Dr+ive'` | 0 | -1.3 | 0.1 |
| `'est+e'` | 0 | -1.3 | 0.1 |
| `'os+us'` | 0 | -1.3 | 0.1 |
| `'osp+heric'` | 0 | -1.3 | 0.1 |
| `' M+uh'` | 0 | -1.3 | 0.2 |
| `' co+ag'` | 0 | -1.3 | 0.2 |
| `' est+im'` | 0 | -1.3 | 0.2 |
| `' inter+mitt'` | 0 | -1.3 | 0.2 |
| `' star+ving'` | 0 | -1.3 | 0.2 |
| `'-t+aking'` | 0 | -1.3 | 0.2 |
| `'/m+ol'` | 0 | -1.3 | 0.2 |
| `'E+le'` | 0 | -1.3 | 0.2 |
| `'Key+words'` | 0 | -1.3 | 0.2 |
| `'L+ength'` | 0 | -1.3 | 0.2 |
| `'T+rees'` | 0 | -1.3 | 0.2 |
| `'W+aste'` | 0 | -1.3 | 0.2 |
| `'ect+ors'` | 0 | -1.3 | 0.2 |
| `'eth+ical'` | 0 | -1.3 | 0.2 |
| `'n+ames'` | 0 | -1.3 | 0.2 |
| `'ric+ed'` | 0 | -1.3 | 0.2 |
| `'AP+P'` | 0 | -1.2 | -1.1 |
| `' endomet+riosis'` | 0 | -1.2 | -0.8 |
| `' Med+icaid'` | 0 | -1.2 | -0.7 |
| `' a+rachn'` | 0 | -1.2 | -0.6 |
| `' Ec+uador'` | 0 | -1.2 | -0.4 |
| `' p+engu'` | 0 | -1.2 | -0.4 |
| `'ro+foam'` | 0 | -1.2 | -0.4 |
| `'-vent+ilated'` | 0 | -1.2 | -0.3 |
| `'opy+right'` | 0 | -1.2 | -0.2 |
| `'olom+ite'` | 0 | -1.2 | -0.1 |
| `' E+sp'` | 0 | -1.2 | 0.0 |
| `' JP+EG'` | 0 | -1.2 | 0.0 |
| `' cal+ving'` | 0 | -1.2 | 0.0 |
| `'-t+uning'` | 0 | -1.2 | 0.0 |
| `'Mon+itoring'` | 0 | -1.2 | 0.0 |
| `'h+istoric'` | 0 | -1.2 | 0.0 |
| `'il+on'` | 0 | -1.2 | 0.0 |
| `'rom+etry'` | 0 | -1.2 | 0.0 |
| `' D+AC'` | 0 | -1.2 | 0.1 |
| `' Fer+r'` | 0 | -1.2 | 0.1 |
| `' a+xi'` | 0 | -1.2 | 0.1 |
| `' air+space'` | 0 | -1.2 | 0.1 |
| `'M+edia'` | 0 | -1.2 | 0.1 |
| `'arr+hea'` | 0 | -1.2 | 0.1 |
| `'craft+ed'` | 0 | -1.2 | 0.1 |
| `'en+oids'` | 0 | -1.2 | 0.1 |
| `'hol+m'` | 0 | -1.2 | 0.1 |
| `'id+ation'` | 0 | -1.2 | 0.1 |
| `'s+ocial'` | 0 | -1.2 | 0.1 |
| `'ter+ies'` | 0 | -1.2 | 0.1 |
| `' exch+anging'` | 0 | -1.2 | 0.2 |
| `' reg+urg'` | 0 | -1.2 | 0.2 |
| `' tum+ours'` | 0 | -1.2 | 0.2 |
| `'-b+ook'` | 0 | -1.2 | 0.2 |
| `'-p+re'` | 0 | -1.2 | 0.2 |
| `'C+le'` | 0 | -1.2 | 0.2 |
| `'IS+O'` | 0 | -1.2 | 0.2 |
| `'h+om'` | 0 | -1.2 | 0.2 |
| `'in+itions'` | 0 | -1.2 | 0.2 |
| `'ind+ers'` | 0 | -1.2 | 0.2 |
| `'lect+ic'` | 0 | -1.2 | 0.2 |
| `'ring+es'` | 0 | -1.2 | 0.2 |
| `'t+ale'` | 0 | -1.2 | 0.2 |
| `'“+We'` | 0 | -1.2 | 0.2 |
| `' behav+i'` | 0 | -1.1 | -0.7 |
| `'bden+um'` | 0 | -1.1 | -0.6 |
| `' ore+gano'` | 0 | -1.1 | -0.5 |
| `' Y+outube'` | 0 | -1.1 | -0.4 |
| `'ac+ao'` | 0 | -1.1 | -0.4 |
| `' SC+I'` | 0 | -1.1 | -0.3 |
| `' c+ripp'` | 0 | -1.1 | -0.3 |
| `' tele+commun'` | 0 | -1.1 | -0.3 |
| `' dist+ur'` | 0 | -1.1 | -0.1 |
| `'ond+ay'` | 0 | -1.1 | -0.1 |
| `' K+le'` | 0 | -1.1 | 0.0 |
| `' Neuro+science'` | 0 | -1.1 | 0.0 |
| `'-+ring'` | 0 | -1.1 | 0.0 |
| `'Effect+s'` | 0 | -1.1 | 0.0 |
| `'c+ur'` | 0 | -1.1 | 0.0 |
| `'it+tle'` | 0 | -1.1 | 0.0 |
| `'t+an'` | 0 | -1.1 | 0.0 |
| `'t+led'` | 0 | -1.1 | 0.0 |
| `' NC+ERT'` | 0 | -1.1 | 0.1 |
| `' cl+o'` | 0 | -1.1 | 0.1 |
| `' prop+os'` | 0 | -1.1 | 0.1 |
| `'E+lev'` | 0 | -1.1 | 0.1 |
| `'Har+vest'` | 0 | -1.1 | 0.1 |
| `'b+f'` | 0 | -1.1 | 0.1 |
| `' Cal+c'` | 0 | -1.1 | 0.2 |
| `' Recogn+izing'` | 0 | -1.1 | 0.2 |
| `' aut+onom'` | 0 | -1.1 | 0.2 |
| `' end+ors'` | 0 | -1.1 | 0.2 |
| `'CO+VID'` | 0 | -1.1 | 0.2 |
| `'D+en'` | 0 | -1.1 | 0.2 |
| `'F+rank'` | 0 | -1.1 | 0.2 |
| `'Fl+at'` | 0 | -1.1 | 0.2 |
| `'M+ad'` | 0 | -1.1 | 0.2 |
| `'S+em'` | 0 | -1.1 | 0.2 |
| `'St+em'` | 0 | -1.1 | 0.2 |
| `'k+t'` | 0 | -1.1 | 0.2 |
| `'m+eter'` | 0 | -1.1 | 0.2 |
| `'or+f'` | 0 | -1.1 | 0.2 |
| `'r+ion'` | 0 | -1.1 | 0.2 |
| `'v+acc'` | 0 | -1.1 | 0.2 |
| `' emb+ro'` | 0 | -1.1 | 0.4 |
| `'omm+on'` | 0 | -1.1 | 0.7 |
| `' mag+az'` | 0 | -1.0 | -0.7 |
| `'o+very'` | 0 | -1.0 | -0.7 |
| `' circum+st'` | 0 | -1.0 | -0.6 |
| `'ect+ure'` | 0 | -1.0 | -0.4 |
| `' In+vis'` | 0 | -1.0 | -0.2 |
| `' Phil+os'` | 0 | -1.0 | -0.1 |
| `' ph+ag'` | 0 | -1.0 | -0.1 |
| `'og+l'` | 0 | -1.0 | -0.1 |
| `' Mend+el'` | 0 | -1.0 | 0.0 |
| `' V+ul'` | 0 | -1.0 | 0.0 |
| `' h+ous'` | 0 | -1.0 | 0.0 |
| `'ac+hel'` | 0 | -1.0 | 0.0 |
| `'ij+i'` | 0 | -1.0 | 0.0 |
| `'pro+ject'` | 0 | -1.0 | 0.0 |
| `' E+NT'` | 0 | -1.0 | 0.1 |
| `' assemb+l'` | 0 | -1.0 | 0.1 |
| `' cancell+ed'` | 0 | -1.0 | 0.1 |
| `' fo+oth'` | 0 | -1.0 | 0.1 |
| `' ref+eren'` | 0 | -1.0 | 0.1 |
| `'-e+fficiency'` | 0 | -1.0 | 0.1 |
| `'C+ast'` | 0 | -1.0 | 0.1 |
| `'Ind+ex'` | 0 | -1.0 | 0.1 |
| `'M+ade'` | 0 | -1.0 | 0.1 |
| `'US+E'` | 0 | -1.0 | 0.1 |
| `'ang+i'` | 0 | -1.0 | 0.1 |
| `'ip+ot'` | 0 | -1.0 | 0.1 |
| `'izz+ard'` | 0 | -1.0 | 0.1 |
| `'ores+cent'` | 0 | -1.0 | 0.1 |
| `'os+ene'` | 0 | -1.0 | 0.1 |
| `' F+erm'` | 0 | -1.0 | 0.2 |
| `' Fl+av'` | 0 | -1.0 | 0.2 |
| `' Sal+v'` | 0 | -1.0 | 0.2 |
| `' ch+rys'` | 0 | -1.0 | 0.2 |
| `' deb+ugging'` | 0 | -1.0 | 0.2 |
| `' e+book'` | 0 | -1.0 | 0.2 |
| `' over+sh'` | 0 | -1.0 | 0.2 |
| `' sim+ulating'` | 0 | -1.0 | 0.2 |
| `' super+computer'` | 0 | -1.0 | 0.2 |
| `' v+ou'` | 0 | -1.0 | 0.2 |
| `'-+absor'` | 0 | -1.0 | 0.2 |
| `'-s+elling'` | 0 | -1.0 | 0.2 |
| `'-w+in'` | 0 | -1.0 | 0.2 |
| `'OC+K'` | 0 | -1.0 | 0.2 |
| `'OT+O'` | 0 | -1.0 | 0.2 |
| `'R+D'` | 0 | -1.0 | 0.2 |
| `'ant+hus'` | 0 | -1.0 | 0.2 |
| `'b+ane'` | 0 | -1.0 | 0.2 |
| `'um+i'` | 0 | -1.0 | 0.2 |
| `'ve+iling'` | 0 | -1.0 | 0.2 |
| `'í+a'` | 0 | -1.0 | 0.2 |
| `' S+au'` | 0 | -1.0 | 0.3 |
| `' im+bal'` | 0 | -1.0 | 0.3 |
| `'as+ive'` | 0 | -1.0 | 0.4 |
| `'v+id'` | 0 | -1.0 | 0.4 |
| `'vel+and'` | 0 | -1.0 | 0.8 |
| `'IM+P'` | 0 | -0.9 | -0.6 |
| `' reg+en'` | 0 | -0.9 | -0.5 |
| `'in+is'` | 0 | -0.9 | -0.4 |
| `'icular+ly'` | 0 | -0.9 | -0.3 |
| `' antidepress+ants'` | 0 | -0.9 | -0.2 |
| `' b+ip'` | 0 | -0.9 | -0.1 |
| `' ink+jet'` | 0 | -0.9 | -0.1 |
| `'G+Y'` | 0 | -0.9 | -0.1 |
| `' Cro+hn'` | 0 | -0.9 | 0.0 |
| `'ab+e'` | 0 | -0.9 | 0.0 |
| `'oh+ns'` | 0 | -0.9 | 0.0 |
| `' sub+mers'` | 0 | -0.9 | 0.1 |
| `' y+og'` | 0 | -0.9 | 0.1 |
| `'L+M'` | 0 | -0.9 | 0.1 |
| `'N+ormally'` | 0 | -0.9 | 0.1 |
| `'carbon+ate'` | 0 | -0.9 | 0.1 |
| `'gg+ings'` | 0 | -0.9 | 0.1 |
| `'ob+last'` | 0 | -0.9 | 0.1 |
| `'oss+ing'` | 0 | -0.9 | 0.1 |
| `'st+icks'` | 0 | -0.9 | 0.1 |
| `'ug+al'` | 0 | -0.9 | 0.1 |
| `' Comp+an'` | 0 | -0.9 | 0.2 |
| `' am+alg'` | 0 | -0.9 | 0.2 |
| `' batt+ling'` | 0 | -0.9 | 0.2 |
| `' cal+end'` | 0 | -0.9 | 0.2 |
| `' counter+top'` | 0 | -0.9 | 0.2 |
| `' de+act'` | 0 | -0.9 | 0.2 |
| `'-t+arget'` | 0 | -0.9 | 0.2 |
| `'G+row'` | 0 | -0.9 | 0.2 |
| `'M+ini'` | 0 | -0.9 | 0.2 |
| `'c+id'` | 0 | -0.9 | 0.2 |
| `'s+pecial'` | 0 | -0.9 | 0.2 |
| `' My+anmar'` | 0 | -0.8 | -0.4 |
| `'ile+psy'` | 0 | -0.8 | -0.3 |
| `'nd+rome'` | 0 | -0.8 | -0.3 |
| `'ur+ther'` | 0 | -0.8 | -0.2 |
| `' App+lic'` | 0 | -0.8 | -0.1 |
| `' metam+orph'` | 0 | -0.8 | -0.1 |
| `' t+acos'` | 0 | -0.8 | -0.1 |
| `'.+…'` | 0 | -0.8 | -0.1 |
| `'f+ram'` | 0 | -0.8 | -0.1 |
| `' Gl+ac'` | 0 | -0.8 | 0.0 |
| `' ve+g'` | 0 | -0.8 | 0.0 |
| `'c+ats'` | 0 | -0.8 | 0.0 |
| `'le+ge'` | 0 | -0.8 | 0.0 |
| `'ne+xt'` | 0 | -0.8 | 0.0 |
| `'osph+orus'` | 0 | -0.8 | 0.0 |
| `' repro+gram'` | 0 | -0.8 | 0.1 |
| `' sign+age'` | 0 | -0.8 | 0.1 |
| `'-c+ountry'` | 0 | -0.8 | 0.1 |
| `'-m+etal'` | 0 | -0.8 | 0.1 |
| `'M+iss'` | 0 | -0.8 | 0.1 |
| `'________________+________________'` | 0 | -0.8 | 0.1 |
| `'gen+eration'` | 0 | -0.8 | 0.1 |
| `'iz+ards'` | 0 | -0.8 | 0.1 |
| `'n+ose'` | 0 | -0.8 | 0.1 |
| `'stud+ents'` | 0 | -0.8 | 0.1 |
| `' Dem+onst'` | 0 | -0.8 | 0.2 |
| `' house+plant'` | 0 | -0.8 | 0.2 |
| `' pet+ting'` | 0 | -0.8 | 0.2 |
| `' sp+h'` | 0 | -0.8 | 0.2 |
| `'-+right'` | 0 | -0.8 | 0.2 |
| `'-pr+one'` | 0 | -0.8 | 0.2 |
| `'B+ill'` | 0 | -0.8 | 0.2 |
| `'OR+D'` | 0 | -0.8 | 0.2 |
| `'aps+ed'` | 0 | -0.8 | 0.2 |
| `'d+in'` | 0 | -0.8 | 0.2 |
| `'em+en'` | 0 | -0.8 | 0.2 |
| `'en+ia'` | 0 | -0.8 | 0.2 |
| `'ex+c'` | 0 | -0.8 | 0.2 |
| `'ile+t'` | 0 | -0.8 | 0.2 |
| `'it+ize'` | 0 | -0.8 | 0.2 |
| `'m+ans'` | 0 | -0.8 | 0.2 |
| `'m+ight'` | 0 | -0.8 | 0.2 |
| `'re+rs'` | 0 | -0.8 | 0.2 |
| `'semb+ling'` | 0 | -0.8 | 0.2 |
| `'tra+ck'` | 0 | -0.8 | 0.2 |
| `'ud+ers'` | 0 | -0.8 | 0.2 |
| `'w+riter'` | 0 | -0.8 | 0.2 |
| `'ount+ain'` | 0 | -0.8 | 0.3 |
| `'ut+ure'` | 0 | -0.7 | -0.9 |
| `' bypro+ducts'` | 0 | -0.7 | -0.8 |
| `'orpor+ating'` | 0 | -0.7 | -0.6 |
| `'yt+ocin'` | 0 | -0.7 | -0.6 |
| `' mon+it'` | 0 | -0.7 | -0.5 |
| `'ab+et'` | 0 | -0.7 | -0.5 |
| `' Sc+ripps'` | 0 | -0.7 | -0.4 |
| `' Ayurved+a'` | 0 | -0.7 | -0.3 |
| `' dazz+ling'` | 0 | -0.7 | -0.3 |
| `' P+ret'` | 0 | -0.7 | -0.2 |
| `'art+en'` | 0 | -0.7 | -0.2 |
| `'L+AB'` | 0 | -0.7 | -0.1 |
| `'p+ositive'` | 0 | -0.7 | -0.1 |
| `' B+iden'` | 0 | -0.7 | 0.0 |
| `' C+U'` | 0 | -0.7 | 0.0 |
| `' [+...'` | 0 | -0.7 | 0.0 |
| `' enc+oder'` | 0 | -0.7 | 0.0 |
| `' tag+ging'` | 0 | -0.7 | 0.0 |
| `'L+anguage'` | 0 | -0.7 | 0.0 |
| `'as+urable'` | 0 | -0.7 | 0.0 |
| `'in+ner'` | 0 | -0.7 | 0.0 |
| `'ip+hy'` | 0 | -0.7 | 0.0 |
| `' qu+arant'` | 0 | -0.7 | 0.1 |
| `' sym+p'` | 0 | -0.7 | 0.1 |
| `'-f+i'` | 0 | -0.7 | 0.1 |
| `'B+aking'` | 0 | -0.7 | 0.1 |
| `'ed+a'` | 0 | -0.7 | 0.1 |
| `'f+igure'` | 0 | -0.7 | 0.1 |
| `'om+i'` | 0 | -0.7 | 0.1 |
| `'st+ructure'` | 0 | -0.7 | 0.1 |
| `'vid+ia'` | 0 | -0.7 | 0.1 |
| `' initi+ating'` | 0 | -0.7 | 0.2 |
| `' res+iding'` | 0 | -0.7 | 0.2 |
| `' st+ric'` | 0 | -0.7 | 0.2 |
| `'-+going'` | 0 | -0.7 | 0.2 |
| `'-f+ried'` | 0 | -0.7 | 0.2 |
| `'-st+ar'` | 0 | -0.7 | 0.2 |
| `'-st+art'` | 0 | -0.7 | 0.2 |
| `'T+ogether'` | 0 | -0.7 | 0.2 |
| `'bro+ken'` | 0 | -0.7 | 0.2 |
| `'c+ott'` | 0 | -0.7 | 0.2 |
| `'l+ides'` | 0 | -0.7 | 0.2 |
| `'l+ocal'` | 0 | -0.7 | 0.2 |
| `'light+ing'` | 0 | -0.7 | 0.2 |
| `'our+nal'` | 0 | -0.7 | 0.2 |
| `'pract+ice'` | 0 | -0.7 | 0.2 |
| `'press+ure'` | 0 | -0.7 | 0.2 |
| `'represent+ed'` | 0 | -0.7 | 0.2 |
| `'ron+omic'` | 0 | -0.7 | 0.2 |
| `'s+hips'` | 0 | -0.7 | 0.2 |
| `'t+um'` | 0 | -0.7 | 0.2 |
| `' ex+cell'` | 0 | -0.7 | 0.5 |
| `'ustain+ability'` | 0 | -0.6 | -0.7 |
| `' ap+optosis'` | 0 | -0.6 | -0.6 |
| `' inst+rum'` | 0 | -0.6 | -0.4 |
| `' Is+a'` | 0 | -0.6 | -0.3 |
| `' cyl+ind'` | 0 | -0.6 | -0.3 |
| `' mid+w'` | 0 | -0.6 | -0.3 |
| `' seag+rass'` | 0 | -0.6 | -0.3 |
| `'-bl+ooded'` | 0 | -0.6 | -0.3 |
| `'P+ract'` | 0 | -0.6 | -0.2 |
| `'ag+ascar'` | 0 | -0.6 | -0.2 |
| `'b+ps'` | 0 | -0.6 | -0.1 |
| `'C+oc'` | 0 | -0.6 | 0.0 |
| `'Comp+ar'` | 0 | -0.6 | 0.0 |
| `'P+ath'` | 0 | -0.6 | 0.0 |
| `'R+ab'` | 0 | -0.6 | 0.0 |
| `'`+s'` | 0 | -0.6 | 0.0 |
| `'his+m'` | 0 | -0.6 | 0.0 |
| `'ur+ated'` | 0 | -0.6 | 0.0 |
| `' M+K'` | 0 | -0.6 | 0.1 |
| `' P+BS'` | 0 | -0.6 | 0.1 |
| `' P+seud'` | 0 | -0.6 | 0.1 |
| `' T+C'` | 0 | -0.6 | 0.1 |
| `' al+ve'` | 0 | -0.6 | 0.1 |
| `' inter+fer'` | 0 | -0.6 | 0.1 |
| `' inv+as'` | 0 | -0.6 | 0.1 |
| `' pub+l'` | 0 | -0.6 | 0.1 |
| `' snow+fl'` | 0 | -0.6 | 0.1 |
| `'/+('` | 0 | -0.6 | 0.1 |
| `'Diff+iculty'` | 0 | -0.6 | 0.1 |
| `'F+emale'` | 0 | -0.6 | 0.1 |
| `'G+erm'` | 0 | -0.6 | 0.1 |
| `'Mon+th'` | 0 | -0.6 | 0.1 |
| `'gg+le'` | 0 | -0.6 | 0.1 |
| `'ne+al'` | 0 | -0.6 | 0.1 |
| `'ort+ic'` | 0 | -0.6 | 0.1 |
| `'our+ced'` | 0 | -0.6 | 0.1 |
| `' A+ren'` | 0 | -0.6 | 0.2 |
| `' H+amm'` | 0 | -0.6 | 0.2 |
| `' NS+F'` | 0 | -0.6 | 0.2 |
| `' Prep+aring'` | 0 | -0.6 | 0.2 |
| `' fertil+iser'` | 0 | -0.6 | 0.2 |
| `' sur+ve'` | 0 | -0.6 | 0.2 |
| `"'+-"` | 0 | -0.6 | 0.2 |
| `'-d+ec'` | 0 | -0.6 | 0.2 |
| `'Comp+are'` | 0 | -0.6 | 0.2 |
| `'ED+IT'` | 0 | -0.6 | 0.2 |
| `'Em+otional'` | 0 | -0.6 | 0.2 |
| `'IN+S'` | 0 | -0.6 | 0.2 |
| `'Reg+arding'` | 0 | -0.6 | 0.2 |
| `'Te+acher'` | 0 | -0.6 | 0.2 |
| `'W+et'` | 0 | -0.6 | 0.2 |
| `'add+ers'` | 0 | -0.6 | 0.2 |
| `'amb+ia'` | 0 | -0.6 | 0.2 |
| `'at+uring'` | 0 | -0.6 | 0.2 |
| `'az+ard'` | 0 | -0.6 | 0.2 |
| `'el+ope'` | 0 | -0.6 | 0.2 |
| `'ens+en'` | 0 | -0.6 | 0.2 |
| `'fact+s'` | 0 | -0.6 | 0.2 |
| `'or+ide'` | 0 | -0.6 | 0.2 |
| `'s+um'` | 0 | -0.6 | 0.2 |
| `'sor+iasis'` | 0 | -0.6 | 0.2 |
| `'v+ag'` | 0 | -0.6 | 0.2 |
| `'ign+ificant'` | 0 | -0.6 | 0.3 |
| `'attles+n'` | 0 | -0.5 | -1.1 |
| `' ve+gg'` | 0 | -0.5 | -0.9 |
| `' neurotransmit+ters'` | 0 | -0.5 | -0.8 |
| `'ill+us'` | 0 | -0.5 | -0.3 |
| `' Do+ber'` | 0 | -0.5 | -0.2 |
| `' F+em'` | 0 | -0.5 | -0.2 |
| `' Or+nith'` | 0 | -0.5 | -0.2 |
| `' c+ephal'` | 0 | -0.5 | -0.2 |
| `'rec+iation'` | 0 | -0.5 | -0.2 |
| `' analys+ing'` | 0 | -0.5 | -0.1 |
| `' phyl+ogen'` | 0 | -0.5 | -0.1 |
| `'-+imp'` | 0 | -0.5 | -0.1 |
| `'cess+ive'` | 0 | -0.5 | -0.1 |
| `'ur+ple'` | 0 | -0.5 | -0.1 |
| `' Ch+and'` | 0 | -0.5 | 0.0 |
| `' GM+T'` | 0 | -0.5 | 0.0 |
| `' R+NAs'` | 0 | -0.5 | 0.0 |
| `' ____+_'` | 0 | -0.5 | 0.0 |
| `' pr+ud'` | 0 | -0.5 | 0.0 |
| `'AY+S'` | 0 | -0.5 | 0.0 |
| `'Per+formance'` | 0 | -0.5 | 0.0 |
| `'er+on'` | 0 | -0.5 | 0.0 |
| `'ipel+ago'` | 0 | -0.5 | 0.0 |
| `'organ+isms'` | 0 | -0.5 | 0.0 |
| `'ors+et'` | 0 | -0.5 | 0.0 |
| `' *+\n\n'` | 0 | -0.5 | 0.1 |
| `' am+n'` | 0 | -0.5 | 0.1 |
| `' des+p'` | 0 | -0.5 | 0.1 |
| `' pro+kary'` | 0 | -0.5 | 0.1 |
| `' sub+div'` | 0 | -0.5 | 0.1 |
| `'-r+anging'` | 0 | -0.5 | 0.1 |
| `'B+attery'` | 0 | -0.5 | 0.1 |
| `'Ind+ustrial'` | 0 | -0.5 | 0.1 |
| `'P+arent'` | 0 | -0.5 | 0.1 |
| `'R+ose'` | 0 | -0.5 | 0.1 |
| `'Sh+op'` | 0 | -0.5 | 0.1 |
| `'ain+ed'` | 0 | -0.5 | 0.1 |
| `'aly+tic'` | 0 | -0.5 | 0.1 |
| `'at+on'` | 0 | -0.5 | 0.1 |
| `'b+ags'` | 0 | -0.5 | 0.1 |
| `'ert+ile'` | 0 | -0.5 | 0.1 |
| `'ez+ers'` | 0 | -0.5 | 0.1 |
| `'p+ersonal'` | 0 | -0.5 | 0.1 |
| `'un+ky'` | 0 | -0.5 | 0.1 |
| `' C+asc'` | 0 | -0.5 | 0.2 |
| `' H+EL'` | 0 | -0.5 | 0.2 |
| `' ap+ric'` | 0 | -0.5 | 0.2 |
| `' comp+rehens'` | 0 | -0.5 | 0.2 |
| `' el+ast'` | 0 | -0.5 | 0.2 |
| `' ph+arm'` | 0 | -0.5 | 0.2 |
| `' sa+pp'` | 0 | -0.5 | 0.2 |
| `' scrub+bing'` | 0 | -0.5 | 0.2 |
| `' un+belie'` | 0 | -0.5 | 0.2 |
| `'-bl+ue'` | 0 | -0.5 | 0.2 |
| `'-c+ooked'` | 0 | -0.5 | 0.2 |
| `'-y+ellow'` | 0 | -0.5 | 0.2 |
| `'A+ction'` | 0 | -0.5 | 0.2 |
| `'AG+R'` | 0 | -0.5 | 0.2 |
| `'AM+A'` | 0 | -0.5 | 0.2 |
| `'Cong+ratulations'` | 0 | -0.5 | 0.2 |
| `'F+uel'` | 0 | -0.5 | 0.2 |
| `'IF+E'` | 0 | -0.5 | 0.2 |
| `'M+ale'` | 0 | -0.5 | 0.2 |
| `'OS+S'` | 0 | -0.5 | 0.2 |
| `'app+oint'` | 0 | -0.5 | 0.2 |
| `'arm+ed'` | 0 | -0.5 | 0.2 |
| `'c+old'` | 0 | -0.5 | 0.2 |
| `'g+overnmental'` | 0 | -0.5 | 0.2 |
| `'it+ivity'` | 0 | -0.5 | 0.2 |
| `'n+n'` | 0 | -0.5 | 0.2 |
| `'oler+ance'` | 0 | -0.5 | 0.2 |
| `'rom+eter'` | 0 | -0.5 | 0.2 |
| `' hur+d'` | 0 | -0.5 | 0.5 |
| `'o+ar'` | 0 | -0.5 | 0.5 |
| `'aryn+geal'` | 0 | -0.4 | -1.1 |
| `' E+verest'` | 0 | -0.4 | -0.6 |
| `' log+arith'` | 0 | -0.4 | -0.5 |
| `':+||'` | 0 | -0.4 | -0.4 |
| `'V+ERT'` | 0 | -0.4 | -0.3 |
| `' She+pher'` | 0 | -0.4 | -0.2 |
| `' co+val'` | 0 | -0.4 | -0.2 |
| `' ind+ivid'` | 0 | -0.4 | -0.2 |
| `'"+)\n'` | 0 | -0.4 | -0.2 |
| `'gan+o'` | 0 | -0.4 | -0.2 |
| `' U+AE'` | 0 | -0.4 | -0.1 |
| `' inter+v'` | 0 | -0.4 | -0.1 |
| `' vel+oc'` | 0 | -0.4 | -0.1 |
| `'(Phys+Org'` | 0 | -0.4 | -0.1 |
| `' Ch+iropract'` | 0 | -0.4 | 0.0 |
| `' f+if'` | 0 | -0.4 | 0.0 |
| `' wood+pe'` | 0 | -0.4 | 0.0 |
| `'.j+pg'` | 0 | -0.4 | 0.0 |
| `'B+one'` | 0 | -0.4 | 0.0 |
| `'amm+ad'` | 0 | -0.4 | 0.0 |
| `'amps+hire'` | 0 | -0.4 | 0.0 |
| `'asm+ine'` | 0 | -0.4 | 0.0 |
| `'be+y'` | 0 | -0.4 | 0.0 |
| `'il+ateral'` | 0 | -0.4 | 0.0 |
| `'ill+aries'` | 0 | -0.4 | 0.0 |
| `'ro+kes'` | 0 | -0.4 | 0.0 |
| `'rust+ed'` | 0 | -0.4 | 0.0 |
| `'ul+ance'` | 0 | -0.4 | 0.0 |
| `'ulf+ur'` | 0 | -0.4 | 0.0 |
| `' N+T'` | 0 | -0.4 | 0.1 |
| `' Pod+cast'` | 0 | -0.4 | 0.1 |
| `' U+F'` | 0 | -0.4 | 0.1 |
| `' an+em'` | 0 | -0.4 | 0.1 |
| `' download+s'` | 0 | -0.4 | 0.1 |
| `' ha+irst'` | 0 | -0.4 | 0.1 |
| `' microch+ip'` | 0 | -0.4 | 0.1 |
| `'-b+illed'` | 0 | -0.4 | 0.1 |
| `'-c+arb'` | 0 | -0.4 | 0.1 |
| `'-d+irected'` | 0 | -0.4 | 0.1 |
| `'S+ervice'` | 0 | -0.4 | 0.1 |
| `'cycl+ine'` | 0 | -0.4 | 0.1 |
| `'f+itting'` | 0 | -0.4 | 0.1 |
| `'ic+ative'` | 0 | -0.4 | 0.1 |
| `'ive+l'` | 0 | -0.4 | 0.1 |
| `'low+ing'` | 0 | -0.4 | 0.1 |
| `'ot+rophic'` | 0 | -0.4 | 0.1 |
| `'qu+iry'` | 0 | -0.4 | 0.1 |
| `'yn+chron'` | 0 | -0.4 | 0.1 |
| `' B+CE'` | 0 | -0.4 | 0.2 |
| `' Leg+isl'` | 0 | -0.4 | 0.2 |
| `' N+R'` | 0 | -0.4 | 0.2 |
| `' NG+Os'` | 0 | -0.4 | 0.2 |
| `' S+AF'` | 0 | -0.4 | 0.2 |
| `' clog+ging'` | 0 | -0.4 | 0.2 |
| `' in+just'` | 0 | -0.4 | 0.2 |
| `'-c+ore'` | 0 | -0.4 | 0.2 |
| `'-en+h'` | 0 | -0.4 | 0.2 |
| `'B+aby'` | 0 | -0.4 | 0.2 |
| `'B+ottom'` | 0 | -0.4 | 0.2 |
| `'C+aring'` | 0 | -0.4 | 0.2 |
| `'Comp+ared'` | 0 | -0.4 | 0.2 |
| `'N+EW'` | 0 | -0.4 | 0.2 |
| `'N+or'` | 0 | -0.4 | 0.2 |
| `'Orig+inal'` | 0 | -0.4 | 0.2 |
| `'ath+ic'` | 0 | -0.4 | 0.2 |
| `'en+eration'` | 0 | -0.4 | 0.2 |
| `'il+ance'` | 0 | -0.4 | 0.2 |
| `'in+el'` | 0 | -0.4 | 0.2 |
| `'let+ic'` | 0 | -0.4 | 0.2 |
| `'ps+i'` | 0 | -0.4 | 0.2 |
| `'rit+o'` | 0 | -0.4 | 0.2 |
| `'vant+age'` | 0 | -0.4 | 0.2 |
| `' un+for'` | 0 | -0.4 | 0.3 |
| `'ar+ning'` | 0 | -0.4 | 0.4 |
| `'S+che'` | 0 | -0.4 | 0.6 |
| `'n+ters'` | 0 | -0.4 | 0.9 |
| `'AT+ER'` | 0 | -0.3 | -0.7 |
| `'•+•'` | 0 | -0.3 | -0.6 |
| `' sur+pr'` | 0 | -0.3 | -0.3 |
| `'Now+adays'` | 0 | -0.3 | -0.3 |
| `'opt+osis'` | 0 | -0.3 | -0.3 |
| `'v+ada'` | 0 | -0.3 | -0.3 |
| `' The+atre'` | 0 | -0.3 | -0.2 |
| `' ant+ise'` | 0 | -0.3 | -0.2 |
| `'Dire+ctions'` | 0 | -0.3 | -0.2 |
| `'S+ustainability'` | 0 | -0.3 | -0.2 |
| `'ern+ame'` | 0 | -0.3 | -0.2 |
| `' hydrop+ower'` | 0 | -0.3 | -0.1 |
| `'ra+x'` | 0 | -0.3 | -0.1 |
| `'{+{'` | 0 | -0.3 | -0.1 |
| `' er+yth'` | 0 | -0.3 | 0.0 |
| `' rem+ed'` | 0 | -0.3 | 0.0 |
| `'Develop+ment'` | 0 | -0.3 | 0.0 |
| `'H+L'` | 0 | -0.3 | 0.0 |
| `'b+ud'` | 0 | -0.3 | 0.0 |
| `'erent+ial'` | 0 | -0.3 | 0.0 |
| `'is+ite'` | 0 | -0.3 | 0.0 |
| `'l+anguage'` | 0 | -0.3 | 0.0 |
| `'ros+clerosis'` | 0 | -0.3 | 0.0 |
| `' E+ST'` | 0 | -0.3 | 0.1 |
| `' ang+i'` | 0 | -0.3 | 0.1 |
| `' n+arc'` | 0 | -0.3 | 0.1 |
| `' personal+ised'` | 0 | -0.3 | 0.1 |
| `' phot+oc'` | 0 | -0.3 | 0.1 |
| `' ra+v'` | 0 | -0.3 | 0.1 |
| `'-+att'` | 0 | -0.3 | 0.1 |
| `'=+\\'` | 0 | -0.3 | 0.1 |
| `'Encou+rage'` | 0 | -0.3 | 0.1 |
| `'IT+ES'` | 0 | -0.3 | 0.1 |
| `'Main+tenance'` | 0 | -0.3 | 0.1 |
| `'R+R'` | 0 | -0.3 | 0.1 |
| `'ast+es'` | 0 | -0.3 | 0.1 |
| `'ott+on'` | 0 | -0.3 | 0.1 |
| `'ovolta+ic'` | 0 | -0.3 | 0.1 |
| `'pe+re'` | 0 | -0.3 | 0.1 |
| `'ru+ff'` | 0 | -0.3 | 0.1 |
| `' Cont+rib'` | 0 | -0.3 | 0.2 |
| `' D+H'` | 0 | -0.3 | 0.2 |
| `' Ob+st'` | 0 | -0.3 | 0.2 |
| `' an+atom'` | 0 | -0.3 | 0.2 |
| `' dec+oding'` | 0 | -0.3 | 0.2 |
| `' ind+ist'` | 0 | -0.3 | 0.2 |
| `' mother+board'` | 0 | -0.3 | 0.2 |
| `' pin+n'` | 0 | -0.3 | 0.2 |
| `' se+ag'` | 0 | -0.3 | 0.2 |
| `' stabil+izing'` | 0 | -0.3 | 0.2 |
| `'"+-'` | 0 | -0.3 | 0.2 |
| `'-g+ra'` | 0 | -0.3 | 0.2 |
| `'-l+aw'` | 0 | -0.3 | 0.2 |
| `'-l+it'` | 0 | -0.3 | 0.2 |
| `'-t+une'` | 0 | -0.3 | 0.2 |
| `'C+ategories'` | 0 | -0.3 | 0.2 |
| `'Comp+an'` | 0 | -0.3 | 0.2 |
| `'EM+S'` | 0 | -0.3 | 0.2 |
| `'Ear+lier'` | 0 | -0.3 | 0.2 |
| `'G+ather'` | 0 | -0.3 | 0.2 |
| `'L+oss'` | 0 | -0.3 | 0.2 |
| `'N+early'` | 0 | -0.3 | 0.2 |
| `'OU+S'` | 0 | -0.3 | 0.2 |
| `'T+Y'` | 0 | -0.3 | 0.2 |
| `'T+ree'` | 0 | -0.3 | 0.2 |
| `'ah+n'` | 0 | -0.3 | 0.2 |
| `'ls+on'` | 0 | -0.3 | 0.2 |
| `'orn+ed'` | 0 | -0.3 | 0.2 |
| `'our+t'` | 0 | -0.3 | 0.2 |
| `'re+ally'` | 0 | -0.3 | 0.2 |
| `'resp+onse'` | 0 | -0.3 | 0.2 |
| `'ro+ves'` | 0 | -0.3 | 0.2 |
| `'semb+led'` | 0 | -0.3 | 0.2 |
| `'ur+gical'` | 0 | -0.3 | 0.2 |
| `'ut+il'` | 0 | -0.3 | 0.2 |
| `'per+haps'` | 0 | -0.2 | -0.8 |
| `' O+scill'` | 0 | -0.2 | -0.5 |
| `'icult+y'` | 0 | -0.2 | -0.4 |
| `'roubles+hoot'` | 0 | -0.2 | -0.4 |
| `'Ex+cellent'` | 0 | -0.2 | -0.3 |
| `' Tas+man'` | 0 | -0.2 | -0.2 |
| `' polic+ym'` | 0 | -0.2 | -0.2 |
| `'ent+iful'` | 0 | -0.2 | -0.2 |
| `'ist+ur'` | 0 | -0.2 | -0.2 |
| `'pre+ne'` | 0 | -0.2 | -0.2 |
| `' Stre+pt'` | 0 | -0.2 | -0.1 |
| `' ox+al'` | 0 | -0.2 | -0.1 |
| `' to+ile'` | 0 | -0.2 | -0.1 |
| `'ipot+ent'` | 0 | -0.2 | -0.1 |
| `'ol+n'` | 0 | -0.2 | -0.1 |
| `'qu+isite'` | 0 | -0.2 | -0.1 |
| `'uc+ent'` | 0 | -0.2 | -0.1 |
| `' Cl+in'` | 0 | -0.2 | 0.0 |
| `' NI+H'` | 0 | -0.2 | 0.0 |
| `' R+OM'` | 0 | -0.2 | 0.0 |
| `' bio+fuels'` | 0 | -0.2 | 0.0 |
| `' or+ph'` | 0 | -0.2 | 0.0 |
| `' pe+pt'` | 0 | -0.2 | 0.0 |
| `' rel+ig'` | 0 | -0.2 | 0.0 |
| `'D+iam'` | 0 | -0.2 | 0.0 |
| `'G+G'` | 0 | -0.2 | 0.0 |
| `'I+B'` | 0 | -0.2 | 0.0 |
| `'eral+a'` | 0 | -0.2 | 0.0 |
| `'it+on'` | 0 | -0.2 | 0.0 |
| `'s+ers'` | 0 | -0.2 | 0.0 |
| `' At+l'` | 0 | -0.2 | 0.1 |
| `' O+B'` | 0 | -0.2 | 0.1 |
| `' S+B'` | 0 | -0.2 | 0.1 |
| `' fundra+ising'` | 0 | -0.2 | 0.1 |
| `' on+t'` | 0 | -0.2 | 0.1 |
| `' price+y'` | 0 | -0.2 | 0.1 |
| `' z+irc'` | 0 | -0.2 | 0.1 |
| `'"+?\n'` | 0 | -0.2 | 0.1 |
| `'-+angle'` | 0 | -0.2 | 0.1 |
| `'am+pton'` | 0 | -0.2 | 0.1 |
| `'erv+ices'` | 0 | -0.2 | 0.1 |
| `'istor+ic'` | 0 | -0.2 | 0.1 |
| `' Ch+rys'` | 0 | -0.2 | 0.2 |
| `' G+W'` | 0 | -0.2 | 0.2 |
| `' deg+rad'` | 0 | -0.2 | 0.2 |
| `' m+t'` | 0 | -0.2 | 0.2 |
| `' mouth+watering'` | 0 | -0.2 | 0.2 |
| `' new+com'` | 0 | -0.2 | 0.2 |
| `' over+ha'` | 0 | -0.2 | 0.2 |
| `' re+un'` | 0 | -0.2 | 0.2 |
| `' take+off'` | 0 | -0.2 | 0.2 |
| `' tool+kit'` | 0 | -0.2 | 0.2 |
| `' u+k'` | 0 | -0.2 | 0.2 |
| `'-+z'` | 0 | -0.2 | 0.2 |
| `'-e+lect'` | 0 | -0.2 | 0.2 |
| `'Al+cohol'` | 0 | -0.2 | 0.2 |
| `'Austral+ia'` | 0 | -0.2 | 0.2 |
| `'C+hem'` | 0 | -0.2 | 0.2 |
| `'C+ultural'` | 0 | -0.2 | 0.2 |
| `'F+ULL'` | 0 | -0.2 | 0.2 |
| `'FOR+M'` | 0 | -0.2 | 0.2 |
| `'M+er'` | 0 | -0.2 | 0.2 |
| `'Plan+ning'` | 0 | -0.2 | 0.2 |
| `'Rem+oving'` | 0 | -0.2 | 0.2 |
| `'S+eed'` | 0 | -0.2 | 0.2 |
| `'Supp+ose'` | 0 | -0.2 | 0.2 |
| `'a+verage'` | 0 | -0.2 | 0.2 |
| `'an+ne'` | 0 | -0.2 | 0.2 |
| `'cl+ose'` | 0 | -0.2 | 0.2 |
| `'em+ically'` | 0 | -0.2 | 0.2 |
| `'oc+rine'` | 0 | -0.2 | 0.2 |
| `'or+neys'` | 0 | -0.2 | 0.2 |
| `'our+ism'` | 0 | -0.2 | 0.2 |
| `'pl+oad'` | 0 | -0.2 | 0.2 |
| `'ron+ics'` | 0 | -0.2 | 0.2 |
| `'t+ag'` | 0 | -0.2 | 0.2 |
| `'te+c'` | 0 | -0.2 | 0.2 |
| `'ver+ted'` | 0 | -0.2 | 0.2 |
| `'/+v'` | 0 | -0.2 | 0.3 |
| `'g+ran'` | 0 | -0.2 | 0.3 |
| `'alle+led'` | 0 | -0.2 | 0.5 |
| `'k+ens'` | 0 | -0.2 | 0.5 |
| `' re+du'` | 0 | -0.2 | 0.6 |
| `'p+read'` | 0 | -0.2 | 1.0 |
| `'Pro+bably'` | 0 | -0.1 | -0.7 |
| `'est+hesia'` | 0 | -0.1 | -0.4 |
| `' merch+and'` | 0 | -0.1 | -0.2 |
| `' or+nith'` | 0 | -0.1 | -0.2 |
| `' Tasman+ia'` | 0 | -0.1 | -0.1 |
| `' cont+ag'` | 0 | -0.1 | -0.1 |
| `' un+avoid'` | 0 | -0.1 | -0.1 |
| `'ah+aran'` | 0 | -0.1 | -0.1 |
| `'aw+k'` | 0 | -0.1 | -0.1 |
| `'ena+issance'` | 0 | -0.1 | -0.1 |
| `'he+nd'` | 0 | -0.1 | -0.1 |
| `'in+ine'` | 0 | -0.1 | -0.1 |
| `'rim+ental'` | 0 | -0.1 | -0.1 |
| `' I+BS'` | 0 | -0.1 | 0.0 |
| `' Sch+w'` | 0 | -0.1 | 0.0 |
| `' electro+ph'` | 0 | -0.1 | 0.0 |
| `' reg+ist'` | 0 | -0.1 | 0.0 |
| `'Be+havior'` | 0 | -0.1 | 0.0 |
| `'J+ournal'` | 0 | -0.1 | 0.0 |
| `'exper+ienced'` | 0 | -0.1 | 0.0 |
| `'ol+ta'` | 0 | -0.1 | 0.0 |
| `'or+ning'` | 0 | -0.1 | 0.0 |
| `'st+uff'` | 0 | -0.1 | 0.0 |
| `'w+agen'` | 0 | -0.1 | 0.0 |
| `' O+y'` | 0 | -0.1 | 0.1 |
| `' Rep+ro'` | 0 | -0.1 | 0.1 |
| `' UC+LA'` | 0 | -0.1 | 0.1 |
| `' in+hal'` | 0 | -0.1 | 0.1 |
| `' l+ymp'` | 0 | -0.1 | 0.1 |
| `' m+alle'` | 0 | -0.1 | 0.1 |
| `' ma+gg'` | 0 | -0.1 | 0.1 |
| `'-+occ'` | 0 | -0.1 | 0.1 |
| `'.+There'` | 0 | -0.1 | 0.1 |
| `'IN+K'` | 0 | -0.1 | 0.1 |
| `'ac+i'` | 0 | -0.1 | 0.1 |
| `'id+ad'` | 0 | -0.1 | 0.1 |
| `'iss+ors'` | 0 | -0.1 | 0.1 |
| `'me+at'` | 0 | -0.1 | 0.1 |
| `'o+C'` | 0 | -0.1 | 0.1 |
| `'oc+in'` | 0 | -0.1 | 0.1 |
| `'od+ox'` | 0 | -0.1 | 0.1 |
| `'ol+ian'` | 0 | -0.1 | 0.1 |
| `'ond+o'` | 0 | -0.1 | 0.1 |
| `'y+ellow'` | 0 | -0.1 | 0.1 |
| `' Grand+e'` | 0 | -0.1 | 0.2 |
| `' H+C'` | 0 | -0.1 | 0.2 |
| `' L+l'` | 0 | -0.1 | 0.2 |
| `' Met+ro'` | 0 | -0.1 | 0.2 |
| `' dry+ers'` | 0 | -0.1 | 0.2 |
| `' em+an'` | 0 | -0.1 | 0.2 |
| `' fol+ate'` | 0 | -0.1 | 0.2 |
| `' g+asses'` | 0 | -0.1 | 0.2 |
| `' rare+st'` | 0 | -0.1 | 0.2 |
| `' th+ym'` | 0 | -0.1 | 0.2 |
| `'-p+itched'` | 0 | -0.1 | 0.2 |
| `'Egg+s'` | 0 | -0.1 | 0.2 |
| `'Elect+ronic'` | 0 | -0.1 | 0.2 |
| `'Er+ror'` | 0 | -0.1 | 0.2 |
| `'Gold+en'` | 0 | -0.1 | 0.2 |
| `'I+E'` | 0 | -0.1 | 0.2 |
| `'N+at'` | 0 | -0.1 | 0.2 |
| `'T+reat'` | 0 | -0.1 | 0.2 |
| `'W+alk'` | 0 | -0.1 | 0.2 |
| `'W+arm'` | 0 | -0.1 | 0.2 |
| `'W+ire'` | 0 | -0.1 | 0.2 |
| `'arm+ac'` | 0 | -0.1 | 0.2 |
| `'be+ck'` | 0 | -0.1 | 0.2 |
| `'ig+uity'` | 0 | -0.1 | 0.2 |
| `'il+ight'` | 0 | -0.1 | 0.2 |
| `'ond+itions'` | 0 | -0.1 | 0.2 |
| `'ong+ed'` | 0 | -0.1 | 0.2 |
| `'os+ites'` | 0 | -0.1 | 0.2 |
| `'psy+ch'` | 0 | -0.1 | 0.2 |
| `'rig+ation'` | 0 | -0.1 | 0.2 |
| `'—+are'` | 0 | -0.1 | 0.2 |
| `'id+uous'` | 0 | -0.1 | 0.5 |
| `'oc+ur'` | 0 | -0.1 | 0.8 |
| `' neurotransmit+ter'` | 0 | 0.0 | -0.8 |
| `'o+oth'` | 0 | 0.0 | -0.5 |
| `'ure+th'` | 0 | 0.0 | -0.5 |
| `'au+kee'` | 0 | 0.0 | -0.4 |
| `' Am+sterdam'` | 0 | 0.0 | -0.3 |
| `' h+ambur'` | 0 | 0.0 | -0.3 |
| `' t+onn'` | 0 | 0.0 | -0.3 |
| `' Dy+nam'` | 0 | 0.0 | -0.2 |
| `' r+if'` | 0 | 0.0 | -0.2 |
| `'af+ood'` | 0 | 0.0 | -0.2 |
| `'und+ay'` | 0 | 0.0 | -0.2 |
| `' Biom+edical'` | 0 | 0.0 | -0.1 |
| `' a+ure'` | 0 | 0.0 | -0.1 |
| `' ex+clus'` | 0 | 0.0 | -0.1 |
| `' f+ict'` | 0 | 0.0 | -0.1 |
| `' sub+mar'` | 0 | 0.0 | -0.1 |
| `'m+ega'` | 0 | 0.0 | -0.1 |
| `'math+rm'` | 0 | 0.0 | -0.1 |

## kept (126) — LIVE contribution (LOO vs the rest, recomputed every round; rounds table above is commit-time history and drifts). Promote these into BLOCKED_PAIRS.
| pair | Δcov | Δcomp | Δdead | earns now? |
|---|---|---|---|---|
| `' inv'+'ol'` | 3 | -9.4 | 0.2 | yes |
| `' inc'+'re'` | 1 | -23.6 | 0.8 | yes |
| `'f'+'ter'` | 2 | -8.7 | -0.1 | yes |
| `' Man'+'ufact'` | 2 | -0.4 | 0.2 | yes |
| `' pro'+'du'` | 2 | -27.4 | -0.5 | yes |
| `'ah'+'u'` | 1 | -5.2 | 0.0 | yes |
| `' contin'+'u'` | 1 | -5.7 | -0.1 | yes |
| `' imm'+'edi'` | 1 | -9.6 | 0.2 | yes |
| `'....'+'....'` | 1 | -2.9 | 0.1 | yes |
| `' diff'+'ere'` | 2 | -9.7 | 0.0 | yes |
| `' tra'+'um'` | 2 | -4.4 | 0.4 | yes |
| `'I'+'H'` | 1 | -7.2 | 0.0 | yes |
| `' ag'+'ricult'` | 1 | -4.8 | -0.2 | yes |
| `' antib'+'iot'` | 1 | -2.7 | -0.4 | yes |
| `' Ind'+'ivid'` | 1 | -2.7 | 0.1 | yes |
| `' Pro'+'ble'` | 1 | -2.6 | -0.3 | yes |
| `' Te'+'ac'` | 1 | -2.5 | -0.2 | yes |
| `' ass'+'um'` | 0 | -4.2 | -0.8 | yes |
| `' P'+'upp'` | 0 | -3.8 | -0.2 | yes |
| `' D'+'ise'` | 0 | -3.0 | -0.5 | yes |
| `' F'+'if'` | 0 | -2.7 | 0.0 | yes |
| `' V'+'ac'` | 0 | -2.3 | 0.1 | yes |
| `' acc'+'ur'` | 0 | -2.2 | -0.8 | yes |
| `' m'+'ov'` | 1 | -6.2 | 0.3 | yes |
| `'·'+'·'` | 1 | -5.9 | -0.7 | yes |
| `' inc'+'lud'` | 1 | -5.2 | 0.5 | yes |
| `' exper'+'im'` | 1 | -5.7 | -0.4 | yes |
| `'ot'+'ted'` | 1 | -7.0 | -0.2 | yes |
| `'ot'+'ropic'` | 0 | -14.6 | -0.1 | yes |
| `'av'+'ig'` | 1 | -11.0 | 0.7 | yes |
| `'u'+'ild'` | 1 | -10.4 | 0.5 | yes |
| `' w'+'r'` | 0 | -12.0 | 0.0 | yes |
| `' cit'+'iz'` | 0 | -8.1 | -0.9 | yes |
| `' est'+'ab'` | 0 | -7.9 | -0.5 | yes |
| `'th'+'rop'` | 0 | -7.3 | -0.2 | yes |
| `' inst'+'it'` | 0 | -7.4 | -0.4 | yes |
| `'S'+'F'` | 0 | -6.8 | -0.1 | yes |
| `'om'+'ile'` | 0 | -6.8 | -0.1 | yes |
| `' ex'+'ce'` | 0 | -6.5 | -0.7 | yes |
| `'op'+'art'` | 0 | -5.7 | 0.0 | yes |
| `' rep'+'res'` | 1 | -3.9 | -0.2 | yes |
| `' t'+'iss'` | 0 | -5.3 | -1.0 | yes |
| `' reg'+'ul'` | 1 | -3.7 | -0.7 | yes |
| `' vol'+'un'` | 1 | -3.7 | -0.4 | yes |
| `' encou'+'rag'` | 0 | -5.0 | -1.9 | yes |
| `' cont'+'roll'` | 0 | -5.0 | 0.0 | yes |
| `'ar'+'as'` | 0 | -6.8 | 0.0 | yes |
| `' prov'+'id'` | 1 | -2.9 | 0.3 | yes |
| `','+'00'` | 1 | -506.3 | -0.4 | yes |
| `'on'+'el'` | 1 | -1.4 | -0.3 | yes |
| `' b'+'reat'` | 1 | -0.8 | -0.4 | yes |
| `'el'+'ry'` | 1 | -0.6 | -0.1 | yes |
| `'ther'+'net'` | 1 | -1.8 | -0.3 | yes |
| `'ress'+'ions'` | 1 | -1.2 | -0.2 | yes |
| `'ign'+'ing'` | 1 | -0.4 | 0.0 | yes |
| `'n'+'ate'` | 1 | -0.8 | -0.1 | yes |
| `'ang'+'ar'` | 2 | -0.9 | -0.3 | yes |
| `' '+'\uf0b7'` | 1 | -0.8 | -0.2 | yes |
| `' Ad'+'vis'` | 0 | -1.2 | 0.0 | yes |
| `' An'+'alog'` | 1 | 0.4 | 0.4 | yes |
| `' An'+'im'` | 1 | 0.6 | 0.6 | yes |
| `' >'+'>'` | 1 | 0.6 | 0.5 | yes |
| `' A'+'AA'` | 1 | 0.7 | 0.4 | yes |
| `' Air'+'bus'` | 1 | 1.3 | 0.4 | yes |
| `'n'+'iv'` | 0 | -14.1 | -0.9 | yes |
| `'por'+'ary'` | 0 | -10.2 | -0.5 | yes |
| `'x'+'im'` | 0 | -7.6 | 0.9 | yes |
| `'ec'+'ause'` | 1 | -3.3 | 0.6 | yes |
| `' desc'+'rib'` | 0 | -7.8 | -0.6 | yes |
| `' ele'+'ph'` | 0 | -6.2 | -1.6 | yes |
| `'auc'+'oma'` | 0 | -6.0 | 0.1 | yes |
| `' colle'+'ag'` | 1 | -2.7 | -0.5 | yes |
| `' cl'+'us'` | 1 | -2.7 | -0.4 | yes |
| `' d'+'imens'` | 1 | -2.7 | -0.4 | yes |
| `' mat'+'hemat'` | 1 | -2.7 | -0.4 | yes |
| `' sh'+'ap'` | 0 | -5.1 | 0.4 | yes |
| `' c'+'ateg'` | 0 | -4.4 | -0.7 | yes |
| `' diss'+'oci'` | 0 | -4.2 | 0.0 | yes |
| `'du'+'ate'` | 0 | -4.2 | -0.4 | yes |
| `'nov'+'ation'` | 0 | -4.2 | 0.0 | yes |
| `' ch'+'ann'` | 0 | -4.1 | 0.1 | yes |
| `' influ'+'en'` | 0 | -4.0 | -0.7 | yes |
| `'y'+'pe'` | 0 | -4.3 | -0.2 | yes |
| `'ak'+'u'` | 0 | -4.1 | 0.2 | yes |
| `'e'+'ah'` | 1 | -0.9 | -0.3 | yes |
| `' es'+'oph'` | 0 | -3.9 | 0.5 | yes |
| `'N'+'OT'` | 0 | -3.9 | -0.1 | yes |
| `' se'+'arc'` | 0 | -3.7 | -0.6 | yes |
| `' dis'+'pl'` | 0 | -3.6 | 0.0 | yes |
| `' pl'+'aus'` | 0 | -3.6 | 0.1 | yes |
| `'ric'+'ulum'` | 0 | -3.5 | 0.0 | yes |
| `' out'+'bre'` | 0 | -3.4 | -0.9 | yes |
| `' port'+'ra'` | 0 | -4.0 | -0.1 | yes |
| `' sub'+'sequ'` | 0 | -3.4 | -0.8 | yes |
| `' Int'+'ellig'` | 0 | -3.2 | -0.5 | yes |
| `' bi'+'op'` | 0 | -3.2 | -0.2 | yes |
| `' c'+'igaret'` | 0 | -3.2 | 0.0 | yes |
| `' pers'+'pect'` | 0 | -3.2 | -0.8 | yes |
| `'an'+'ish'` | 1 | -9.1 | -0.2 | yes |
| `' const'+'ra'` | 0 | -3.2 | -0.6 | yes |
| `'ic'+'ose'` | 0 | -3.2 | -0.2 | yes |
| `'n'+'om'` | 0 | -3.2 | -0.4 | yes |
| `' rep'+'ut'` | 0 | -3.1 | -0.4 | yes |
| `'ns'+'ure'` | 1 | -3.1 | -1.1 | yes |
| `'iber'+'ty'` | 1 | -3.0 | -0.5 | yes |
| `'ol'+'ver'` | 0 | -3.0 | 0.2 | yes |
| `' exper'+'ien'` | 0 | -2.9 | -0.8 | yes |
| `' rein'+'for'` | 0 | -2.9 | -0.4 | yes |
| `'ab'+'it'` | 0 | -2.9 | -1.0 | yes |
| `'ab'+'ul'` | 0 | -2.9 | 0.1 | yes |
| `'cc'+'ess'` | 0 | -2.9 | -0.3 | yes |
| `'meric'+'an'` | 0 | -2.9 | -0.6 | yes |
| `' rest'+'ric'` | 0 | -2.8 | 0.0 | yes |
| `'-'+'hyd'` | 0 | -2.8 | -0.1 | yes |
| `'under'+'st'` | 0 | -2.8 | -0.3 | yes |
| `' p'+'estic'` | 0 | -2.7 | -0.8 | yes |
| `'.'+'in'` | 0 | -2.7 | 0.0 | yes |
| `'r'+'ateg'` | 0 | -2.7 | -0.8 | yes |
| `'ros'+'so'` | 1 | -2.7 | -0.2 | yes |
| `' micro'+'cont'` | 0 | -2.6 | 0.1 | yes |
| `'ay'+'enne'` | 1 | -2.6 | 0.1 | yes |
| `'yth'+'ons'` | 0 | -2.6 | 0.0 | yes |
| `' earth'+'qu'` | 0 | -2.5 | -0.8 | yes |
| `'-b'+'ut'` | 0 | -2.5 | -0.1 | yes |
| `'S'+'ar'` | 0 | -2.5 | -0.1 | yes |
| `'ur'+'is'` | 2 | -2.5 | -0.6 | yes |

## cooling (3) — benched by the eject-cooldown (exponential backoff; re-enters the pool when the bench expires, or earlier via a 0-keep-round release)
| pair | ejects | benched until round |
|---|---|---|
| `'f'+'ur'` | 1 | 256 |
| `'ste'+'ine'` | 1 | 257 |
| `' comp'+'ut'` | 1 | 259 |
