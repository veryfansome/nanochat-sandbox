# auto_tune held-out-Pareto FIXPOINT ledger (corpus-cached)

- held-out shards: 10 (10,000,000 chars each), avg-gated
- gate: held-out Pareto — cov≥0 AND comp≤0 AND dead≤0 AND ≥1 strict improvement (coverage-up OR coverage-flat comp/dead win) + dead-tolerance 1.0 (Δcov≥0,Δcomp<0,Δdead≤1.0, committed after free keeps) + comp-tolerance 6.0 (Δcov>0,Δcomp≤6.0,Δdead≤1.0, committed after free keeps); coverage prioritized; recheck ALL + recycle non-earners every round; eject-cooldown base 4 (k-th jail event benches a pair 4*2^(k-1) rounds; early release on a 0-keep round); transactional net-commit (trial → settle ejections → net gate; net-negative trials vetoed + jailed)
- candidates: adaptive mine (suffix, thr 0.15, sib 3, all-routes True, +dead-orphans)

## running tally
- status **running** | rounds 300 (commits 178, cleanups 18, pairs recycled 49, vetoes 79) | pool 1815 | kept **129**
- coverage **18249** (base 14039 / infl 4210)  ·  Δ vs baseline +111
- compression avg/shard **2140099**  ·  Δ vs baseline +38.0
- dead avg/shard **1209.1**  ·  Δ vs baseline +4.3

## rounds (one commit each; pool re-evaluated vs the new reference)
| round | pool | keeps | committed | Δcov | Δcomp | Δdead |
|---|---|---|---|---|---|---|
| 1 | 2110 | 34 | `'ot'+'ting'` | 2 | -3.0 | 0.0 |
| 2 | 2106 | 105 | `'op'+'ed'` | 2 | -1.3 | 0.0 |
| 3 | 2100 | 54 | `'ick'+'er'` | 2 | -0.2 | -0.2 |
| 4 | 2096 | 32 | `'ough'+'ly'` | 2 | -0.1 | 0.0 |
| 5 | 2095 | 103 | `'an'+'ish'` | 1 | -9.1 | -0.2 |
| 6 | 2094 | 102 | `'oad'+'ing'` | 1 | -6.3 | 0.0 |
| 7 | 2091 | 101 | `'ns'+'ure'` | 1 | -3.1 | -1.1 |
| 8 | 2090 | 100 | `'y'+'al'` | 1 | -1.7 | -0.9 |
| 9 | 2088 | 28 | `'erson'+'al'` | 1 | -3.7 | -0.1 |
| 10 | 2086 | 48 | `'os'+'ers'` | 1 | -1.0 | 0.0 |
| 11 | 2083 | 47 | `'an'+'ed'` | 1 | -0.9 | -0.3 |
| 12 | 2083 | 27 | `'ct'+'ure'` | 1 | -3.4 | -0.2 |
| 13 | 2081 | 46 | `'ot'+'hes'` | 1 | -0.5 | -0.2 |
| 14 | 2082 | 26 | `'yp'+'es'` | 1 | -2.5 | -0.1 |
| 15 | 2078 | 44 | `'ield'+'ing'` | 1 | -0.4 | 0.0 |
| 16 | 2076 | 43 | `'pect'+'ing'` | 1 | -0.3 | -0.1 |
| 17 | 2073 | 42 | `'ad'+'ians'` | 1 | -0.2 | -0.2 |
| 18 | 2070 | 41 | `'il'+'ant'` | 1 | -0.2 | -0.1 |
| 19 | 2065 | 40 | `'ress'+'or'` | 1 | -0.2 | 0.0 |
| 20 | 2063 | 39 | `'ist'+'ication'` | 1 | -0.1 | -0.1 |
| 21 | 2061 | 38 | `'ung'+'ent'` | 1 | -0.1 | 0.0 |
| 22 | 2059 | 37 | `'og'+'ging'` | 1 | 0.0 | 0.0 |
| 23 | 2056 | 36 | `'orn'+'ings'` | 1 | 0.0 | 0.0 |
| 24 | 2054 | 35 | `'ab'+'or'` | 0 | -12.3 | -0.1 |
| 25 | 2055 | 55 | `'o'+'es'` | 1 | -1.3 | -1.4 |
| 26 | 2055 | 14 | `'op'+'uses'` | 1 | -2.2 | -0.1 |
| 27 | 2053 | 31 | `'ite'+'ly'` | 0 | -10.5 | -0.1 |
| 28 | 2052 | 76 | `'en'+'ers'` | 2 | -0.9 | 0.0 |
| 29 | 2048 | 12 | `'ol'+'ition'` | 0 | -4.2 | -0.1 |
| 30 | 2045 | 80 | `'i'+'king'` | 0 | -3.0 | 0.0 |
| 31 | 2043 | 11 | `'el'+'es'` | 0 | -2.9 | -0.2 |
| 32 | 2041 | 10 | `'et'+'ed'` | 0 | -2.8 | 0.0 |
| 33 | 2038 | 9 | `'hys'+'ical'` | 0 | -2.5 | -0.3 |
| 34 | 2036 | 8 | `'ig'+'ure'` | 0 | -2.4 | 0.0 |
| 35 | 2035 | 7 | `'celer'+'ation'` | 0 | -1.8 | 0.0 |
| 36 | 2033 | 6 | `'i'+'ological'` | 0 | -1.8 | 0.0 |
| 37 | 2033 | 73 | `'eg'+'al'` | 0 | -2.9 | -0.1 |
| 38 | 2030 | 5 | `'orp'+'hism'` | 0 | -0.7 | -0.3 |
| 39 | 2028 | 4 | `'unction'+'al'` | 0 | -0.4 | 0.0 |
| 40 | 2026 | 3 | `'or'+'ning'` | 0 | -0.3 | 0.0 |
| 41 | 2024 | 104 | `'oci'+'ety'` | 0 | -4.0 | 0.0 |
| 42 | 2023 | 2 | `'ent'+'iful'` | 0 | -0.2 | -0.2 |
| 43 | 2021 | 1 | `'uc'+'ent'` | 0 | -0.2 | -0.1 |
| 44 | 2026 | 4 | `' streng'+'the'` | 1 | -2.8 | -0.9 |
| 45 | 2024 | 21 | `'s'+'els'` | 1 | -0.4 | -0.1 |
| 46 | 2021 | 20 | `'ut'+'her'` | 0 | -5.0 | -0.2 |
| 47 | 2018 | 2 | `' phot'+'ograp'` | 0 | -2.0 | -0.1 |
| 48 | 2019 | 1 | `' fram'+'ew'` | 0 | -1.5 | -0.7 |
| 49 | 2017 | 2 | `'lish'+'ing'` | 1 | -1.5 | 0.1 |
| 50 | 2014 | 1 | `'ort'+'ment'` | 1 | -0.2 | 0.1 |
| 51 | 2013 | 0 | — cleanup −4 recycled | - | - | - |
| 52 | 2013 | 0 | — cleanup −5 recycled | - | - | - |
| 53 | 2013 | 0 | — cleanup −11 recycled | - | - | - |
| 54 | 2013 | 0 | — cleanup −1 recycled | - | - | - |
| 55 | 2013 | 0 | — cleanup −1 recycled | - | - | - |
| 56 | 2057 | 22 | `'ough'+'ly'` | 2 | -0.1 | 0.0 |
| 57 | 2056 | 87 | `'op'+'ed'` | 2 | -1.3 | 0.0 |
| 58 | 2050 | 35 | `'ick'+'er'` | 2 | -0.2 | -0.2 |
| 59 | 2046 | 21 | `'erson'+'al'` | 1 | -3.7 | -0.1 |
| 60 | 2046 | 0 | — cleanup −1 recycled | - | - | - |
| 61 | 2046 | 0 | — cleanup −1 recycled | - | - | - |
| 62 | 2054 | 20 | `'ct'+'ure'` | 1 | -3.4 | -0.2 |
| 63 | 2052 | 34 | `'op'+'ed'` | 2 | -1.5 | -0.3 |
| 64 | 2046 | 9 | `'ite'+'ly'` | 0 | -9.5 | -0.1 |
| 65 | 2045 | 77 | `'o'+'es'` | 3 | -0.8 | -1.3 |
| 66 | 2045 | 52 | `'y'+'al'` | 1 | -0.9 | -0.8 |
| 67 | 2045 | 0 | — cleanup −1 recycled | - | - | - |
| 68 | 2044 | 87 | `'yp'+'es'` | 0 | -5.5 | -0.4 |
| 69 | 2040 | 17 | `'en'+'er'` | 0 | -7.1 | -0.7 |
| 70 | 2034 | 86 | `'op'+'uses'` | 0 | -5.2 | -0.4 |
| 71 | 2032 | 15 | `'ol'+'ition'` | 0 | -3.8 | -0.1 |
| 72 | 2029 | 84 | `'en'+'ers'` | 0 | -4.6 | 0.0 |
| 73 | 2029 | 0 | — cleanup −1 recycled | - | - | - |
| 74 | 2031 | 0 | — cleanup −1 recycled | - | - | - |
| 75 | 2034 | 144 | `'el'+'es'` | 0 | -4.1 | -0.2 |
| 76 | 2032 | 85 | `'ol'+'ition'` | 0 | -3.8 | -0.1 |
| 77 | 2029 | 140 | `'i'+'king'` | 0 | -3.4 | 0.0 |
| 78 | 2027 | 84 | `'i'+'ological'` | 0 | -1.4 | 0.0 |
| 79 | 2027 | 139 | `'eg'+'al'` | 0 | -3.3 | -0.1 |
| 80 | 2024 | 82 | `'or'+'ning'` | 0 | -1.0 | 0.0 |
| 81 | 2022 | 138 | `'oci'+'ety'` | 0 | -3.3 | 0.0 |
| 82 | 2021 | 81 | `'l'+'ished'` | 0 | -0.7 | -0.2 |
| 83 | 2022 | 80 | `'op'+'hor'` | 0 | -0.7 | 0.0 |
| 84 | 2020 | 0 | — cleanup −5 recycled | - | - | - |
| 85 | 2028 | 67 | `'ough'+'ly'` | 1 | -0.8 | 0.0 |
| 86 | 2027 | 251 | `'an'+'ed'` | 2 | -0.3 | -0.1 |
| 87 | 2025 | 79 | `'ol'+'ition'` | 0 | -3.8 | -0.1 |
| 88 | 2022 | 133 | `'i'+'king'` | 0 | -3.4 | 0.0 |
| 89 | 2020 | 78 | `'i'+'ological'` | 0 | -1.4 | 0.0 |
| 90 | 2020 | 132 | `'oci'+'ety'` | 0 | -3.3 | 0.0 |
| 91 | 2019 | 77 | `'or'+'ning'` | 0 | -1.0 | 0.0 |
| 92 | 2017 | 131 | `'ut'+'ory'` | 0 | -3.1 | 0.0 |
| 93 | 2014 | 76 | `'id'+'ation'` | 0 | -0.4 | 0.0 |
| 94 | 2012 | 0 | — cleanup −6 recycled | - | - | - |
| 95 | 2023 | 249 | `'ut'+'her'` | 1 | -4.4 | 0.0 |
| 96 | 2020 | 77 | `'ol'+'ition'` | 0 | -3.8 | -0.1 |
| 97 | 2017 | 131 | `'i'+'king'` | 0 | -3.4 | 0.0 |
| 98 | 2015 | 76 | `'i'+'ological'` | 0 | -1.4 | 0.0 |
| 99 | 2015 | 130 | `'oci'+'ety'` | 0 | -3.3 | 0.0 |
| 100 | 2014 | 75 | `'or'+'ning'` | 0 | -1.0 | 0.0 |
| 101 | 2012 | 129 | `'ut'+'ory'` | 0 | -3.1 | 0.0 |
| 102 | 2009 | 74 | `'k'+'al'` | 0 | -0.1 | -0.3 |
| 103 | 2007 | 61 | `'gg'+'ings'` | 0 | -0.1 | 0.0 |
| 104 | 2005 | 0 | — cleanup −3 recycled | - | - | - |
| 105 | 2010 | 0 | — cleanup −4 recycled | - | - | - |
| 106 | 2022 | 0 | — cleanup −1 recycled | - | - | - |
| 107 | 2024 | 145 | `'op'+'ed'` | 1 | -2.3 | -0.2 |
| 108 | 2018 | 260 | `'i'+'king'` | 1 | -2.3 | 0.0 |
| 109 | 2016 | 153 | `'roubleshoot'+'ing'` | 0 | -1.5 | -0.4 |
| 110 | 2009 | 92 | `'ab'+'or'` | 0 | -8.4 | -0.1 |
| 111 | 2009 | 0 | — cleanup −1 recycled | - | - | - |
| 112 | 2009 | 87 | `'k'+'al'` | 0 | -0.1 | -0.3 |
| 113 | 2007 | 73 | `'ure'+'rs'` — 2 vetoed | 2 | 2.2 | 0.3 |
| 114 | 2001 | 69 | `'ogene'+'ous'` | 2 | 2.3 | -0.2 |
| 115 | 1998 | 68 | `'ig'+'er'` — 1 vetoed | 2 | 4.8 | 0.3 |
| 116 | 1992 | 65 | `'on'+'ies'` | 2 | 5.1 | -0.1 |
| 117 | 1987 | 63 | `'pe'+'ed'` | 1 | -4.2 | 0.4 |
| 118 | 1991 | 84 | `'u'+'ces'` | 0 | -1.3 | -0.2 |
| 119 | 1988 | 142 | `'emet'+'ery'` | 0 | -1.2 | -0.4 |
| 120 | 1987 | 82 | `'ot'+'hes'` | 2 | 2.7 | -0.2 |
| 121 | 1988 | 141 | `'o'+'ices'` | 0 | -0.5 | 0.0 |
| 122 | 1988 | 81 | `'ick'+'ness'` | 2 | 3.9 | -0.3 |
| 123 | 1987 | 139 | `'omencl'+'ature'` | 0 | 0.0 | -0.4 |
| 124 | 1986 | 79 | `'ct'+'uary'` | 2 | 5.4 | -0.1 |
| 125 | 1984 | 137 | `'i'+'ants'` | 1 | -1.4 | 0.3 |
| 126 | 1981 | 136 | `'aps'+'es'` | 1 | -0.9 | 0.4 |
| 127 | 1975 | 77 | `'cher'+'y'` | 2 | 5.5 | 0.0 |
| 128 | 1972 | 133 | `'ceed'+'ings'` | 1 | 0.2 | 0.0 |
| 129 | 1971 | 132 | `'it'+'ness'` | 1 | 0.2 | 0.0 |
| 130 | 1970 | 130 | `'agnet'+'ism'` | 1 | 0.2 | 0.1 |
| 131 | 1969 | 129 | `'oot'+'ing'` | 1 | 0.5 | 0.0 |
| 132 | 1967 | 129 | `'ick'+'er'` — −1 ejected, net 1/-0.9/0.1 | 1 | -0.9 | 0.1 |
| 133 | 1964 | 128 | `'ruct'+'or'` | 1 | 0.5 | 0.1 |
| 134 | 1962 | 127 | `'en'+'ment'` | 1 | 0.5 | 0.2 |
| 135 | 1961 | 126 | `'ert'+'ation'` | 1 | 0.5 | 0.4 |
| 136 | 1960 | 125 | `'ull'+'ing'` | 1 | 0.6 | 0.0 |
| 137 | 1960 | 124 | `'or'+'ious'` | 1 | 0.6 | 0.4 |
| 138 | 1956 | 124 | `'an'+'cers'` | 1 | 0.7 | 0.0 |
| 139 | 1952 | 123 | `'mit'+'ted'` | 1 | 0.8 | -0.9 |
| 140 | 1947 | 65 | `'i'+'ological'` | 0 | -1.2 | -0.3 |
| 141 | 1947 | 120 | `'ab'+'ies'` | 1 | 0.8 | 0.0 |
| 142 | 1946 | 119 | `'ra'+'ble'` | 1 | 0.9 | -0.5 |
| 143 | 1943 | 118 | `'os'+'her'` | 1 | 0.9 | 0.0 |
| 144 | 1941 | 117 | `'u'+'ber'` | 1 | 1.1 | -0.1 |
| 145 | 1938 | 116 | `'ol'+'ing'` | 1 | 1.2 | 0.0 |
| 146 | 1934 | 115 | `'ud'+'ding'` | 1 | 1.2 | 0.2 |
| 147 | 1932 | 114 | `'uss'+'ion'` | 1 | 1.3 | 0.3 |
| 148 | 1930 | 113 | `'row'+'ning'` | 1 | 1.4 | 0.2 |
| 149 | 1928 | 112 | `'ire'+'ments'` | 1 | 1.7 | 0.4 |
| 150 | 1926 | 56 | `'i'+'kes'` | 0 | -0.9 | 0.0 |
| 151 | 1924 | 110 | `'zym'+'es'` | 1 | 1.8 | 0.1 |
| 152 | 1923 | 54 | `'or'+'ning'` | 0 | -0.8 | -0.3 |
| 153 | 1921 | 109 | `'ew'+'able'` | 1 | 1.9 | 0.2 |
| 154 | 1920 | 108 | `'tain'+'ment'` | 1 | 2.1 | 0.0 |
| 155 | 1919 | 107 | `'on'+'ian'` | 1 | 2.3 | -0.1 |
| 156 | 1916 | 106 | `'ruct'+'ed'` | 1 | 2.3 | 0.0 |
| 157 | 1915 | 105 | `'uff'+'er'` | 1 | 2.3 | 0.0 |
| 158 | 1913 | 104 | `'prising'+'ly'` | 1 | 2.4 | 0.0 |
| 159 | 1912 | 103 | `'aw'+'ning'` | 1 | 2.6 | 0.3 |
| 160 | 1908 | 101 | `'ar'+'king'` | 1 | 2.9 | -0.2 |
| 161 | 1906 | 100 | `'idd'+'ing'` | 1 | 3.0 | 0.0 |
| 162 | 1904 | 99 | `'as'+'ures'` | 1 | 3.1 | -0.2 |
| 163 | 1899 | 27 | `'ip'+'ed'` — 2 vetoed | 1 | 1.2 | 0.3 |
| 164 | 1890 | 95 | `'ort'+'ing'` | 1 | 3.2 | 0.0 |
| 165 | 1888 | 95 | `'ond'+'er'` | 1 | 3.3 | 0.0 |
| 166 | 1885 | 94 | `'os'+'ion'` | 1 | 3.3 | 0.4 |
| 167 | 1883 | 94 | `'ider'+'y'` | 1 | 3.5 | 0.1 |
| 168 | 1880 | 93 | `'viron'+'ments'` | 1 | 3.6 | 0.0 |
| 169 | 1879 | 92 | `'at'+'hered'` | 1 | 3.7 | 0.3 |
| 170 | 1876 | 108 | `'ard'+'ed'` — 1 vetoed | 2 | 4.8 | 0.8 |
| 171 | 1873 | 34 | `'ight'+'ly'` — 1 vetoed | 1 | 0.1 | 0.9 |
| 172 | 1873 | 34 | `'ient'+'ial'` — 2 vetoed | 1 | 0.6 | 0.6 |
| 173 | 1869 | 31 | `'a'+'very'` | 1 | 1.3 | 0.6 |
| 174 | 1865 | 30 | `'ri'+'ors'` | 1 | 3.0 | 1.0 |
| 175 | 1863 | 30 | `'arent'+'ly'` — 1 vetoed | 1 | 3.5 | 0.6 |
| 176 | 1861 | 29 | `'is'+'ure'` — 2 vetoed | 1 | 3.8 | 0.9 |
| 177 | 1858 | 26 | `'ot'+'ation'` | 1 | 4.1 | 0.6 |
| 178 | 1856 | 25 | `'inite'+'ly'` | 1 | 4.2 | 0.8 |
| 179 | 1855 | 24 | `'ura'+'bility'` | 1 | 4.3 | 0.0 |
| 180 | 1853 | 23 | `'ip'+'ment'` | 1 | 4.4 | 0.6 |
| 181 | 1853 | 23 | `'reat'+'ing'` — 1 vetoed | 1 | 4.6 | 0.3 |
| 182 | 1850 | 21 | `'ad'+'ian'` — −1 ejected, net 1/5.3/1.0 | 1 | 5.3 | 1.0 |
| 183 | 1849 | 20 | `'ress'+'es'` | 1 | 5.4 | 0.0 |
| 184 | 1848 | 20 | `'e'+'per'` — 1 vetoed | 1 | 5.4 | 0.6 |
| 185 | 1846 | 19 | `'ol'+'ly'` — 1 vetoed | 1 | 5.7 | 0.3 |
| 186 | 1843 | 17 | `'at'+'tered'` | 1 | 5.8 | 0.4 |
| 187 | 1840 | 16 | `'ool'+'ing'` | 0 | -2.9 | 0.1 |
| 188 | 1838 | 15 | `'orm'+'ally'` | 0 | -1.5 | 0.2 |
| 189 | 1836 | 16 | `'av'+'al'` — 2 vetoed | 0 | -1.2 | 0.2 |
| 190 | 1832 | 14 | `'ug'+'al'` — 3 vetoed | 0 | -0.9 | 0.1 |
| 191 | 1827 | 10 | `'ne'+'al'` — 3 vetoed | 0 | -0.6 | 0.1 |
| 192 | 1822 | 6 | `'ar'+'ning'` — 1 vetoed | 0 | -0.4 | 0.4 |
| 193 | 1819 | 4 | `'alle'+'led'` — 1 vetoed | 0 | -0.2 | 0.5 |
| 194 | 1816 | 2 | `'n'+'ters'` — 1 vetoed | 0 | -0.1 | 0.9 |
| 195 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 196 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 197 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 198 | 1814 | 0 | — 0 keeps → released `'ect'+'ion'` from cooldown | - | - | - |
| 199 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 200 | 1814 | 0 | — 0 keeps → released `'n'+'ter'` from cooldown | - | - | - |
| 201 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 202 | 1814 | 0 | — 0 keeps → released `'ic'+'ity'` from cooldown | - | - | - |
| 203 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 204 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 205 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 206 | 1814 | 0 | — 0 keeps → released `'ut'+'ory'` from cooldown | - | - | - |
| 207 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 208 | 1814 | 0 | — 0 keeps → released `'ic'+'ity'` from cooldown | - | - | - |
| 209 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 210 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 211 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 212 | 1814 | 0 | — 0 keeps → released `'oss'+'ing'` from cooldown | - | - | - |
| 213 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 214 | 1814 | 0 | — 0 keeps → released `'ou'+'les'` from cooldown | - | - | - |
| 215 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 216 | 1814 | 0 | — 0 keeps → released `'ect'+'ion'` from cooldown | - | - | - |
| 217 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 218 | 1814 | 0 | — 0 keeps → released `'n'+'ter'` from cooldown | - | - | - |
| 219 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 220 | 1814 | 0 | — 0 keeps → released `'ut'+'ory'` from cooldown | - | - | - |
| 221 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 222 | 1814 | 0 | — 0 keeps → released `'ic'+'ity'` from cooldown | - | - | - |
| 223 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 224 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 225 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 226 | 1814 | 0 | — 0 keeps → released `'oci'+'ety'` from cooldown | - | - | - |
| 227 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 228 | 1814 | 0 | — 0 keeps → released `'oss'+'ing'` from cooldown | - | - | - |
| 229 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 230 | 1814 | 0 | — 0 keeps → released `'ou'+'les'` from cooldown | - | - | - |
| 231 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 232 | 1814 | 0 | — 0 keeps → released `'ect'+'ion'` from cooldown | - | - | - |
| 233 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 234 | 1814 | 0 | — 0 keeps → released `'n'+'ter'` from cooldown | - | - | - |
| 235 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 236 | 1814 | 0 | — 0 keeps → released `'ut'+'ory'` from cooldown | - | - | - |
| 237 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 238 | 1814 | 0 | — 0 keeps → released `'ic'+'ity'` from cooldown | - | - | - |
| 239 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 240 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 241 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 242 | 1814 | 0 | — 0 keeps → released `'oci'+'ety'` from cooldown | - | - | - |
| 243 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 244 | 1814 | 0 | — 0 keeps → released `'oss'+'ing'` from cooldown | - | - | - |
| 245 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 246 | 1814 | 0 | — 0 keeps → released `'ou'+'les'` from cooldown | - | - | - |
| 247 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 248 | 1814 | 0 | — 0 keeps → released `'ect'+'ion'` from cooldown | - | - | - |
| 249 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 250 | 1814 | 0 | — 0 keeps → released `'n'+'ter'` from cooldown | - | - | - |
| 251 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 252 | 1814 | 0 | — 0 keeps → released `'ut'+'ory'` from cooldown | - | - | - |
| 253 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 254 | 1814 | 0 | — 0 keeps → released `'ic'+'ity'` from cooldown | - | - | - |
| 255 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 256 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 257 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 258 | 1814 | 0 | — 0 keeps → released `'oci'+'ety'` from cooldown | - | - | - |
| 259 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 260 | 1814 | 0 | — 0 keeps → released `'oss'+'ing'` from cooldown | - | - | - |
| 261 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 262 | 1814 | 0 | — 0 keeps → released `'ou'+'les'` from cooldown | - | - | - |
| 263 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 264 | 1814 | 0 | — 0 keeps → released `'ect'+'ion'` from cooldown | - | - | - |
| 265 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 266 | 1814 | 0 | — 0 keeps → released `'n'+'ter'` from cooldown | - | - | - |
| 267 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 268 | 1814 | 0 | — 0 keeps → released `'ut'+'ory'` from cooldown | - | - | - |
| 269 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 270 | 1814 | 0 | — 0 keeps → released `'ic'+'ity'` from cooldown | - | - | - |
| 271 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 272 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 273 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 274 | 1814 | 0 | — 0 keeps → released `'oci'+'ety'` from cooldown | - | - | - |
| 275 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 276 | 1814 | 0 | — 0 keeps → released `'oss'+'ing'` from cooldown | - | - | - |
| 277 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 278 | 1814 | 0 | — 0 keeps → released `'ou'+'les'` from cooldown | - | - | - |
| 279 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 280 | 1814 | 0 | — 0 keeps → released `'ect'+'ion'` from cooldown | - | - | - |
| 281 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 282 | 1814 | 0 | — 0 keeps → released `'n'+'ter'` from cooldown | - | - | - |
| 283 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 284 | 1814 | 0 | — 0 keeps → released `'ut'+'ory'` from cooldown | - | - | - |
| 285 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 286 | 1814 | 0 | — 0 keeps → released `'ic'+'ity'` from cooldown | - | - | - |
| 287 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 288 | 1814 | 0 | — 0 keeps → released `'as'+'ive'` from cooldown | - | - | - |
| 289 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 290 | 1814 | 0 | — 0 keeps → released `'oci'+'ety'` from cooldown | - | - | - |
| 291 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 292 | 1814 | 0 | — 0 keeps → released `'oss'+'ing'` from cooldown | - | - | - |
| 293 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 294 | 1814 | 0 | — 0 keeps → released `'ou'+'les'` from cooldown | - | - | - |
| 295 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 296 | 1814 | 0 | — 0 keeps → released `'ect'+'ion'` from cooldown | - | - | - |
| 297 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 298 | 1814 | 0 | — 0 keeps → released `'n'+'ter'` from cooldown | - | - | - |
| 299 | 1815 | 1 | — no commit: 1 trialed keep(s) vetoed (net-negative) | - | - | - |
| 300 | 1814 | 0 | — 0 keeps → released `'ut'+'ory'` from cooldown | - | - | - |

## round 301 IN PROGRESS — 1822/1815 evaluated, 8 keep(s) so far (best committed at round end)
| pair | Δcov | Δcomp | Δdead |
|---|---|---|---|
| `ect+ion` | 2 | -11.1 | 0.8 |
| `n+ter` | 1 | -2.1 | 0.0 |
| `ic+ity` | 0 | -1.0 | 0.3 |
| `oci+ety` | 0 | -0.9 | 0.1 |
| `ut+ory` | 0 | -0.7 | 0.1 |
| `as+ive` | 0 | -0.2 | 0.3 |
| `oss+ing` | 0 | -0.1 | 0.0 |
| `ou+les` | 0 | -0.1 | 0.0 |

## kept (129) — LIVE contribution (LOO vs the rest, recomputed every round; rounds table above is commit-time history and drifts). Promote these into BLOCKED_PAIRS.
| pair | Δcov | Δcomp | Δdead | earns now? |
|---|---|---|---|---|
| `'ot'+'ting'` | 3 | -2.4 | -0.2 | yes |
| `'an'+'ish'` | 1 | -11.2 | -0.2 | yes |
| `'oad'+'ing'` | 1 | -6.3 | 0.0 | yes |
| `'ns'+'ure'` | 1 | -3.1 | -1.1 | yes |
| `'os'+'ers'` | 1 | -1.0 | 0.0 | yes |
| `'ield'+'ing'` | 1 | -0.4 | 0.0 | yes |
| `'pect'+'ing'` | 1 | -0.3 | -0.1 | yes |
| `'il'+'ant'` | 1 | -0.2 | -0.1 | yes |
| `'ress'+'or'` | 1 | -0.8 | 0.0 | yes |
| `'ist'+'ication'` | 1 | -0.1 | -0.1 | yes |
| `'ung'+'ent'` | 1 | -0.1 | 0.0 | yes |
| `'og'+'ging'` | 1 | 0.0 | 0.0 | yes |
| `'orn'+'ings'` | 1 | 0.0 | 0.0 | yes |
| `'et'+'ed'` | 0 | -2.8 | 0.0 | yes |
| `'hys'+'ical'` | 0 | -2.9 | -0.3 | yes |
| `'ig'+'ure'` | 0 | -2.2 | 0.0 | yes |
| `'celer'+'ation'` | 0 | -1.8 | 0.0 | yes |
| `'orp'+'hism'` | 0 | -0.7 | -0.3 | yes |
| `'unction'+'al'` | 0 | -0.4 | 0.0 | yes |
| `'ent'+'iful'` | 0 | -0.2 | -0.2 | yes |
| `'uc'+'ent'` | 0 | -0.2 | -0.1 | yes |
| `' streng'+'the'` | 1 | -6.0 | -1.2 | yes |
| `'s'+'els'` | 1 | -0.4 | -0.1 | yes |
| `' phot'+'ograp'` | 0 | -2.0 | -0.1 | yes |
| `' fram'+'ew'` | 0 | -1.5 | -0.7 | yes |
| `'lish'+'ing'` | 1 | -1.5 | 0.1 | yes |
| `'ort'+'ment'` | 1 | -0.2 | 0.1 | yes |
| `'erson'+'al'` | 1 | -6.9 | -0.4 | yes |
| `'ct'+'ure'` | 1 | -6.6 | -0.5 | yes |
| `'ite'+'ly'` | 2 | -8.8 | 0.7 | yes |
| `'o'+'es'` | 3 | -3.4 | -1.2 | yes |
| `'y'+'al'` | 2 | -2.3 | -0.8 | yes |
| `'yp'+'es'` | 1 | -5.7 | -0.4 | yes |
| `'op'+'uses'` | 1 | -5.4 | -0.4 | yes |
| `'en'+'ers'` | 2 | 0.9 | 0.2 | yes |
| `'el'+'es'` | 1 | -4.3 | -0.2 | yes |
| `'eg'+'al'` | 1 | -3.5 | -0.1 | yes |
| `'op'+'hor'` | 1 | -3.3 | -0.1 | yes |
| `'ough'+'ly'` | 2 | -2.1 | -0.1 | yes |
| `'an'+'ed'` | 2 | -0.3 | -0.1 | yes |
| `'id'+'ation'` | 1 | -3.0 | -0.1 | yes |
| `'ut'+'her'` | 1 | -4.4 | 0.0 | yes |
| `'gg'+'ings'` | 1 | -2.7 | -0.1 | yes |
| `'op'+'ed'` | 2 | -4.9 | -0.2 | yes |
| `'i'+'king'` | 1 | -8.0 | -0.6 | yes |
| `'roubleshoot'+'ing'` | 1 | -1.5 | -0.7 | yes |
| `'ab'+'or'` | 0 | -14.2 | 0.0 | yes |
| `'k'+'al'` | 0 | -0.1 | -0.3 | yes |
| `'ure'+'rs'` | 2 | 2.2 | 0.3 | yes |
| `'ogene'+'ous'` | 2 | 2.3 | -0.2 | yes |
| `'ig'+'er'` | 2 | 4.8 | 0.3 | yes |
| `'on'+'ies'` | 2 | 5.1 | -0.1 | yes |
| `'pe'+'ed'` | 2 | -8.3 | 0.1 | yes |
| `'u'+'ces'` | 0 | -3.9 | 0.0 | yes |
| `'emet'+'ery'` | 1 | -1.2 | -0.7 | yes |
| `'ot'+'hes'` | 2 | 0.1 | 0.0 | yes |
| `'o'+'ices'` | 1 | -0.5 | -0.3 | yes |
| `'omencl'+'ature'` | 1 | 0.0 | -0.7 | yes |
| `'ct'+'uary'` | 2 | 2.8 | 0.1 | yes |
| `'i'+'ants'` | 1 | -1.4 | 0.3 | yes |
| `'aps'+'es'` | 2 | -0.9 | 0.1 | yes |
| `'cher'+'y'` | 2 | 2.9 | 0.2 | yes |
| `'ceed'+'ings'` | 1 | 0.2 | 0.0 | yes |
| `'it'+'ness'` | 1 | 0.2 | 0.0 | yes |
| `'agnet'+'ism'` | 1 | 0.2 | 0.1 | yes |
| `'oot'+'ing'` | 1 | 0.5 | 0.0 | yes |
| `'ick'+'er'` | 3 | 0.4 | 0.0 | yes |
| `'ruct'+'or'` | 1 | 0.2 | 0.0 | yes |
| `'en'+'ment'` | 1 | 0.5 | 0.2 | yes |
| `'ert'+'ation'` | 1 | 0.5 | 0.4 | yes |
| `'ull'+'ing'` | 1 | 0.6 | 0.0 | yes |
| `'or'+'ious'` | 1 | 0.6 | 0.4 | yes |
| `'an'+'cers'` | 1 | 0.7 | 0.0 | yes |
| `'mit'+'ted'` | 2 | 0.8 | -1.2 | yes |
| `'i'+'ological'` | 0 | -5.5 | 0.4 | yes |
| `'ab'+'ies'` | 1 | 0.8 | 0.0 | yes |
| `'ra'+'ble'` | 1 | 0.9 | -0.5 | yes |
| `'os'+'her'` | 1 | 0.9 | 0.0 | yes |
| `'u'+'ber'` | 1 | 1.1 | -0.1 | yes |
| `'ol'+'ing'` | 1 | 1.1 | 0.0 | yes |
| `'ud'+'ding'` | 1 | 1.2 | 0.2 | yes |
| `'uss'+'ion'` | 1 | 1.3 | 0.3 | yes |
| `'row'+'ning'` | 1 | 1.4 | 0.2 | yes |
| `'ire'+'ments'` | 2 | 1.7 | 0.1 | yes |
| `'i'+'kes'` | 0 | -3.5 | 0.2 | yes |
| `'zym'+'es'` | 2 | 1.8 | -0.2 | yes |
| `'or'+'ning'` | 0 | -3.4 | -0.1 | yes |
| `'ew'+'able'` | 1 | 1.9 | 0.2 | yes |
| `'tain'+'ment'` | 1 | 2.1 | 0.0 | yes |
| `'on'+'ian'` | 1 | 2.3 | -0.1 | yes |
| `'ruct'+'ed'` | 1 | 2.3 | 0.0 | yes |
| `'uff'+'er'` | 1 | 2.3 | 0.0 | yes |
| `'prising'+'ly'` | 1 | 2.4 | 0.0 | yes |
| `'aw'+'ning'` | 1 | 2.6 | 0.3 | yes |
| `'ar'+'king'` | 1 | 2.9 | -0.2 | yes |
| `'idd'+'ing'` | 1 | 3.0 | 0.0 | yes |
| `'as'+'ures'` | 1 | 3.1 | -0.2 | yes |
| `'ip'+'ed'` | 1 | -2.5 | 0.2 | yes |
| `'ort'+'ing'` | 1 | 3.2 | 0.0 | yes |
| `'ond'+'er'` | 1 | 3.3 | 0.0 | yes |
| `'os'+'ion'` | 1 | 3.3 | 0.4 | yes |
| `'ider'+'y'` | 1 | 3.5 | 0.1 | yes |
| `'viron'+'ments'` | 1 | 3.6 | 0.0 | yes |
| `'at'+'hered'` | 1 | 3.7 | 0.3 | yes |
| `'ard'+'ed'` | 2 | 4.8 | 0.8 | yes |
| `'ight'+'ly'` | 1 | 0.1 | 0.9 | yes |
| `'ient'+'ial'` | 1 | 0.6 | 0.6 | yes |
| `'a'+'very'` | 1 | 1.3 | 0.6 | yes |
| `'ri'+'ors'` | 1 | 3.0 | 1.0 | yes |
| `'arent'+'ly'` | 1 | 3.5 | 0.6 | yes |
| `'is'+'ure'` | 1 | 3.8 | 0.9 | yes |
| `'ot'+'ation'` | 1 | 4.1 | 0.6 | yes |
| `'inite'+'ly'` | 1 | 4.2 | 0.8 | yes |
| `'ura'+'bility'` | 1 | 4.3 | 0.0 | yes |
| `'ip'+'ment'` | 1 | 4.4 | 0.6 | yes |
| `'reat'+'ing'` | 1 | 4.6 | 0.3 | yes |
| `'ad'+'ian'` | 2 | 5.1 | 0.8 | yes |
| `'ress'+'es'` | 1 | 5.4 | 0.0 | yes |
| `'e'+'per'` | 1 | 5.4 | 0.6 | yes |
| `'ol'+'ly'` | 1 | 5.7 | 0.3 | yes |
| `'at'+'tered'` | 1 | 5.8 | 0.4 | yes |
| `'ool'+'ing'` | 0 | -2.9 | 0.1 | yes |
| `'orm'+'ally'` | 0 | -1.5 | 0.2 | yes |
| `'av'+'al'` | 0 | -1.2 | 0.2 | yes |
| `'ug'+'al'` | 0 | -0.9 | 0.1 | yes |
| `'ne'+'al'` | 0 | -0.6 | 0.1 | yes |
| `'ar'+'ning'` | 0 | -0.4 | 0.4 | yes |
| `'alle'+'led'` | 0 | -0.2 | 0.5 | yes |
| `'n'+'ters'` | 0 | -0.1 | 0.9 | yes |

## cooling (7) — benched by the eject-cooldown (exponential backoff; re-enters the pool when the bench expires, or earlier via a 0-keep-round release)
| pair | ejects | benched until round |
|---|---|---|
| `'ic'+'ity'` | 9 | 1311 |
| `'as'+'ive'` | 9 | 1313 |
| `'oci'+'ety'` | 10 | 2339 |
| `'oss'+'ing'` | 10 | 2341 |
| `'ou'+'les'` | 10 | 2343 |
| `'ect'+'ion'` | 10 | 2345 |
| `'n'+'ter'` | 10 | 2347 |
