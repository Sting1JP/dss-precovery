# Known-answer tests

Each test runs the tool on an object whose old positions are already well known from its JPL orbit. `tools/eval_known.py` then scores the run against these criteria:

1. At least one plate with the object on it is found.
2. Where the object is in clear sky (no Gaia star within 5"), it is the only candidate within 3 sigma, or else the top candidate.
3. Measured minus predicted is within 3 sigma in each coordinate (the tool's and JPL's errors combined), and within 2" in total.
4. Timing: on plates where half the exposure moves the object by more than twice the error, the tool's reading of the header time (start or middle of the exposure) fits better than the other reading.
5. Plates taken the same night agree with the predicted motion within 2".

The scorer also prints how many other candidates are in the search box (C6), and the rms and largest value of the residuals divided by their errors (C7).

## Results with version 1.0.0

| Test | Plates with the object | Result |
|---|---|---|
| Quaoar on POSS-I plates, 1950-1958 (`quaoar_1954`) | 4: red and blue, 1954 May 25 and 1956 May 8 | PASS; 0.24-0.87" from JPL's position |
| Quaoar on UK Schmidt plates, 1975-2000 (`quaoar_ukst`) | 2: 1982 Jul 17, 1994 Apr 20 | PASS; 0.05-0.55"; mid-exposure time confirmed on 1 plate |
| Eris on UK Schmidt plates, 1976-2000 (`eris_ukst`) | 2: 1982 Sep 19, 1989 Sep 29 | PASS; 0.20-0.25"; mid-exposure time confirmed on both |

Run on 30 Oct 2026 on macOS with Python 3.9.6, numpy 1.26.4, scipy 1.13.1, astropy 6.0.1 and Pillow 11.3.0.

Each folder here holds one run: `plan.json`, `results.json`, `results.csv` and the scorer's output, `score.txt`. The DSS images are not included. No position in these runs is marked reportable: the known-object check failed, as it does on all DSS plates (see the main README), and UK Schmidt positions are never marked reportable.

JPL's orbits for Quaoar and Eris include old measurements that may come from the same scans, so an error shared with those measurements would not show in these tests.

## Rerun

```
python3 tools/precovery.py quaoar_1954
python3 tools/eval_known.py output/precovery/quaoar_1954

python3 tools/precovery.py quaoar_ukst
python3 tools/eval_known.py output/precovery/quaoar_ukst --ukst

python3 tools/precovery.py eris_ukst
python3 tools/eval_known.py output/precovery/eris_ukst --ukst
```

To score the runs in this folder again without downloading anything, point the scorer at them, for example `python3 tools/eval_known.py known_answers/eris_ukst --ukst`.
