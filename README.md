# dss-precovery

This tool looks for a known, slow-moving Solar System object, such as a trans-Neptunian object, on old photographic plates and measures where it was. It works on the Digitized Sky Survey (DSS) scans of the 1950s Palomar plates (POSS-I) and of the later UK Schmidt plates.

For each object it:

- finds the plates that were taken while the object was on them (the DSS serves one plate per survey at each position, so where plates overlap, only that one is searched);
- refits each plate's astrometry to Gaia DR3 stars, moved to the date of the plate;
- searches the area where JPL Horizons predicts the object, given its uncertainty;
- rejects stars, plate flaws, known asteroids and comets, and anything that also shows on plates from other dates (POSS-II plates included);
- fits close neighbors together with the candidate, and writes the positions, their errors and a picture of each plate.

The tool is one Python file, `tools/precovery.py`.

## Install

You need Python 3.9 or newer.

```
python3 -m pip install -r requirements.txt
```

Tested with Python 3.9 (numpy 1.26.4, scipy 1.13.1, astropy 6.0.1, Pillow 11.3.0) and Python 3.13 (numpy 2.5.4, scipy 1.18.1, astropy 8.0.2, Pillow 13.0.0).

## Use

To check the install, run the self-test. It uses synthetic plates and needs no internet:

```
python3 tools/precovery.py --self-test
```

Then run a search, for example Quaoar on the 1950s Palomar plates:

```
python3 tools/precovery.py quaoar_1954
```

This reads `targets/precovery/quaoar_1954.json` and writes everything to `output/precovery/quaoar_1954/`. The measurements are in `results.csv` and `results.json`, with a picture of each plate in `pictures/`. A search takes from a few minutes to an hour, mostly waiting for the archives. You can also run the steps one at a time with `--plan`, `--fetch` and `--measure`.

If the DSS archive or JPL Horizons fails while plates are being found or downloaded, the tool stops and says so, rather than carry on with plates missing; run the same command again later (downloads are kept). It also stops if the object JPL Horizons finds does not go by the names in the subject. A plate whose Gaia or SkyBoT query fails is not measured, or not marked reportable, and `results.csv` says why.

To search for any other object, give its number, provisional designation or name, and the years:

```
python3 tools/precovery.py --object 50000 --years 1950 1958
python3 tools/precovery.py --object "2026 ED138" --years 1950 1958 --surveys poss1_red poss1_blue
```

Without `--surveys`, every DSS survey is searched. The search box is 1.5 times the 3-sigma uncertainty plus 3", and at least 60" (see the keys below). The results go to `output/precovery/<object>_<first year>-<last year>/`, with the settings in a file of the same name, for example `2026_ED138_1950-1958.json`. To run it again, give that file instead of a subject name.

The tool needs access to archive.stsci.edu (DSS), ssd.jpl.nasa.gov (JPL Horizons), gea.esac.esa.int (Gaia) and ssp.imcce.fr (SkyBoT). It stops if less than 2 GB of disk space is free.

For more control, copy a subject file in `targets/precovery/` (or the file written by `--object`) and edit it:

| Key | What it sets |
|---|---|
| `designation` | the object's name in the outputs |
| `names` | the object's number, name or designation, for finding it in SkyBoT's list of known objects; each must be one that JPL Horizons gives for the object (the tool stops otherwise), and the tool adds the others Horizons gives |
| `horizons_command` | the JPL Horizons command: `50000;` for a numbered object, `DES=<designation>;` for an unnumbered one |
| `years` | the first and last year to search |
| `surveys` | the DSS surveys: `poss1_red`, `poss1_blue`, `phase2_gsc1`, `poss2ukstu_red`, `poss2ukstu_blue`, `poss2ukstu_ir`, `quickv` |
| `survey_years` | optional, the first and last year for each survey |
| `comparison_surveys` | optional, the surveys whose other-date plates are used to reject fixed sources (default: all of them) |
| `box_factor`, `min_box_arcsec` | the half-width of the search box in RA and in Dec: `box_factor` times the 3-sigma uncertainty plus 3", but at least `min_box_arcsec` |
| `cutout_arcmin` | the smallest cutout to download |

## Reading the results

For each plate, `result` says whether there is one candidate within 3 sigma of the prediction, several, or none. A position is marked `reportable` only on POSS-I plates, and only when:

- there is exactly one candidate within 3 sigma;
- the plate's date and clock are established;
- a plate from another date was there to rule out fixed sources;
- the whole search box was on the cutout;
- the known-object check worked, and found nothing that could be confused with the candidate (below).

Positions on later plates are never marked reportable.

The known-object check asks SkyBoT for the asteroids and comets around the search box at the time of the plate, as seen from the plate's observatory (SkyBoT covers 1889 to 2060). It sets the target aside by its number, name or designation, then compares each source with each other object's path during the exposure:

- An asteroid whose position is known to 5" or better (or whose error SkyBoT doesn't give) rejects the sources within 10" of its path, or within 3 times its position error if that is larger. One predicted to be more than 2 magnitudes fainter than the source only adds a note.
- An asteroid whose position is uncertain by 5" to 200" rejects nothing. The sources within 3 times its error get a note, unless it is predicted to be more than 2 magnitudes fainter than them.
- An asteroid uncertain by more than 200" could be anywhere and is only listed.
- A comet rejects nothing, because comet positions and magnitudes this far back are not reliable: the sources within 60" of its path, or 3 times its error if that is larger, get a note.

The plate is not marked reportable if the candidate has such a note, if a source within 3 sigma of the prediction was rejected as a known object, or if SkyBoT does not list the target under its names but lists another object that fits the prediction in position and brightness (it may be the target under another name; it rejects nothing). `results.json` lists the objects (`known_objects`), and SkyBoT's position of the target itself (`skybot_target`), from SkyBoT's own orbit, to compare with the prediction.

## Known-answer tests

`known_answers/` gives the pass criteria and the results of three tests on objects whose old positions are well known: Quaoar on POSS-I plates, and Quaoar and Eris on UK Schmidt plates. To rerun one:

```
python3 tools/precovery.py eris_ukst
python3 tools/eval_known.py output/precovery/eris_ukst --ukst
```

## Files

- `tools/precovery.py`: the tool.
- `tools/eval_known.py`: scores a known-answer run.
- `targets/precovery/quaoar_1954.json`, `quaoar_ukst.json`, `eris_ukst.json`: the subjects of the known-answer tests.
- `targets/precovery/poss1_red_dates_hill2026.csv`, `poss1_red_times_hill2026.csv`: D. Hill's POSS-I plate dates and clock corrections.
- `known_answers/`: the known-answer criteria and results.
- `requirements.txt`, `LICENSE`, `LICENSE-TABLES.md`, `CITATION.cff`, `.zenodo.json`, `SHA256SUMS`.

## Credits

The POSS-I plate dates and clock corrections are from D. Hill, "POSS Transients Re-examined" (2026), doi:10.5281/zenodo.23549537, https://github.com/donalhill/poss-transients, CC BY 4.0. Hill derived them from Palomar's observing log (VizieR VI/25) and from asteroid trails on the plates.

The Digitized Sky Surveys were produced at the Space Telescope Science Institute under U.S. Government grant NAG W-2166. The images of these surveys are based on photographic data obtained using the Oschin Schmidt Telescope on Palomar Mountain and the UK Schmidt Telescope. The plates were processed into the present compressed digital form with the permission of these institutions. The National Geographic Society - Palomar Observatory Sky Atlas (POSS-I) was made by the California Institute of Technology with grants from the National Geographic Society. The Second Palomar Observatory Sky Survey (POSS-II) was made by the California Institute of Technology with funds from the National Science Foundation, the National Geographic Society, the Sloan Foundation, the Samuel Oschin Foundation, and the Eastman Kodak Corporation. The Oschin Schmidt Telescope is operated by the California Institute of Technology and Palomar Observatory. The UK Schmidt Telescope was operated by the Royal Observatory Edinburgh, with funding from the UK Science and Engineering Research Council (later the UK Particle Physics and Astronomy Research Council), until 1988 June, and thereafter by the Anglo-Australian Observatory. The blue plates of the southern Sky Atlas and its Equatorial Extension (together known as the SERC-J), as well as the Equatorial Red (ER), and the Second Epoch [red] Survey (SES) were all taken with the UK Schmidt. Supplemental funding for sky-survey work at the ST ScI is provided by the European Southern Observatory. No DSS images are included here.

This work has made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement. Gaia DR3: Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1; doi:10.5270/esa-qa4lep3.

The predicted positions and their uncertainties come from JPL's Horizons system (https://ssd.jpl.nasa.gov/horizons/; Giorgini et al. 1996, BAAS 28, 1158).

The known-object check uses SkyBoT, designed and developed by IMCCE, now LTE (Observatoire de Paris / CNRS): this research has made use of LTE SkyBoT VO tool (Berthier et al. 2006, ASPC 351, 367).

## Changes

- 1.1.0: the known-object check uses SkyBoT, which covers the dates of the plates, so positions can now be marked reportable (and only on plates with exactly one candidate); `--object` runs a search by object name; an error page from the DSS archive now stops the run instead of reading as "no plate here"; `results.json` writes missing numbers as null.
- 1.0.0: first release.

## License

The code is under the MIT License (`LICENSE`). Hill's two tables are under CC BY 4.0 (`LICENSE-TABLES.md`).

## How to cite

Stinga, S. I. 2026, dss-precovery (version 1.1.0), Zenodo, doi:10.5281/zenodo.23833587

All versions: doi:10.5281/zenodo.23805408

If you use the POSS-I dates or clock corrections, please also cite Hill (2026).

Stefano Ivan Stinga, Independent researcher (ORCID 0009-0005-3860-3944), stefano@stinga.jp
