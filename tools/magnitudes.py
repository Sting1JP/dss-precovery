"""magnitudes.py - magnitudes of candidates on POSS-I plates (Digitized Sky Survey scans), calibrated on Gaia DR3.

Red plates (103a-E) give Cousins R (ADES band Rc), blue plates (103a-O) Johnson B (ADES band Bj).
- Instrumental magnitude: -2.5 log10 of the fitted peak of a Gaussian with the stars' width, from the same fit as the
  position (the joint fit when a neighbor blends with the image).
- Calibration stars: Gaia DR3 stars matched within 1.5", with a proper motion and BP-RP inside the color range of the
  relation, and no other Gaia source or detection within 10".
- Catalog magnitudes: Gaia G and BP-RP converted with the relations of the Gaia DR3 documentation, section 5.5.1,
  Table 5.9.
- Calibration: catalog magnitude = a x instrumental + b, fitted to the stars within 1 mag of the candidate, the star set
  updated until it stops changing. No color term.
- rmsMag: the calibration scatter, the candidate's own noise and, for a blend, half the change between the joint and
  the one-source fit, combined and rounded up to 0.1 mag.
- No magnitude when fewer than 8 stars lie within 1 mag of the candidate, or when the object moved more than half a
  star width during the exposure (a peak fitted to a trailed image comes out too faint).
"""
import math
import re

import numpy as np

# catalog magnitude = G - (c0 + c1 x + c2 x^2 + ...), x = BP-RP; the relation's color range and scatter (mag)
RELATIONS = {
    "Rc": ((-0.02275, 0.3961, -0.1243, -0.01396, 0.003775), (0.0, 4.0), 0.032),
    "Bj": ((0.01448, -0.6874, -0.3604, 0.06718, -0.006061), (-0.5, 4.0), 0.063),
}
MATCH_ARCSEC = 1.5       # a detection is this Gaia star if within this distance
ISOLATION_ARCSEC = 10.0  # no other Gaia source or detection within this distance of a calibration star
WINDOW_MAG = 1.0         # calibration stars within this much of the candidate's magnitude
MIN_STARS = 8
MAX_ITER = 6
TRAIL_LIMIT_FWHM = 0.5   # no magnitude when the trail is longer than this many star widths (FWHM)


def plate_band(h):
    """ADES band of a POSS-I plate from its header: 'Rc' for a red (XE) plate, 'Bj' for a blue (XO) plate, else None"""
    region = str(h.get("REGION", "")).strip()
    if "palomar" not in str(h.get("TELESCOP", "")).lower() or not re.match(r"^X[EO]\d{3}$", region):
        return None
    return "Rc" if region.startswith("XE") else "Bj"


def catalog_mag(g, bprp, band):
    """Rc or Bj from Gaia G and BP-RP (Gaia DR3 documentation, Table 5.9); nan outside the relation's color range"""
    coef, (lo, hi), _ = RELATIONS[band]
    g = np.asarray(g, float); x = np.asarray(bprp, float)
    poly = sum(c * x ** k for k, c in enumerate(coef))
    ok = np.isfinite(g) & np.isfinite(x) & (x > lo) & (x < hi)
    return np.where(ok, g - poly, np.nan)


def select_stars(det_xi, det_eta, gaia_xi, gaia_eta, gaia_g, gaia_bprp, gaia_has_pm, band):
    """the detections usable as calibration stars, whatever instrumental magnitude is then measured for them:
    (indices into the detections, catalog magnitudes, counts of what was set aside). det_*: every detection on the
    plate (standard coordinates, arcsec); gaia_*: every Gaia source at the plate epoch (G and BP-RP nan where Gaia has
    none, has_pm 1 or 0)"""
    from scipy.spatial import cKDTree
    n_det, n_gaia = len(det_xi), len(gaia_xi)
    idx, cat = [], []
    why = dict(not_matched=0, crowded=0, no_color_or_pm=0, color_range=0)
    if n_det == 0 or n_gaia < 2:
        return np.zeros(0, int), np.zeros(0), why
    gt = cKDTree(np.column_stack([gaia_xi, gaia_eta]))
    dt_ = cKDTree(np.column_stack([det_xi, det_eta]))
    d, j = gt.query(np.column_stack([det_xi, det_eta]), k=2)
    for i in range(n_det):
        if d[i, 0] > MATCH_ARCSEC:
            why["not_matched"] += 1; continue
        if d[i, 1] < ISOLATION_ARCSEC or len(dt_.query_ball_point([det_xi[i], det_eta[i]], ISOLATION_ARCSEC)) > 1:
            why["crowded"] += 1; continue
        k = int(j[i, 0])
        if not (np.isfinite(gaia_g[k]) and np.isfinite(gaia_bprp[k])) or gaia_has_pm[k] < 0.5:
            why["no_color_or_pm"] += 1; continue
        m = float(catalog_mag(gaia_g[k], gaia_bprp[k], band))
        if not np.isfinite(m):
            why["color_range"] += 1; continue
        idx.append(i); cat.append(m)
    return np.array(idx, int), np.array(cat), why


def calibration_stars(det_xi, det_eta, det_amp, det_flux, gaia_xi, gaia_eta, gaia_g, gaia_bprp, gaia_has_pm, band):
    """the plate's calibration stars for star-like images: (instrumental magnitudes from the fitted peaks, catalog
    magnitudes, counts of what was set aside); det_amp the fitted peaks, det_flux the aperture fluxes"""
    idx, cat, why = select_stars(det_xi, det_eta, gaia_xi, gaia_eta, gaia_g, gaia_bprp, gaia_has_pm, band)
    amp = np.asarray(det_amp, float)[idx]; flux = np.asarray(det_flux, float)[idx]
    ok = np.isfinite(amp) & (amp > 0) & (flux > 0)
    why["bad_fit"] = int((~ok).sum())
    return -2.5 * np.log10(amp[ok]), cat[ok], why


def calibrate(inst, cat, inst_obj, window=WINDOW_MAG, min_stars=MIN_STARS, max_iter=MAX_ITER):
    """straight-line calibration for one candidate: start from all stars, then keep those within window mag of the
    candidate's magnitude and fit again, until the set stops changing. Returns dict(mag, a, b, n, scatter), or None
    when fewer than min_stars lie within the window or the slope is not positive (calibration_problem says which)."""
    return _calibrate(inst, cat, inst_obj, window, min_stars, max_iter)[0]


def calibration_problem(inst, cat, inst_obj, window=WINDOW_MAG, min_stars=MIN_STARS, max_iter=MAX_ITER):
    """why calibrate() gives None for these inputs ('' if it does not)"""
    return _calibrate(inst, cat, inst_obj, window, min_stars, max_iter)[1]


def _calibrate(inst, cat, inst_obj, window, min_stars, max_iter):
    inst = np.asarray(inst, float); cat = np.asarray(cat, float)
    if not np.isfinite(inst_obj):
        return None, "no instrumental magnitude"
    if len(inst) < min_stars:
        return None, "%d calibration stars on the plate, fewer than %d" % (len(inst), min_stars)

    def line(s):
        A = np.column_stack([inst[s], np.ones(int(s.sum()))])
        c, *_ = np.linalg.lstsq(A, cat[s], rcond=None)
        return float(c[0]), float(c[1])
    sel = np.ones(len(inst), bool)
    windowed = False  # True once the star set is the stars within the window
    for _ in range(max_iter):
        a, b = line(sel)
        new = np.abs(cat - (a * inst_obj + b)) < window
        if new.sum() < min_stars:
            break
        windowed = True
        if np.array_equal(new, sel):
            break
        sel = new
    if not windowed:
        return None, "fewer than %d calibration stars within %g mag of the candidate" % (min_stars, window)
    a, b = line(sel)
    if not a > 0:
        return None, "the calibration's slope is not positive (%.3f)" % a
    resid = cat[sel] - (a * inst[sel] + b)
    return dict(mag=a * inst_obj + b, a=a, b=b, n=int(sel.sum()), scatter=float(np.std(resid))), ""


def round_up(x, step=0.1):
    """x rounded up to the next multiple of step (an exact multiple stays), and at least one step"""
    return round(max(math.ceil(x / step - 1e-6), 1) * step, 6)


def rms_mag(scatter, terms):
    """the magnitude's uncertainty from the calibration scatter and any other terms (dict of name -> mag), combined
    in quadrature: (unrounded, rounded up to 0.1, parts); nan if a term is not finite"""
    parts = dict(scatter=scatter, **terms)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in parts.values()):
        return float("nan"), float("nan"), parts
    tot = math.sqrt(sum(v * v for v in parts.values()))
    return tot, round_up(tot), parts


def peak_noise_mag(a, amp, amp_noise):
    """the noise of a fitted peak amp, in calibrated mag: amp_noise is the scatter of fitted peaks at empty places of
    the plate, a the calibration's slope"""
    if not (np.isfinite(amp_noise) and np.isfinite(amp) and amp > 0):
        return float("nan")
    return a * 2.5 / math.log(10.0) * amp_noise / amp


def deblend_mag(a, amp, amp_one_source):
    """half the change in calibrated mag between the joint fit (amp) and a one-source fit (amp_one_source); 0 when the
    image was not fitted jointly"""
    if amp_one_source is None or not (np.isfinite(amp_one_source) and amp_one_source > 0 and amp > 0):
        return 0.0
    return 0.5 * a * 2.5 * abs(math.log10(amp_one_source / amp))


def trail_too_long(trail_arcsec, fwhm_arcsec):
    """True when the object moved more than TRAIL_LIMIT_FWHM star widths during the exposure (or either is unknown)"""
    if not (np.isfinite(trail_arcsec) and np.isfinite(fwhm_arcsec) and fwhm_arcsec > 0):
        return True
    return trail_arcsec > TRAIL_LIMIT_FWHM * fwhm_arcsec


def candidate_mag(inst_star, cat_star, amp, amp_noise, band, amp_one_source=None):
    """the magnitude of one candidate, as a dict for the results (band, band_mag, rms_mag, ...), or a dict with
    band_mag nan and the reason in mag_note"""
    out = dict(band=band, band_mag=float("nan"), rms_mag=float("nan"))
    if not (np.isfinite(amp) and amp > 0):
        out["mag_note"] = "no fitted peak"
        return out
    cal, why = _calibrate(inst_star, cat_star, -2.5 * math.log10(amp), WINDOW_MAG, MIN_STARS, MAX_ITER)
    if cal is None:
        out["mag_note"] = why
        return out
    err, rnd, parts = rms_mag(cal["scatter"], dict(noise=peak_noise_mag(cal["a"], amp, amp_noise),
                                                   deblend=deblend_mag(cal["a"], amp, amp_one_source)))
    out.update(band_mag=float(cal["mag"]), band_mag_err=float(err), rms_mag=rnd,
               mag_calibration=dict(stars=cal["n"], slope=round(cal["a"], 4), zero=round(cal["b"], 4),
                                    scatter=round(cal["scatter"], 3), noise=round(parts["noise"], 3),
                                    deblend=round(parts["deblend"], 3)))
    return out
