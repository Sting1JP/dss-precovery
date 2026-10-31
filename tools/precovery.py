#!/usr/bin/env python3
"""precovery.py - look for a known, slow-moving object on old survey plates (Digitized Sky Survey scans).

Steps (each can be run alone; results go to output/precovery/<subject>/):
  --plan     ephemeris along the track; ask the DSS which plate covers each part of it and when it was taken;
             keep the plates whose date puts the object on that same plate.
  --fetch    for each kept plate: exact position at the plate's mid-exposure (JPL Horizons, plate site),
             a cutout of the target plate there, and cutouts of the same sky from the other surveys (other dates).
  --measure  astrometry refit to Gaia DR3 moved to the plate epoch; sources in the search box that are not
             Gaia stars and are not on the other-date plates; offsets from the prediction; pictures to blink.
  (no step)  all three.
  --self-test  synthetic plates; no network.

A subject is a file in targets/precovery/, or an object given with --object NAME --years FIRST LAST.

Python 3.9+, numpy, scipy, astropy, Pillow. Network: archive.stsci.edu (DSS), ssd.jpl.nasa.gov (Horizons),
gea.esac.esa.int (Gaia archive), ssp.imcce.fr (SkyBoT, for the known asteroids and comets on each plate).
"""
import argparse, csv, datetime as dt, io, json, math, os, re, shutil, ssl, sys, time, urllib.parse, urllib.request

import numpy as np
from scipy import ndimage

try:  # no downloads by astropy itself; only astropy.io.fits is used
    from astropy.utils import data as _apdata, iers as _apiers
    _apdata.conf.allow_internet = False
    _apiers.conf.auto_download = False
except Exception:  # noqa
    pass

VERSION = "1.1.0"
HOSTS = {"archive.stsci.edu", "ssd.jpl.nasa.gov", "gea.esac.esa.int", "ssp.imcce.fr"}
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,80}$")
UA = "dss-precovery/%s (+https://github.com/Sting1JP/dss-precovery)" % VERSION
SURVEYS = ["poss1_red", "poss1_blue", "phase2_gsc1", "poss2ukstu_red", "poss2ukstu_blue", "poss2ukstu_ir", "quickv"]
# telescope -> (site for the ephemerides, Horizons and SkyBoT; MPC code for reporting DSS measurements)
SITES = [("palomar", "675", "261"), ("uk 48", "413", "260"), ("uk schmidt", "413", "260"),
         ("siding", "413", "260"), ("eso", "809", "262")]

# ----------------------------------------------------------------------------------------------- utilities

def log(*a):
    print(*a, flush=True)


def _check_url(url):
    u = urllib.parse.urlsplit(url)
    if u.scheme != "https" or u.hostname not in HOSTS:
        raise RuntimeError("refused: only HTTPS to %s (got %s)" % (", ".join(sorted(HOSTS)), url[:120]))


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context()), _SafeRedirect)


def http_get(url, tries=3, timeout=120, max_bytes=64 * 1024 * 1024):
    _check_url(url)
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with _OPENER.open(req, timeout=timeout) as r:
                data = r.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise RuntimeError("answer larger than %d bytes" % max_bytes)
                return data
        except Exception as e:  # noqa
            last = e
            if "refused:" in str(e) or "larger than" in str(e):
                break
            time.sleep(3 * (k + 1))
    raise RuntimeError("GET failed after %d tries: %s (%s)" % (tries, url[:160], last))


def json_clean(x):
    """for writing standard JSON: NaN and infinities become null, numpy numbers plain numbers, tuples lists"""
    if isinstance(x, dict):
        return {str(k): json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [json_clean(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, obj):
    with open(path + ".part", "w") as f:
        json.dump(json_clean(obj), f, indent=1, default=str, allow_nan=False)
    os.replace(path + ".part", path)


def safe_name(s):
    """a file-name piece from header or subject text: anything odd becomes '_', then strictly checked"""
    t = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(s).strip()).strip("._") or "x"
    t = t[:80]
    if not SAFE.match(t):
        raise ValueError("unsafe name %r" % s)
    return t


def disk_ok(path, need_bytes=2 * 1024 ** 3):
    free = shutil.disk_usage(path).free
    if free < need_bytes:
        raise RuntimeError("less than %.1f GB free on the disk of %s; stopping" % (need_bytes / 1024 ** 3, path))


def jd_from_datetime(t):
    return (t - dt.datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0 + 2451545.0


def datetime_from_jd(jd):
    return dt.datetime(2000, 1, 1, 12, 0, 0) + dt.timedelta(days=jd - 2451545.0)


def decyear(t):
    y0 = dt.datetime(t.year, 1, 1); y1 = dt.datetime(t.year + 1, 1, 1)
    return t.year + (t - y0).total_seconds() / (y1 - y0).total_seconds()


def tangent(ra, dec, ra0, dec0):
    """standard coordinates (arcsec) of ra,dec (deg) about ra0,dec0"""
    a = np.radians(ra); d = np.radians(dec); a0 = math.radians(ra0); d0 = math.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0


def untangent(xi, eta, ra0, dec0):
    x = np.radians(np.asarray(xi) / 3600.0); y = np.radians(np.asarray(eta) / 3600.0)
    a0 = math.radians(ra0); d0 = math.radians(dec0)
    rho = np.hypot(x, y); c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.where(rho == 0, d0, np.arcsin(np.cos(c) * np.sin(d0) + y * np.sin(c) * np.cos(d0) / np.where(rho == 0, 1, rho)))
        ra = a0 + np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360.0, np.degrees(dec)

# ----------------------------------------------------------------------------------------- DSS plate header

def dss_pix2sky(h, x, y):
    """DSS plate solution. x,y are 1-based FITS pixels of the cutout. Checked against astropy on a real DSS file."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = x + h["CNPIX1"] - 0.5
    Y = y + h["CNPIX2"] - 0.5
    xm = (h["PPO3"] - h["XPIXELSZ"] * X) / 1000.0
    ym = (h["YPIXELSZ"] * Y - h["PPO6"]) / 1000.0
    A = [float(h.get("AMDX%d" % i, 0.0)) for i in range(1, 21)]
    B = [float(h.get("AMDY%d" % i, 0.0)) for i in range(1, 21)]
    r2 = xm ** 2 + ym ** 2
    xi = (A[0] * xm + A[1] * ym + A[2] + A[3] * xm ** 2 + A[4] * xm * ym + A[5] * ym ** 2 + A[6] * r2
          + A[7] * xm ** 3 + A[8] * xm ** 2 * ym + A[9] * xm * ym ** 2 + A[10] * ym ** 3 + A[11] * xm * r2 + A[12] * xm * r2 ** 2)
    eta = (B[0] * ym + B[1] * xm + B[2] + B[3] * ym ** 2 + B[4] * xm * ym + B[5] * xm ** 2 + B[6] * r2
           + B[7] * ym ** 3 + B[8] * ym ** 2 * xm + B[9] * ym * xm ** 2 + B[10] * xm ** 3 + B[11] * ym * r2 + B[12] * ym * r2 ** 2)
    xi = np.radians(xi / 3600.0); eta = np.radians(eta / 3600.0)
    ra0 = math.radians((h["PLTRAH"] + h["PLTRAM"] / 60.0 + h["PLTRAS"] / 3600.0) * 15.0)
    sg = -1.0 if str(h["PLTDECSN"]).strip() == "-" else 1.0
    de0 = math.radians(sg * (h["PLTDECD"] + h["PLTDECM"] / 60.0 + h["PLTDECS"] / 3600.0))
    t = math.tan(de0)
    ra = np.arctan2(xi / math.cos(de0), 1.0 - eta * t) + ra0
    dec = np.arctan(((eta + t) * np.cos(ra - ra0)) / (1.0 - eta * t))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def dss_sky2pix(h, ra, dec, x0=None, y0=None):
    """inverse of dss_pix2sky by Newton iteration (numeric Jacobian)"""
    ra = np.atleast_1d(np.asarray(ra, float)); dec = np.atleast_1d(np.asarray(dec, float))
    nx, ny = h["NAXIS1"], h["NAXIS2"]
    x = np.full(ra.shape, (nx + 1) / 2.0 if x0 is None else x0)
    y = np.full(ra.shape, (ny + 1) / 2.0 if y0 is None else y0)
    rc, dc = dss_pix2sky(h, (nx + 1) / 2.0, (ny + 1) / 2.0)
    rc = float(rc); dc = float(dc)
    txi, teta = tangent(ra, dec, rc, dc)
    for _ in range(20):
        a, d = dss_pix2sky(h, x, y); xi, eta = tangent(a, d, rc, dc)
        a1, d1 = dss_pix2sky(h, x + 1, y); xi1, eta1 = tangent(a1, d1, rc, dc)
        a2, d2 = dss_pix2sky(h, x, y + 1); xi2, eta2 = tangent(a2, d2, rc, dc)
        J11 = xi1 - xi; J21 = eta1 - eta; J12 = xi2 - xi; J22 = eta2 - eta
        det = J11 * J22 - J12 * J21
        ex = txi - xi; ey = teta - eta
        dx = (J22 * ex - J12 * ey) / det; dy = (-J21 * ex + J11 * ey) / det
        x = x + dx; y = y + dy
        if np.all(np.hypot(dx, dy) < 1e-4):
            break
    return x, y


# POSS-I (Palomar, 1949-1958) plate dates and times, following D. Hill, "POSS Transients Re-examined" (2026,
# doi:10.5281/zenodo.23549537) and Palomar's observing log (VizieR VI/25):
#  * the header time is the exposure start, in hours and hundredths of an hour ("05:82" = 05:49:12 UT);
#  * the header date is usually the local evening date, one day before the UT date. Hill's corrected date for each
#    red (XE) plate is read from targets/precovery/poss1_red_dates_hill2026.csv; a blue (XO) plate takes the date of
#    the red plate with the same field and number;
#  * the written start is 0.6-1.2 min late: Hill's per-plate clock offset is used where his table has the plate,
#    otherwise the survey median of 0.8 min.
POSS1_CLOCK_MIN = 0.8
_POSS1_DATES = None
_POSS1_CLOCK = None
HILL_TABLES = ("poss1_red_dates_hill2026.csv", "poss1_red_times_hill2026.csv")


def table_path(name):
    """a data table shipped with the tool, in targets/precovery/ next to tools/"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "targets", "precovery", name)


def poss1_block_offset(label):
    """Hill's rule from plate serial labels: days to add to the header date (None: not pinned down)"""
    if label <= 28 or 639 <= label <= 962 or 1611 <= label <= 1616:
        return 0
    if 30 <= label <= 637 or 968 <= label <= 1610 or 1617 <= label <= 1618:
        return 1
    return None


def load_poss1_dates(path=None):
    global _POSS1_DATES
    if _POSS1_DATES is not None and path is None:
        return _POSS1_DATES
    p = path or table_path(HILL_TABLES[0])
    tab = {}
    if os.path.exists(p):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                reg = r.get("plate", "").strip()
                if not re.match(r"^XE\d{3}$", reg):
                    continue
                try:
                    lab = int(float(r["label"])) if r.get("label") else None
                except ValueError:
                    lab = None
                off = r.get("offset_days", "").strip()
                tab[reg] = dict(label=lab, header_date=r.get("header_date", "").strip(),
                                offset=(int(float(off)) if off else None), source=r.get("source", "").strip())
    _POSS1_DATES = tab
    return tab


def load_poss1_clock(path=None):
    """Hill's per-plate times (targets/precovery/poss1_red_times_hill2026.csv): plate -> clock offset tau in minutes
    (true start = written start + tau), half its 16-84% range, on_line, and Hill's mid-exposure UT.
    No file: an empty table."""
    global _POSS1_CLOCK
    if _POSS1_CLOCK is not None and path is None:
        return _POSS1_CLOCK
    p = path or table_path(HILL_TABLES[1])
    tab = {}
    if os.path.exists(p):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                reg = r.get("plate", "").strip()
                if not re.match(r"^XE\d{3}$", reg):
                    continue
                try:
                    tau, q16, q84 = float(r["tau_median"]), float(r["tau_q16"]), float(r["tau_q84"])
                    ut = dt.datetime.fromisoformat(r["ut"].strip()[:26]) if r.get("ut") else None
                    n_on = int(float(r.get("on_line") or 0))
                except (KeyError, ValueError) as e:
                    raise ValueError("cannot read plate %s in %s (%s)" % (reg, p, e))
                tab[reg] = dict(tau=tau, sig=0.5 * (q84 - q16), on_line=n_on, ut=ut)
    _POSS1_CLOCK = tab
    return tab


def _split_dateobs(d, ut):
    """-> (date, hh, mm, ss, had_seconds) from DATE-OBS (ISO with or without seconds, or DD/MM/YY) and UT"""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:T(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d*)?))?)?$", d)
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if m.group(4) is not None:
            return dt.date(y, mo, da), int(m.group(4)), int(m.group(5)), float(m.group(6) or 0.0), m.group(6) is not None
        d0 = dt.date(y, mo, da)
    else:
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", d)
        if not m:
            raise ValueError("cannot read DATE-OBS %r" % d)
        da, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 1900 if y < 100 else y
        d0 = dt.date(y, mo, da)
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d*)?))?$", ut)
    if not m:
        raise ValueError("no time of day (DATE-OBS %r, UT %r)" % (d, ut))
    return d0, int(m.group(1)), int(m.group(2)), float(m.group(3) or 0.0), m.group(3) is not None


def plate_info(h):
    """date/time of the plate, exposure, telescope -> dict (times as datetime UT).
    'start' is the corrected exposure start; 'time_note' says how the header was read; 'reportable' is False when
    the date or time convention is not established for this plate."""
    d = str(h.get("DATE-OBS", "")).strip()
    ut = str(h.get("UT", "") or "").strip()
    day, hh, mm, ss, _ = _split_dateobs(d, ut)
    flags = []
    reportable = True
    if ut and "T" in d:  # both carry a time of day: they must agree
        m_ = re.match(r"^(\d{1,2}):(\d{2})", ut)
        if not m_ or (int(m_.group(1)), int(m_.group(2))) != (hh, mm):
            raise ValueError("DATE-OBS %r and UT %r give different times" % (d, ut))
    try:
        exp = float(h["EXPOSURE"])  # minutes
    except (KeyError, TypeError, ValueError):
        raise ValueError("no EXPOSURE in the header")
    if not (0.0 < exp < 600.0):
        raise ValueError("EXPOSURE %r min is not plausible" % exp)
    tel = str(h.get("TELESCOP", "")).strip()
    region = str(h.get("REGION", "")).strip()
    label = str(h.get("PLTLABEL", "")).strip()
    site, mpc = None, None
    for key, s, m in SITES:
        if key in tel.lower():
            site, mpc = s, m
            break
    if site is None:
        raise ValueError("unknown telescope %r: no site or MPC code" % tel)
    epoch = h.get("EPOCH", None)
    if epoch is not None:  # EPOCH is computed from the header's own date and time (before any correction)
        try:
            t_hdr = dt.datetime(day.year, day.month, day.day) + dt.timedelta(hours=hh, minutes=mm, seconds=ss)
            t_hdr2 = dt.datetime(day.year, day.month, day.day) + dt.timedelta(hours=hh + mm / 100.0)
            if min(abs(float(epoch) - decyear(t_hdr)), abs(float(epoch) - decyear(t_hdr2))) > 0.0005:
                flags.append("EPOCH %.5f disagrees with DATE-OBS %s" % (float(epoch), d))
                reportable = False
        except Exception:  # noqa
            pass
    poss1 = "palomar" in tel.lower() and re.match(r"^X[EO]\d{3}$", region) is not None
    if poss1:
        if ss != 0.0:
            flags.append("POSS-I time has seconds %r (unexpected)" % ss)
        clk = load_poss1_clock().get("XE" + region[2:])  # a blue (XO) plate takes its red partner's clock
        tau = clk["tau"] if clk else -POSS1_CLOCK_MIN
        clock_sig = clk["sig"] if clk else 0.6
        start = dt.datetime(day.year, day.month, day.day) + dt.timedelta(hours=hh + mm / 100.0) + dt.timedelta(minutes=tau)
        tab = load_poss1_dates()
        num = re.match(r"^[EO](\d+)", label)
        lab = int(num.group(1)) if num else None
        ent = tab.get("XE" + region[2:])
        off, src = None, ""
        if ent is not None:
            if lab is not None and ent["label"] is not None and ent["label"] != lab:
                flags.append("plate number %s differs from the red plate's %s in the date table" % (lab, ent["label"]))
                reportable = False
            if ent["header_date"] and ent["header_date"] != day.isoformat():
                flags.append("header date %s differs from the date table's %s" % (day.isoformat(), ent["header_date"]))
                reportable = False
            off, src = ent["offset"], "Hill 2026 (%s)" % ent["source"]
        elif lab is not None:
            off, src = poss1_block_offset(lab), "Hill 2026 plate-number rule (field not in the table)"
        if off is None:
            flags.append("POSS-I date not established (%s)" % (src or "no plate number"))
            reportable = False
            off = 1  # the usual case, so the plan can still look; never reported
        start = start + dt.timedelta(days=off)
        if clk:
            csrc = "Hill 2026 per-plate clock (%s, on_line %d)" % ("red partner" if region.startswith("XO") else "this plate",
                                                                   clk["on_line"])
            if region.startswith("XE") and clk["ut"] is not None:
                dmid = (start + dt.timedelta(minutes=exp / 2.0) - clk["ut"]).total_seconds() / 60.0
                if abs(dmid) > 1.0:
                    flags.append("mid-exposure differs from Hill's plate time %s by %+.1f min" % (clk["ut"].isoformat()[:19], dmid))
                    reportable = False
        else:
            csrc = "survey median"
        time_note = "POSS-I: hours+hundredths, start, clock %+.2f min (%s); date %+d d from %s" % (tau, csrc, off, src)
        if region.startswith("XE") and exp < 10:
            flags.append("red exposure %.1f min may be a slipped decimal (x10)" % exp)
            reportable = False
    else:
        if mm > 59:
            raise ValueError("minutes %d > 59 in %r on a non-POSS-I plate: convention unknown" % (mm, d))
        t_hdr = dt.datetime(day.year, day.month, day.day) + dt.timedelta(hours=hh, minutes=mm, seconds=ss)
        if "uk schmidt" in tel.lower() or "uk 48" in tel.lower():
            # UK Schmidt DSS headers give the middle of the exposure (known-answer tests on Quaoar and Eris)
            start = t_hdr - dt.timedelta(minutes=exp / 2.0)
            time_note = "UK Schmidt: header time read as MID-exposure (not marked reportable)"
        else:
            start = t_hdr
            time_note = "read as UT hours:minutes, exposure start (not checked against a known answer)"
        reportable = False
    return dict(start=start, mid=start + dt.timedelta(minutes=exp / 2.0), exp_min=exp, telescope=tel, site=site,
                mpc=mpc, plate=safe_name("%s_%s" % (region, str(h.get("PLATEID", "")).strip())),
                label=label, flag="; ".join(flags), time_note=time_note, reportable=reportable,
                header_dateobs=d, clock_sigma_min=(clock_sig if poss1 else None))

def comparison_info(h):
    """plate_info() for an other-date (comparison) plate. A comparison only needs its date to within a day, so a
    plate whose telescope or time convention is not established for reporting (e.g. POSS-II, 'Oschin Schmidt - D')
    is still used, dated by the header's EPOCH (decimal year) or else by the date in DATE-OBS. Never reportable;
    raises if neither gives a date."""
    info = _comparison_info(h)
    em = str(h.get("EMULSION", "")).strip().upper().replace(" ", "")
    tel = str(h.get("TELESCOP", "")).lower()
    if (em in ("IIIAJ", "IIIAF", "IVN") or "oschin" in tel) and info["mid"].year < 1983:
        info["flag"] = ((info["flag"] + "; ") if info["flag"] else "") + \
            "header date %s implausible for a %s plate (survey of 1985-2000); used only as a comparison" % (
                info["mid"].isoformat()[:10], h.get("EMULSION", "POSS-II"))
    return info


def _comparison_info(h):
    try:
        return plate_info(h)
    except ValueError as e:
        mid, how = None, ""
        try:
            ep = float(h.get("EPOCH", None))
            if 1900.0 < ep < 2100.0:
                y = int(ep); y0 = dt.datetime(y, 1, 1); y1 = dt.datetime(y + 1, 1, 1)
                mid = y0 + dt.timedelta(seconds=(ep - y) * (y1 - y0).total_seconds())
                how = "the header EPOCH %.4f" % ep
        except (TypeError, ValueError):
            pass
        if mid is None:
            try:
                day, hh, mm, ss, _ = _split_dateobs(str(h.get("DATE-OBS", "")).strip(), str(h.get("UT", "") or "").strip())
            except ValueError:
                raise e
            if not (1900 < day.year < 2100):
                raise e
            frac = (hh + mm / 60.0 + ss / 3600.0) if mm <= 59 else (hh + mm / 100.0)
            mid = dt.datetime(day.year, day.month, day.day) + dt.timedelta(hours=min(frac, 23.99))
            how = "the DATE-OBS date"
        try:
            exp = float(h.get("EXPOSURE", 0.0) or 0.0)
        except (TypeError, ValueError):
            exp = 0.0
        return dict(start=mid, mid=mid, exp_min=exp, telescope=str(h.get("TELESCOP", "")).strip(), site=None, mpc=None,
                    plate=safe_name("%s_%s" % (str(h.get("REGION", "")).strip(), str(h.get("PLATEID", "")).strip())),
                    label=str(h.get("PLTLABEL", "")).strip(), flag="comparison only (%s)" % e,
                    time_note="dated by %s (comparison only)" % how, reportable=False,
                    header_dateobs=str(h.get("DATE-OBS", "")).strip())

# --------------------------------------------------------------------------------------------- the network

DSS_RETRY_WAIT = 10.0  # seconds before asking again after an error page (the archive answers so when busy)


def dss_page(raw):
    """the DSS archive's answer when it is not an image: ('none', text) for its page saying that the survey has no
    plate here ('Calibration and image data not available for field', with an empty plate list), ('error', text)
    for anything else (busy, out of memory or disk, maintenance, a bad request)"""
    txt = raw.decode("latin-1", "replace")
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", txt)).strip()
    m = re.search(r"\n-{4}[ -]+\n(.*?)FATAL ERROR", txt, re.S)
    if "Calibration and image data not available for field" in txt and m and not m.group(1).strip():
        return "none", plain
    return "error", plain


def fetch_dss(survey, ra, dec, size_arcmin, path=None, tries=3):
    """a DSS cutout (astropy HDU), or None where the survey has no plate; an error page from the archive is asked
    again, and raises if it stays"""
    if survey not in SURVEYS:
        raise ValueError("unknown survey %r" % survey)
    size_arcmin = min(float(size_arcmin), 60.0)  # the DSS limit
    # position in decimal degrees, which avoids sexagesimal rounding
    q = urllib.parse.urlencode(dict(v=survey, r="%.6f" % (ra % 360.0), d="%.6f" % dec, e="J2000",
                                    h="%.2f" % size_arcmin, w="%.2f" % size_arcmin, f="fits", c="none"))
    for k in range(tries):
        raw = http_get("https://archive.stsci.edu/cgi-bin/dss_search?" + q)
        if raw[:6] == b"SIMPLE":
            break
        kind, text = dss_page(raw)
        if kind == "none":
            return None
        if k == tries - 1:
            raise RuntimeError("DSS %s at %.5f %+.5f: the archive answered with an error page, not an image, %d times: %s"
                               % (survey, ra, dec, tries, text[:300]))
        time.sleep(DSS_RETRY_WAIT * (k + 1))
    from astropy.io import fits
    hdu = fits.open(io.BytesIO(raw))[0]
    if hdu.data is None or "CNPIX1" not in hdu.header or "AMDX1" not in hdu.header:
        raise RuntimeError("DSS %s at %.5f %+.5f: a FITS answer without an image or a plate solution" % (survey, ra, dec))
    if path:
        with open(path + ".part", "wb") as f:
            f.write(raw)
        os.replace(path + ".part", path)
    return hdu


def horizons_at(command, site, jd_ut, cache=None):
    """astrometric RA/Dec (ICRF), rates, V, 3-sigma at one UT time, from the given site code."""
    key = "%s|%s|%.6f" % (command, site, jd_ut)
    if cache is not None and key in cache:
        return cache[key]
    q = urllib.parse.urlencode({
        "format": "text", "COMMAND": "'%s'" % command, "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'%s@399'" % site,
        "START_TIME": "'JD %.6f'" % jd_ut, "STOP_TIME": "'JD %.6f'" % (jd_ut + 0.001), "STEP_SIZE": "'1 m'",
        "TIME_TYPE": "'UT'", "QUANTITIES": "'1,3,9,36,37'", "ANG_FORMAT": "'DEG'", "CSV_FORMAT": "'YES'",
        "EXTRA_PREC": "'YES'"})
    txt = http_get("https://ssd.jpl.nasa.gov/api/horizons.api?" + q).decode("utf-8", "replace")
    out = parse_horizons_first(txt)
    if cache is not None:
        cache[key] = out
    return out


def parse_horizons_first(txt):
    """columns by header name; the 3-sigma ellipse: SMAA, SMIA (arcsec) and Theta, the angle from +RA toward +Dec
    (Horizons: 'clockwise angle from the direction of increasing RA ... in the direction of increasing DEC')"""
    if "$$SOE" not in txt or "$$EOE" not in txt:
        raise RuntimeError("Horizons: no ephemeris\n" + txt[-800:])
    head = txt.split("$$SOE")[0].strip().splitlines()
    hdr = [l for l in head if "R.A." in l and "Date" in l]
    if not hdr:
        raise RuntimeError("Horizons: no column header")
    names = [v.strip() for v in hdr[-1].split(",")]
    row = txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()[0]
    p = [v.strip() for v in row.split(",")]
    col = {n: p[i] for i, n in enumerate(names) if n and i < len(p)}

    def num(prefix):
        for n, v in col.items():
            if n.startswith(prefix):
                try:
                    return float(v)
                except ValueError:
                    return float("nan")
        return float("nan")
    tm = re.search(r"Target body name:\s*(.*?)\s*\{source", txt)
    return dict(ra=num("R.A."), dec=num("DEC"), dra=num("dRA*cosD"), ddec=num("d(DEC)/dt"), V=num("APmag"),
                ra3=num("RA_3sigma"), dec3=num("DEC_3sigma"), smaa=num("SMAA_3sig"), smia=num("SMIA_3sig"),
                theta=num("Theta"), target=tm.group(1).strip() if tm else "")


def horizons_track(command, start, stop, step="5 d"):
    q = urllib.parse.urlencode({
        "format": "text", "COMMAND": "'%s'" % command, "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'500@399'", "START_TIME": "'%s'" % start, "STOP_TIME": "'%s'" % stop,
        "STEP_SIZE": "'%s'" % step, "QUANTITIES": "'1,9,23,36'", "ANG_FORMAT": "'DEG'", "CSV_FORMAT": "'YES'"})
    txt = http_get("https://ssd.jpl.nasa.gov/api/horizons.api?" + q).decode("utf-8", "replace")
    if "$$SOE" not in txt or "$$EOE" not in txt:
        raise RuntimeError("Horizons: no ephemeris\n" + txt[-800:])
    tm = re.search(r"Target body name:\s*(.*?)\s*\{source", txt)
    rows = []
    for line in txt.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        p = [v.strip() for v in line.split(",")]
        t = dt.datetime.strptime(p[0][:17], "%Y-%b-%d %H:%M")
        rows.append(dict(t=t, ra=float(p[3]), dec=float(p[4]), V=float(p[5]), elong=float(p[7]),
                         ra3=float(p[9]), dec3=float(p[10]), target=tm.group(1).strip() if tm else ""))
    return rows


GAIA_CAP = 300000


def gaia_cone(ra, dec, radius_deg, gmax=21.0):
    """every Gaia DR3 source in the cone (with or without proper motion); refuses a result at the row cap"""
    adql = ("SELECT TOP %d source_id, ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source "
            "WHERE 1=CONTAINS(POINT('ICRS',ra,dec), CIRCLE('ICRS',%.7f,%.7f,%.5f)) "
            "AND (phot_g_mean_mag < %.2f OR phot_g_mean_mag IS NULL)" % (GAIA_CAP, ra, dec, radius_deg, gmax))
    q = urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=adql, MAXREC=str(GAIA_CAP)))
    raw = http_get("https://gea.esac.esa.int/tap-server/tap/sync?" + q).decode("utf-8", "replace")
    if not raw.startswith("source_id"):
        raise RuntimeError("Gaia archive: unexpected answer %r" % raw[:200])
    rd = csv.DictReader(io.StringIO(raw))
    rows = []
    for r in rd:
        def f(k):
            try:
                return float(r[k])
            except (ValueError, TypeError):
                return float("nan")
        rows.append((f("ra"), f("dec"), f("pmra"), f("pmdec"), f("phot_g_mean_mag")))
    if len(rows) >= GAIA_CAP:
        raise RuntimeError("Gaia query hit the %d-row cap; refusing a truncated star list" % GAIA_CAP)
    return np.array(rows, float).reshape(-1, 5)


def gaia_at_epoch(cat, year):
    """Gaia DR3 (epoch 2016.0) moved to the plate epoch. Columns: ra, dec, G, has_pm (1/0).
    Sources without proper motion stay at their 2016 position (they are still stars to reject; not references)."""
    has = np.isfinite(cat[:, 2]) & np.isfinite(cat[:, 3])
    dty = year - 2016.0
    pmra = np.where(has, cat[:, 2], 0.0); pmde = np.where(has, cat[:, 3], 0.0)
    dec = cat[:, 1] + pmde * dty / 3.6e6
    ra = cat[:, 0] + pmra * dty / 3.6e6 / np.cos(np.radians(cat[:, 1]))
    g = np.where(np.isfinite(cat[:, 4]), cat[:, 4], 21.0)
    return np.column_stack([ra, dec, g, has.astype(float)])


SKYBOT_URL = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
SKYBOT_RETRY_WAIT = 10.0  # seconds; SkyBoT now and then answers a good request with an error, and the same works later


def known_objects(site, jd_ut, ra, dec, radius_arcsec):
    """SkyBoT (LTE, Paris Observatory; formerly IMCCE): the asteroids, comets and planets (dates 1889-2060) in a cone
    around ra, dec at jd_ut, seen from the given observatory code. A list of dicts (num, name, label, xi and eta: the
    offsets in arcsec east and north of ra, dec; V, err: position error in arcsec; cls; dra, ddec: motion in arcsec/h
    east and north), or raises"""
    rs = int(min(math.ceil(radius_arcsec), 36000))
    q = urllib.parse.urlencode({"-ep": "%.6f" % jd_ut, "-ra": "%.6f" % (ra % 360.0), "-dec": "%.6f" % dec,
                                "-rs": "%d" % rs, "-mime": "votable", "-output": "all", "-observer": site,
                                "-filter": "0", "-objFilter": "111", "-from": "dss-precovery"})
    for attempt in range(3):
        try:
            rows = parse_skybot(http_get(SKYBOT_URL + "?" + q), expect=(jd_ut, ra % 360.0, dec, rs))
            break
        except RuntimeError as e:
            if attempt == 2:
                raise RuntimeError("%s (3 tries)" % e)
            time.sleep(SKYBOT_RETRY_WAIT * (attempt + 1))
    found = []
    for k in rows:
        xi, eta = tangent(np.array([k["ra"]]), np.array([k["dec"]]), ra, dec)
        found.append(dict(k, xi=float(xi[0]), eta=float(eta[0])))
    return found


def _num(v):
    try:
        x = float(str(v).strip())
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


SKYBOT_NONE = "No solar system object was found"


def parse_skybot(raw, expect=None):
    """SkyBoT's cone-search answer as a VOTable (-mime=votable) -> list of dicts (num, name, label "num name", ra, dec,
    V, err, cls, dra, ddec). SkyBoT's 'No solar system object was found' with an empty table is an empty list; any
    other answer without rows, or one that echoes another epoch or cone than expect=(jd, ra, dec, radius), is an
    error, never 'no objects'."""
    import xml.etree.ElementTree as ET
    data = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise RuntimeError("SkyBoT: not a VOTable: %r" % data.strip()[:200])

    def tag(e):
        return e.tag.rsplit("}", 1)[-1]

    def kids(e, name):
        return [k for k in e if tag(k) == name]

    def status(e):
        for k in kids(e, "INFO"):
            if k.get("name") == "QUERY_STATUS":
                return k.get("value", ""), (k.text or "").strip()
        return None, ""
    if tag(root) != "VOTABLE":
        raise RuntimeError("SkyBoT: not a VOTable: %r" % data.strip()[:200])
    st, msg = status(root)
    if st != "OK":
        raise RuntimeError("SkyBoT: query status %s %s" % (st, msg[:200]))
    if expect is not None:  # the answer must be for the epoch and cone asked
        info = {k.get("ID"): k.get("value", "") for k in kids(root, "INFO")}
        m = re.match(r"^\s*([0-9.]+)\s*([+-][0-9.]+)\s*,\s*rs=([0-9.]+)\s*$", info.get("reqTarget", ""))
        ep = _num(info.get("reqEpoch"))
        if not (m and abs(ep - expect[0]) < 1e-5 and abs(float(m.group(1)) - expect[1]) < 1e-5
                and abs(float(m.group(2)) - expect[2]) < 1e-5 and abs(float(m.group(3)) - expect[3]) < 0.5):
            raise RuntimeError("SkyBoT: the answer is for epoch %r, cone %r, not the one asked (%s)"
                               % (info.get("reqEpoch"), info.get("reqTarget"), expect))
    res = kids(root, "RESOURCE")
    tabs = kids(res[0], "TABLE") if len(res) == 1 else []
    if len(tabs) != 1:
        raise RuntimeError("SkyBoT: expected one table, got %d resources and %d tables" % (len(res), len(tabs)))
    t = tabs[0]
    datas = kids(t, "DATA")
    if len(datas) != 1 or len(kids(datas[0], "TABLEDATA")) != 1 or len(datas[0]) != 1:
        raise RuntimeError("SkyBoT: the table has no TABLEDATA part")
    trs = kids(kids(datas[0], "TABLEDATA")[0], "TR")
    if t.get("nrows") is None or int(t.get("nrows")) != len(trs):
        raise RuntimeError("SkyBoT: nrows=%s but %d rows" % (t.get("nrows"), len(trs)))
    rst, rmsg = status(res[0])
    if rst == "ERROR" and rmsg.startswith(SKYBOT_NONE) and not trs:
        return []
    if rst not in (None, "OK"):
        raise RuntimeError("SkyBoT: %s %s" % (rst, rmsg[:200]))
    if not trs:
        raise RuntimeError("SkyBoT: no rows, and not SkyBoT's 'nothing found' answer")
    fields = [f.get("ID") or f.get("name") or "" for f in kids(t, "FIELD")]
    col = {f.lower(): i for i, f in enumerate(fields)}
    miss = [k for k in ("num", "name", "ra", "de", "class", "magv", "errpos", "dracosdec", "ddec") if k not in col]
    if miss:
        raise RuntimeError("SkyBoT: no %s column (columns %s)" % (", ".join(miss), fields))
    out = []
    for tr in trs:
        v = [(td.text or "").strip() for td in kids(tr, "TD")]
        if len(v) != len(fields):
            raise RuntimeError("SkyBoT: a row with %d values for %d columns" % (len(v), len(fields)))
        get = lambda k: v[col[k]] if k in col else ""  # noqa: E731
        num = "" if get("num").lower() in ("", "-", "none", "null", "nan") else get("num")
        name = get("name")
        label = (num + " " + name).strip() if name != num else num
        kra, kdec = _parse_sexa(get("ra"), True), _parse_sexa(get("de"), False)
        if not label or not (np.isfinite(kra) and np.isfinite(kdec)):
            raise RuntimeError("SkyBoT: unreadable row (name %r, RA %r, Dec %r)" % (label, get("ra"), get("de")))
        out.append(dict(num=num, name=name, label=label, ra=kra, dec=kdec, V=_num(get("magv")), err=_num(get("errpos")),
                        cls=get("class"), dra=_num(get("dracosdec")), ddec=_num(get("ddec"))))
    return out


_DESIG = re.compile(r"^(?:(1[89]\d\d|20\d\d) ?([A-Z]{2}\d*)|(\d{4}) (P-L|T-[123]))$", re.I)


def _desig(s):
    """'2026 ED138', '2026ed138', '2040 P-L', '1125 t-2' -> the designation as printed ('2026 ED138'); else None"""
    m = _DESIG.match(re.sub(r"\s+", " ", str(s)).strip())
    if not m:
        return None
    return ("%s %s" % (m.group(1), m.group(2)) if m.group(1) else "%s %s" % (m.group(3), m.group(4))).upper()


def _norm_name(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return (_desig(s) or s).lower()


def name_keys(names):
    """(numbers, names) that identify an object, from strings such as '50000', 'Quaoar', '2002 LM60',
    '(50000) Quaoar' or Horizons' '50000 Quaoar (2002 LM60)'; a designation such as '2026 ED138' is a name, not
    number 2026"""
    nums, nams = set(), set()
    for s in names:
        s = re.sub(r"\s+", " ", str(s or "")).strip()
        for part in re.findall(r"\(([^)]*)\)", s):
            part = part.strip()
            (nums if part.isdigit() else nams).add(str(int(part)) if part.isdigit() else _norm_name(part))
        rest = re.sub(r"\([^)]*\)", " ", s).strip()
        rest = re.sub(r"\s+", " ", rest)
        if not rest:
            continue
        m = re.match(r"^(\d+)(?: (.+))?$", rest)
        if _desig(rest) or not m:
            nams.add(_norm_name(rest))
        else:
            nums.add(str(int(m.group(1))))
            if m.group(2):
                nams.add(_norm_name(m.group(2)))
    nams.discard("")
    return nums, nams


def _is_target(k, keys):
    """by number, or by name or designation. SkyBoT also lists some objects as comets, with the comet's name in the
    number column: (60558) Echeclus comes back as '60558 | Echeclus' and as 'P/Echeclus | 174P'; both are the target."""
    nums, nams = keys
    num, nm = str(k.get("num", "") or ""), str(k.get("name", "") or "")
    if num.isdigit() and num.lstrip("0") in nums:
        return True
    return any(_norm_name(re.sub(r"^\d*[PCDXI]/", "", x)) in nams for x in (nm, num) if x)


def _is_comet(k):
    return "comet" in str(k.get("cls", "")).lower() or bool(re.match(r"^(\d+[PCDXI]\b|[PCDXI]/)", str(k.get("name", ""))))


def _trail_distance(x, y, k, exp_min):
    """distance (arcsec) from x, y to the object's path during the exposure (to its mid-exposure place if no motion)"""
    h = exp_min / 120.0  # half the exposure in hours
    vx, vy = k.get("dra", float("nan")), k.get("ddec", float("nan"))
    if not (np.isfinite(vx) and np.isfinite(vy)) or h <= 0:
        return math.hypot(x - k["xi"], y - k["eta"])
    ax, ay = k["xi"] - vx * h, k["eta"] - vy * h
    bx, by = k["xi"] + vx * h, k["eta"] + vy * h
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 == 0 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / L2))
    return math.hypot(x - ax - u * dx, y - ay - u * dy)


KNOWN_MIN_RADIUS = 10.0    # arcsec: a source this close to a well-placed known object's path is that object
KNOWN_SHARP_ERR = 5.0      # arcsec: objects placed better than this (3 sigma 15") reject sources outright
KNOWN_MAX_ERR = 200.0      # arcsec: objects placed worse than this could be anywhere and are only listed
KNOWN_FAINT_MARGIN = 2.0   # mag: an asteroid predicted fainter than the source by more than this is not the source
KNOWN_CONE_MARGIN = 3.0 * KNOWN_MAX_ERR  # arcsec added to the box's half-diagonal for the SkyBoT cone


def known_object_verdicts(cands, known, keys, exp_min, pred_V=float("nan"), pred_cov=None):
    """the known objects from known_objects() against the sources in the box.
    - The target is the entry whose number, or name or designation, is in keys (from name_keys). If none is, an
      object whose place and brightness fit the prediction (within 3 sigma of it, its own error, at least 10"/3, added
      to the prediction's covariance pred_cov; V within 2 mag of pred_V) may be the target under another name: it
      rejects nothing ('possible').
    - Each source is compared with each other object's path during the exposure. An asteroid placed to 5" or better
      (or with no error given) rejects the sources within max(10", 3 x error) of its path, except that one predicted
      more than 2 mag fainter than the source only adds a note. An asteroid placed to between 5" and 200" rejects
      nothing; the sources within 3 x error that are not much brighter than it get a note and 'known_uncertain'.
      Asteroids placed worse than 200" are only listed. A comet (its place and magnitude this far back are not
      trusted) rejects nothing and marks the sources within max(60", 3 x error).
    Returns (others, target, possible) as dicts for the results."""
    others, target, possible = [], [], []
    for k in known:
        (target if _is_target(k, keys) else others).append(k)
    if not target:
        for k in list(others):
            if np.isfinite(k["err"]) and k["err"] > KNOWN_MAX_ERR:
                continue
            s_ = max(KNOWN_MIN_RADIUS / 3.0, k["err"] if np.isfinite(k["err"]) else 0.0)
            C = (np.zeros((2, 2)) if pred_cov is None else np.asarray(pred_cov, float)) + np.diag([s_ * s_, s_ * s_])
            x = np.array([k["xi"], k["eta"]])
            vfit = not (np.isfinite(k["V"]) and np.isfinite(pred_V)) or abs(k["V"] - pred_V) <= KNOWN_FAINT_MARGIN
            if float(x @ np.linalg.solve(C, x)) <= 9.0 and vfit:
                others.remove(k); possible.append(k)
    for c in cands:
        m = c.get("mag_g_scale", float("nan"))
        for k in others:
            d = _trail_distance(c["dra_arcsec"], c["ddec_arcsec"], k, exp_min)
            comet = _is_comet(k)
            vtxt = ("V %.1f" % k["V"]) if np.isfinite(k["V"]) else "V unknown"
            faint = (not comet) and np.isfinite(k["V"]) and np.isfinite(m) and k["V"] > m + KNOWN_FAINT_MARGIN
            e = k["err"]
            if not comet and (not np.isfinite(e) or e <= KNOWN_SHARP_ERR):
                if d > max(KNOWN_MIN_RADIUS, 3.0 * e if np.isfinite(e) else 0.0):
                    continue
                if faint:
                    c["note"] = (c["note"] + "; " if c["note"] else "") + \
                        "known object %s %.1f\" away is predicted at %s, too faint to be this source" % (k["label"], d, vtxt)
                else:
                    c["rejected"] = (c["rejected"] + "; " if c["rejected"] else "") + "known object " + k["label"]
            elif not (np.isfinite(e) and e > KNOWN_MAX_ERR):  # a comet's place this far back is never trusted
                r = max(3.0 * e, 60.0 if comet else 0.0) if np.isfinite(e) else 60.0
                if d <= r and not faint:
                    c.setdefault("known_uncertain", []).append(k["label"])
                    c["note"] = (c["note"] + "; " if c["note"] else "") + "could be known object %s (%s; its place is " \
                        "uncertain by %s)" % (k["label"], vtxt, ("%.0f\"" % e) if np.isfinite(e) else "an unknown amount")
    rec = lambda k: dict(name=k["label"], dra_arcsec=round(k["xi"], 2), ddec_arcsec=round(k["eta"], 2), V=k["V"],  # noqa: E731
                         err_arcsec=k["err"], cls=k.get("cls", ""), rate_arcsec_h=[k.get("dra"), k.get("ddec")])
    return [rec(k) for k in others], [rec(k) for k in target], [rec(k) for k in possible]


# --------------------------------------------------------------------------------------- image measurement

def background(img, box=32):
    """robust local background and noise"""
    small = ndimage.median_filter(img, size=(box // 4, box // 4))
    bg = ndimage.median_filter(small, size=(box, box))
    resid = img - bg
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    return bg, max(mad, 1e-6)


def sky_noise(a, nsig_mask=2.5, grow=3):
    """robust noise of the source-free pixels of a (sources masked and grown), not of the whole image"""
    s = 1.4826 * np.median(np.abs(a - np.median(a)))
    for _ in range(3):
        m = ndimage.binary_dilation(a > nsig_mask * s, iterations=grow)
        v = a[~m]
        if v.size < 100:
            break
        s_new = 1.4826 * np.median(np.abs(v - np.median(v)))
        if abs(s_new - s) < 0.01 * s:
            s = s_new
            break
        s = s_new
    return max(float(s), 1e-6)


def detect(img, nsig=4.0):
    """local peaks of the smoothed image (so close neighbors stay separate).
    Returns det, noise, sigma_px. det columns: x, y (1-based FITS pixels), flux (r=3 px), snr (smoothed peak over
    the source-free noise), concentration, fwhm (px), peak, formal x error, formal y error (px), fitted amplitude"""
    img = np.asarray(img, float)
    bg, _ = background(img)
    r = img - bg
    sm = ndimage.gaussian_filter(r, 1.0)
    ssig = sky_noise(sm)
    pk = (sm == ndimage.maximum_filter(sm, size=5)) & (sm > nsig * ssig)
    pk[:3, :] = False; pk[-3:, :] = False; pk[:, :3] = False; pk[:, -3:] = False
    ys, xs = np.nonzero(pk)
    ny, nx = r.shape
    out = []
    if len(ys) == 0:
        return np.zeros((0, 10)), ssig, 1.0
    for y0, x0 in zip(ys, xs):
        cx, cy, fwhm = refine_centroid(r, float(x0), float(y0))
        if not (2 <= cx < nx - 2 and 2 <= cy < ny - 2) or math.hypot(cx - x0, cy - y0) > 2.0:
            cx, cy = float(x0), float(y0)
        yy, xx = np.mgrid[max(0, y0 - 3):min(ny, y0 + 4), max(0, x0 - 3):min(nx, x0 + 4)]
        ap = np.hypot(xx - cx, yy - cy) <= 3.0
        flux = float(r[yy[ap], xx[ap]].sum())
        box = r[max(0, y0 - 2):y0 + 3, max(0, x0 - 2):x0 + 3]
        conc = float(r[y0, x0] / box.sum()) if box.sum() > 0 else float("nan")
        out.append([cx, cy, flux, sm[y0, x0] / ssig if ssig > 0 else 0.0, conc, fwhm, r[y0, x0], float("nan"), float("nan"),
                    float("nan")])
    det = np.array(out, float).reshape(-1, 10)
    # second pass: Gaussian fit with the stars' typical width (better positions)
    mid = (det[:, 3] > 10) & (det[:, 3] < 80) & np.isfinite(det[:, 5])
    fw = float(np.median(det[mid, 5])) if mid.sum() >= 5 else float(np.nanmedian(det[:, 5]))
    sigma_px = max(fw / 2.3548, 0.6) if np.isfinite(fw) else 1.0
    for i in range(len(det)):
        p = psf_fit(r, det[i, 0], det[i, 1], sigma_px)
        if p is not None:
            det[i, 0], det[i, 1], det[i, 7], det[i, 8], det[i, 9] = p[0], p[1], p[2], p[3], p[4]
    det[:, 0] += 1.0; det[:, 1] += 1.0  # 1-based FITS pixels
    return det, ssig, sigma_px


def residual_image(img):
    img = np.asarray(img, float)
    bg, _ = background(img)
    return img - bg


def forced_amp(r, x0, y0, sigma_px, half=4):
    """amplitude of a Gaussian of the stars' width fixed at (x0, y0) (0-based), with a flat local background:
    a linear least-squares fit. Returns nan off the image."""
    ny, nx = r.shape
    xi = int(round(x0)); yi = int(round(y0))
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > nx or yi + half + 1 > ny:
        return float("nan")
    yy, xx = np.mgrid[yi - half:yi + half + 1, xi - half:xi + half + 1]
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma_px ** 2)).ravel()
    A = np.column_stack([g, np.ones_like(g)])
    c, *_ = np.linalg.lstsq(A, r[yi - half:yi + half + 1, xi - half:xi + half + 1].ravel(), rcond=None)
    return float(c[0])


def clean_spots(det, shape, n, rng, min_sep_px=6.0, edge=8):
    """random positions (0-based) at least min_sep_px from every detection and inside the margins"""
    ny, nx = shape
    from scipy.spatial import cKDTree
    tree = cKDTree(det[:, :2] - 1.0) if len(det) else None
    out = []
    for _ in range(n * 40):
        x = rng.uniform(edge, nx - edge); y = rng.uniform(edge, ny - edge)
        if tree is None or tree.query([x, y])[0] >= min_sep_px:
            out.append((x, y))
            if len(out) >= n:
                break
    return out


def forced_noise(r, det, sigma_px, rng, n=150):
    """the scatter of forced amplitudes at empty places: the real noise, correlations included"""
    a = [forced_amp(r, x, y, sigma_px) for x, y in clean_spots(det, r.shape, n, rng)]
    a = np.array([v for v in a if np.isfinite(v)])
    if a.size < 20:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def injection_errors(r, det, sigma_px, amp, rng, n=200):
    """per-axis centroid scatter (px) of artificial stars of amplitude amp added at empty places of this plate:
    the 68.3% point of the absolute offsets from their median (1 sigma for a Gaussian; a scaled median absolute
    deviation would understate faint images' wider tails)"""
    ny, nx = r.shape
    ex, ey = [], []
    yy, xx = np.mgrid[0:ny, 0:nx]
    for x, y in clean_spots(det, r.shape, n, rng, min_sep_px=8.0):
        xa, xb = int(x) - 6, int(x) + 7; ya, yb = int(y) - 6, int(y) + 7
        sub = r.copy()
        sub[ya:yb, xa:xb] += amp * np.exp(-((xx[ya:yb, xa:xb] - x) ** 2 + (yy[ya:yb, xa:xb] - y) ** 2) / (2 * sigma_px ** 2))
        p = psf_fit(sub, x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5), sigma_px)
        if p is not None:
            ex.append(p[0] - x); ey.append(p[1] - y)
    if len(ex) < max(15, n // 3):
        return float("nan"), float("nan"), len(ex)
    ex = np.array(ex); ey = np.array(ey)
    rs = lambda v: float(np.percentile(np.abs(v - np.median(v)), 68.27))  # noqa: E731
    return rs(ex), rs(ey), len(ex)


def psf_fit(r, cx, cy, sigma_px, half=4):
    """least-squares fit of a round Gaussian (width fixed to the stars') + flat background in a small window.
    cx, cy: 0-based start. Returns 0-based x, y, their formal 1-sigma errors (px), amplitude; or None."""
    from scipy.optimize import least_squares
    ny, nx = r.shape
    x0 = int(round(cx)); y0 = int(round(cy))
    xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
    ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = r[ya:yb, xa:xb]
    yy, xx = np.mgrid[ya:yb, xa:xb]
    s2 = 2.0 * sigma_px ** 2

    def resid(p):
        return (p[2] * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / s2) + p[3] - sub).ravel()
    p0 = [cx, cy, max(float(sub.max() - np.median(sub)), 1.0), float(np.median(sub))]
    try:
        f = least_squares(resid, p0, method="lm", max_nfev=200)
    except Exception:  # noqa
        return None
    if not f.success or abs(f.x[0] - cx) > 2.5 or abs(f.x[1] - cy) > 2.5 or f.x[2] <= 0:
        return None
    dof = max(sub.size - 4, 1)
    s2r = float((f.fun ** 2).sum() / dof)
    try:
        cov = np.linalg.inv(f.jac.T @ f.jac) * s2r
        ex, ey = math.sqrt(max(cov[0, 0], 0)), math.sqrt(max(cov[1, 1], 0))
    except np.linalg.LinAlgError:
        ex = ey = float("nan")
    return float(f.x[0]), float(f.x[1]), ex, ey, float(f.x[2])


def psf_fit_joint(r, cx, cy, nbrs, sigma_px, pad=4):
    """the source at (cx, cy) fitted together with its close neighbors [(x, y, amp), ...] (0-based): round Gaussians,
    the source with the stars' width, each neighbor with a free width (it may be a galaxy), and a flat background, in
    one window around them all (a one-source fit is pulled toward a neighbor closer than about 2 FWHM).
    Returns 0-based x, y and amplitude of the source, or None."""
    from scipy.optimize import least_squares
    ny, nx = r.shape
    xs = [cx] + [q[0] for q in nbrs]; ys = [cy] + [q[1] for q in nbrs]
    xa, xb = max(0, int(math.floor(min(xs))) - pad), min(nx, int(math.ceil(max(xs))) + pad + 1)
    ya, yb = max(0, int(math.floor(min(ys))) - pad), min(ny, int(math.ceil(max(ys))) + pad + 1)
    sub = r[ya:yb, xa:xb]
    yy, xx = np.mgrid[ya:yb, xa:xb]
    s2 = 2.0 * sigma_px ** 2
    k = len(nbrs)

    def model(p):
        m = p[2] * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / s2) + p[3]
        for j in range(k):
            x_, y_, a_, w_ = p[4 + 4 * j:8 + 4 * j]
            m = m + a_ * np.exp(-((xx - x_) ** 2 + (yy - y_) ** 2) / (2.0 * w_ ** 2))
        return m
    a0 = max(float(r[int(round(cy)), int(round(cx))] - np.median(sub)), 1.0)
    p0 = [cx, cy, a0, float(np.median(sub))]
    lo = [cx - 2.5, cy - 2.5, 0.0, -np.inf]; hi = [cx + 2.5, cy + 2.5, np.inf, np.inf]
    for x_, y_, a_ in nbrs:
        p0 += [x_, y_, max(float(a_), 1.0), sigma_px]
        lo += [x_ - 2.5, y_ - 2.5, 0.0, 0.5 * sigma_px]; hi += [x_ + 2.5, y_ + 2.5, np.inf, 4.0 * sigma_px]
    p0 = [min(max(v, l_ + 1e-9), h_ - 1e-9) for v, l_, h_ in zip(p0, lo, hi)]
    try:
        f = least_squares(lambda p: (model(p) - sub).ravel(), p0, bounds=(lo, hi), max_nfev=400)
    except Exception:  # noqa
        return None
    if not f.success or f.x[2] <= 0:
        return None
    return float(f.x[0]), float(f.x[1]), float(f.x[2])


def refine_centroid(r, cx, cy, half=3, it=6):
    """iterative centroid in a small window (background-subtracted, positive pixels only)"""
    ny, nx = r.shape
    fwhm = float("nan")
    for _ in range(it):
        x0 = int(round(cx)); y0 = int(round(cy))
        xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
        ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
        sub = np.clip(r[ya:yb, xa:xb], 0, None)
        if sub.sum() <= 0:
            break
        yy, xx = np.mgrid[ya:yb, xa:xb]
        wsum = sub.sum()
        ncx = (sub * xx).sum() / wsum; ncy = (sub * yy).sum() / wsum
        var = ((sub * ((xx - ncx) ** 2 + (yy - ncy) ** 2)).sum() / wsum) / 2.0
        fwhm = 2.3548 * math.sqrt(max(var, 1e-6))
        done = abs(ncx - cx) < 0.005 and abs(ncy - cy) < 0.005
        cx, cy = ncx, ncy
        if done:
            break
    return cx, cy, fwhm


def fit_astrometry(det, h, gaia_ep, ra0, dec0, match_arcsec=3.0):
    """match detections to Gaia stars WITH proper motion at the plate epoch and fit standard coordinates as
    polynomials of pixel x, y. Every Gaia source (with or without proper motion) counts for the blend test."""
    if len(det) == 0 or len(gaia_ep) == 0:
        return None
    from scipy.spatial import cKDTree
    ra, dec = dss_pix2sky(h, det[:, 0], det[:, 1])
    dxi, deta = tangent(ra, dec, ra0, dec0)
    axi, aeta = tangent(gaia_ep[:, 0], gaia_ep[:, 1], ra0, dec0)
    alltree = cKDTree(np.column_stack([axi, aeta]))
    pm = gaia_ep[:, 3] > 0.5 if gaia_ep.shape[1] > 3 else np.ones(len(gaia_ep), bool)
    gaia_ep = gaia_ep[pm]
    if len(gaia_ep) < 6:
        return None
    gxi, geta = axi[pm], aeta[pm]
    # global shift from an offset histogram (the header solution may be off by several arcsec)
    gtree = cKDTree(np.column_stack([gxi, geta]))
    offs = []
    for i, nb in enumerate(gtree.query_ball_point(np.column_stack([dxi, deta]), 15.0)):
        for j in nb:
            offs.append((gxi[j] - dxi[i], geta[j] - deta[i]))
    if not offs:
        return None
    offs = np.array(offs)
    H, xe, ye = np.histogram2d(offs[:, 0], offs[:, 1], bins=30, range=[[-15, 15], [-15, 15]])
    H = ndimage.uniform_filter(H, 3)
    ix, iy = np.unravel_index(np.argmax(H), H.shape)
    sx, sy = 0.5 * (xe[ix] + xe[ix + 1]), 0.5 * (ye[iy] + ye[iy + 1])
    near = np.hypot(offs[:, 0] - sx, offs[:, 1] - sy) < 2.0
    if near.sum() >= 3:
        sx, sy = float(np.median(offs[near, 0])), float(np.median(offs[near, 1]))
    pairs = []
    dd, jj = gtree.query(np.column_stack([dxi + sx, deta + sy]), k=1)
    da, _ = alltree.query(np.column_stack([dxi + sx, deta + sy]), k=min(2, len(axi)))
    da = np.atleast_2d(np.asarray(da).reshape(len(dxi), -1))
    for i in range(len(dxi)):
        second = da[i, 1] if da.shape[1] > 1 else 99.0
        if dd[i] < match_arcsec and second > 2.0 * match_arcsec:  # skip blends (with any Gaia source)
            pairs.append((i, int(jj[i])))
    if len(pairs) < 6:
        return None
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
    x = det[pi, 0]; y = det[pi, 1]
    xc = (h["NAXIS1"] + 1) / 2.0; yc = (h["NAXIS2"] + 1) / 2.0
    order = 2 if len(pairs) >= 25 else 1
    keep = np.ones(len(pairs), bool)
    M = design(x - xc, y - yc, order)

    def solve(k_):
        a, *_ = np.linalg.lstsq(M[k_], gxi[pj][k_], rcond=None)
        b, *_ = np.linalg.lstsq(M[k_], geta[pj][k_], rcond=None)
        return a, b
    for _ in range(8):  # per-coordinate robust 3-sigma clip
        cx, cy = solve(keep)
        rx = gxi[pj] - M @ cx; ry = geta[pj] - M @ cy
        sxr = 1.4826 * np.median(np.abs(rx[keep])); syr = 1.4826 * np.median(np.abs(ry[keep]))
        new = (np.abs(rx) < max(3.0 * sxr, 0.2)) & (np.abs(ry) < max(3.0 * syr, 0.2))
        if new.sum() < 6 or np.array_equal(new, keep):
            break
        keep = new
    cx, cy = solve(keep)  # final fit on the final set
    rx = gxi[pj] - M @ cx; ry = geta[pj] - M @ cy
    rms = math.sqrt(np.mean(rx[keep] ** 2 + ry[keep] ** 2) / 2.0)
    return dict(order=order, cx=cx, cy=cy, xc=xc, yc=yc, ra0=ra0, dec0=dec0, n=int(keep.sum()), rms=rms,
                gmag=gaia_ep[pj[keep], 2], flux=det[pi[keep], 2], matched_det=set(pi.tolist()),
                res_x=rx[keep], res_y=ry[keep], npar=M.shape[1])


def invert_astrometry(fit, h, ra, dec):
    """pixel (1-based) of ra, dec under a fitted solution: Newton steps from the header solution's answer"""
    x, y = dss_sky2pix(h, ra, dec)
    x = float(x[0]); y = float(y[0])
    txi, teta = tangent(np.array([ra]), np.array([dec]), fit["ra0"], fit["dec0"])
    for _ in range(10):
        M = design(np.array([x, x + 1, x]) - fit["xc"], np.array([y, y, y + 1]) - fit["yc"], fit["order"])
        xi = M @ fit["cx"]; et = M @ fit["cy"]
        J = np.array([[xi[1] - xi[0], xi[2] - xi[0]], [et[1] - et[0], et[2] - et[0]]])
        d = np.linalg.solve(J, np.array([txi[0] - xi[0], teta[0] - et[0]]))
        x += d[0]; y += d[1]
        if abs(d[0]) < 1e-4 and abs(d[1]) < 1e-4:
            break
    return x, y


def design(x, y, order):
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.column_stack(cols)


def apply_astrometry(fit, x, y):
    M = design(np.asarray(x, float) - fit["xc"], np.asarray(y, float) - fit["yc"], fit["order"])
    return untangent(M @ fit["cx"], M @ fit["cy"], fit["ra0"], fit["dec0"])


def phot_zero(fit):
    """instrumental -2.5log(flux) against Gaia G for the matched stars: a rough magnitude scale (photographic)"""
    f = fit["flux"]; g = fit["gmag"]
    ok = (f > 0) & np.isfinite(g) & (g > 14) & (g < 20.5)
    if ok.sum() < 5:
        return None
    ins = -2.5 * np.log10(f[ok])
    A = np.column_stack([ins, np.ones_like(ins)])
    c, *_ = np.linalg.lstsq(A, g[ok], rcond=None)
    return c

# --------------------------------------------------------------------------------------------- the steps

def subject_paths(root, subject):
    if not SAFE.match(subject):
        raise ValueError("subject name must be letters, digits, _ . + - (got %r)" % subject)
    out = os.path.join(root, "output", "precovery", subject)
    os.makedirs(os.path.join(out, "cutouts"), exist_ok=True)
    os.makedirs(os.path.join(out, "pictures"), exist_ok=True)
    disk_ok(out)
    return out


class PlanIncomplete(RuntimeError):
    """some archive queries failed, so the list of plates would be incomplete"""


class FetchIncomplete(PlanIncomplete):
    """some downloads failed, so a plate could be measured without all of its other-date plates"""


class SubjectMismatch(ValueError):
    """the subject's names are not names of the object that JPL Horizons found"""


def read_plan(out):
    """plan.json, refused when the last planning run of this subject stopped incomplete"""
    st = os.path.join(out, "plan_status.txt")
    if os.path.exists(st) and open(st).read().startswith("INCOMPLETE"):
        raise PlanIncomplete("the last --plan of this subject was incomplete (%s); plan it again first" % st)
    return json.load(open(os.path.join(out, "plan.json")))


def search_half_box(cfg, pl):
    """half-widths (arcsec) of the search box in RA and Dec: box_factor x 3-sigma + 3", at least min_box_arcsec"""
    k = cfg.get("box_factor", 1.5)
    return (max(k * pl["ra3"] + 3.0, cfg.get("min_box_arcsec", 0.0)),
            max(k * pl["dec3"] + 3.0, cfg.get("min_box_arcsec", 0.0)))


def box_on_cutout(h, ra, dec, half_arcsec):
    """True if a square of half-width half_arcsec around ra, dec lies inside the cutout with header h"""
    try:
        x, y = dss_sky2pix(h, ra, dec)
        m = half_arcsec / (h["XPIXELSZ"] * h.get("PLTSCALE", 67.2) / 1000.0)  # in pixels
        return bool(1 + m <= float(x[0]) <= h["NAXIS1"] - m and 1 + m <= float(y[0]) <= h["NAXIS2"] - m)
    except Exception:  # noqa
        return False


def cutout_covers(path, ra, dec, half_arcsec):
    """True if the cutout file exists and holds the search box (a cutout kept from an earlier plan may not)"""
    if not os.path.exists(path):
        return False
    from astropy.io import fits
    try:
        h = fits.getheader(path)
    except Exception:  # noqa
        return False
    return box_on_cutout(h, ra, dec, half_arcsec)


def _incomplete(status, msg):
    log("PLAN INCOMPLETE: " + msg)
    with open(status, "w") as f:
        f.write("INCOMPLETE %sZ  %s\n" % (dt.datetime.now(dt.timezone.utc).isoformat()[:19], msg))
    raise PlanIncomplete(msg)


def check_target_names(cfg, target):
    """the subject's names must all be names of the object JPL Horizons found (target: its 'Target body name'), so that
    the known-object check sets aside that object and no other"""
    if not target:
        return
    hn, hm = name_keys([target])
    bad = [n for n in cfg.get("names", [cfg["designation"]]) if not (name_keys([n])[0] <= hn and name_keys([n])[1] <= hm)]
    if bad:
        raise SubjectMismatch("JPL Horizons found %s for %r, but the subject's names %s are not names of that object; "
                              "check the Horizons command and the names" % (target, cfg["horizons_command"], bad))


def step_plan(cfg, out):
    cmd = cfg["horizons_command"]
    y0, y1 = cfg["years"]
    status = os.path.join(out, "plan_status.txt")
    log("Track %s %d-%d ..." % (cfg["designation"], y0, y1))
    try:
        track = horizons_track(cmd, "%d-01-01" % y0, "%d-01-01" % (y1 + 1), "5 d")
    except Exception as ex:  # noqa
        _incomplete(status, "JPL Horizons failed (%s). Nothing was planned; run this subject again later." % str(ex)[:300])
    if track:
        log("  JPL Horizons: %s" % (track[0].get("target") or "(no target name given)"))
        try:
            check_target_names(cfg, track[0].get("target", ""))
        except SubjectMismatch as ex:
            with open(status, "w") as f:  # an earlier plan of this subject is not to be used either
                f.write("INCOMPLETE %sZ  %s\n" % (dt.datetime.now(dt.timezone.utc).isoformat()[:19], ex))
            raise
    track = [r for r in track if r["elong"] > cfg.get("min_elongation", 60)]  # night-time plates, twilight included
    step_deg = cfg.get("sample_deg", 1.0)  # DSS plate cells are ~5-6 deg, so 1 deg sampling finds them all
    windows = cfg.get("survey_years", {})
    plates = {}
    asked = failed = 0
    refused = {}  # plate -> why its header could not be read for measuring
    for sv in cfg.get("surveys", SURVEYS):
        ya, yb = windows.get(sv, [y0, y1])
        pts = []
        for r in track:
            if not (ya <= r["t"].year <= yb):
                continue
            if not pts or math.hypot((r["ra"] - pts[-1]["ra"]) * math.cos(math.radians(r["dec"])), r["dec"] - pts[-1]["dec"]) > step_deg:
                pts.append(r)
        log("  %s: %d sample points (%d-%d)" % (sv, len(pts), ya, yb))
        for k, p in enumerate(pts):
            asked += 1
            try:
                hdu = fetch_dss(sv, p["ra"], p["dec"], 1.0)
            except Exception as e:  # noqa
                failed += 1
                log("  %s point %d: %s" % (sv, k, e)); continue
            if hdu is None:
                continue
            try:
                info = plate_info(hdu.header)
            except Exception as e:  # noqa
                try:
                    pid = comparison_info(hdu.header)["plate"]
                    why = "used only as a comparison plate (%s)" % e
                except Exception:  # noqa
                    pid = "%s point %d" % (sv, k)
                    why = "header not readable (%s)" % e
                if pid not in refused:
                    refused[pid] = why
                    log("  %s: plate %s %s" % (sv, pid, why))
                continue
            key = info["plate"]  # one entry per physical plate, even if two survey names serve it
            if key not in plates:
                plates[key] = dict(survey=sv, also_in=[], **{k2: (v.isoformat() if isinstance(v, dt.datetime) else v)
                                                               for k2, v in info.items()})
                log("  plate %-16s (%s) start %s exp %4.0f min  %s  [%s]%s" % (
                    key, sv, info["start"].isoformat(), info["exp_min"], info["telescope"], info["time_note"],
                    ("  FLAG: " + info["flag"]) if info["flag"] else ""))
            elif sv != plates[key]["survey"] and sv not in plates[key]["also_in"]:
                plates[key]["also_in"].append(sv)
            time.sleep(0.3)
    if failed:
        # a plate missed here would silently read as "not on any plate": stop instead, and write nothing
        msg = ("%d of %d archive queries failed (errors on the archive's side, e.g. HTTP 500), so the list of plates "
               "would be incomplete. Nothing was planned or measured; run this subject again later." % (failed, asked))
        log("PLAN INCOMPLETE: " + msg)
        with open(status, "w") as f:
            f.write("INCOMPLETE %sZ  %s\n" % (dt.datetime.now(dt.timezone.utc).isoformat()[:19], msg))
        raise PlanIncomplete(msg)
    # keep plates whose date puts the object on the same plate
    keep = []
    cache = {}
    for key, pl in sorted(plates.items()):
        mid = dt.datetime.fromisoformat(pl["mid"])
        try:
            e = horizons_at(cmd, pl["site"], jd_from_datetime(mid), cache)
            hdu = fetch_dss(pl["survey"], e["ra"], e["dec"], 1.0)
        except Exception as ex:  # noqa
            _incomplete(status, "the archive query for plate %s at the object's position failed (%s), so whether the "
                        "object is on it is unknown. Nothing was planned; run this subject again later." % (key, str(ex)[:300]))
        same = False
        if hdu is not None:
            try:
                same = plate_info(hdu.header)["plate"] == pl["plate"]
            except Exception:  # noqa
                same = False
        pl.update(pred_ra=e["ra"], pred_dec=e["dec"], V=e["V"], ra3=e["ra3"], dec3=e["dec3"], smaa=e["smaa"],
                  smia=e["smia"], theta=e["theta"], rate=math.hypot(e["dra"], e["ddec"]), dra_rate=e["dra"],
                  ddec_rate=e["ddec"], on_plate=same, target_name=e.get("target", ""))
        log("  %-18s mid %s  pred %.5f %+.5f  V %.2f  3sig ellipse %5.1f\" x %4.1f\" at %+5.1f deg  %s%s" % (
            key, pl["mid"][:16], e["ra"], e["dec"], e["V"], e["smaa"], e["smia"], e["theta"],
            "ON PLATE" if same else "-", "" if pl["reportable"] else "  (not reportable: %s)" % (pl["flag"] or pl["time_note"])))
        if same:
            keep.append(pl)
        time.sleep(0.3)
    with open(os.path.join(out, "plates_all.json"), "w") as f:
        json.dump(plates, f, indent=1)
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(keep, f, indent=1)
    with open(os.path.join(out, "fetch_status.txt"), "w") as f:  # a new plan needs a new fetch before measuring
        f.write("INCOMPLETE %sZ  new plan, not fetched yet\n" % dt.datetime.now(dt.timezone.utc).isoformat()[:19])
    unread = sorted(p_ for p_, w_ in refused.items() if not w_.startswith("used only"))
    with open(status, "w") as f:
        f.write("COMPLETE %sZ  %d archive queries, none failed%s\n" % (
            dt.datetime.now(dt.timezone.utc).isoformat()[:19], asked,
            ("; %d plate headers not readable, not searched: %s" % (len(unread), ", ".join(unread))) if unread else ""))
    log("Plan: %d plates seen, %d with the object on them -> plan.json%s" % (
        len(plates), len(keep), ("  (%d plate headers not readable; see plan_status.txt)" % len(unread)) if unread else ""))
    return keep


def step_fetch(cfg, out):
    plan = read_plan(out)
    failed = []
    status = os.path.join(out, "fetch_status.txt")
    with open(status, "w") as f:  # until the end says otherwise: an interrupted fetch must not be measured
        f.write("INCOMPLETE %sZ  fetch started, not finished\n" % dt.datetime.now(dt.timezone.utc).isoformat()[:19])
    for pl in plan:
        tag = "%s_%s" % (pl["survey"], pl["plate"])
        half = max(search_half_box(cfg, pl))
        size = cfg.get("cutout_arcmin", 10.0)
        if math.isfinite(half):
            size = max(size, 2 * half / 60.0 + 4.0)  # the whole search box plus a 2' margin
        if size > 60.0:
            log("  %s: the search box needs a %.0f' cutout, more than the DSS limit of 60'; not fetched" % (tag, size))
            continue
        tgt = os.path.join(out, "cutouts", tag + "__target.fits")
        if not cutout_covers(tgt, pl["pred_ra"], pl["pred_dec"], half):
            try:
                hdu = fetch_dss(pl["survey"], pl["pred_ra"], pl["pred_dec"], size)
            except Exception as e:  # noqa
                failed.append("%s: %s" % (tag, str(e)[:200])); log("  %s: download failed: %s" % (tag, e)); continue
            if hdu is None:  # this plate is then not measured (the measuring step says so)
                log("  %s: no cutout returned" % tag); continue
            try:
                got = plate_info(hdu.header)["plate"]
            except ValueError as e:
                log("  %s: the cutout's header can't be read (%s); skipped" % (tag, e)); continue
            if got != pl["plate"]:
                log("  %s: the archive returned plate %s at this position, not %s; skipped" % (tag, got, pl["plate"]))
                continue
            hdu.writeto(tgt + ".part", overwrite=True, output_verify="silentfix"); os.replace(tgt + ".part", tgt)
        for sv in cfg.get("comparison_surveys", SURVEYS):  # other-date plates from every survey, by default
            cp = os.path.join(out, "cutouts", tag + "__cmp_" + sv + ".fits")
            if cutout_covers(cp, pl["pred_ra"], pl["pred_dec"], half):
                continue
            try:
                hdu = fetch_dss(sv, pl["pred_ra"], pl["pred_dec"], size)
            except Exception as e:  # noqa
                failed.append("%s comparison %s: %s" % (tag, sv, str(e)[:200]))
                log("  %s comparison %s: download failed: %s" % (tag, sv, e)); continue
            if hdu is None:
                continue
            try:
                info = comparison_info(hdu.header)
            except Exception as e:  # noqa
                log("  %s comparison %s: header %s" % (tag, sv, e)); continue
            if info["plate"] == pl["plate"]:
                continue  # the target plate itself (served under another survey name)
            hdu.writeto(cp + ".part", overwrite=True, output_verify="silentfix"); os.replace(cp + ".part", cp)
            time.sleep(0.3)
        log("  fetched %s (%.1f')" % (tag, size))
    now = dt.datetime.now(dt.timezone.utc).isoformat()[:19]
    if failed:
        # a missing other-date plate could let a fixed source pass as the object: no measuring until all are in
        msg = ("%d downloads failed (%s). The cutouts already downloaded are kept; run --fetch again later, then "
               "--measure." % (len(failed), "; ".join(failed)[:600]))
        with open(status, "w") as f:
            f.write("INCOMPLETE %sZ  %s\n" % (now, msg))
        log("FETCH INCOMPLETE: " + msg)
        raise FetchIncomplete(msg)
    with open(status, "w") as f:
        f.write("COMPLETE %sZ\n" % now)


def ellipse_cov(pl):
    """the prediction's uncertainty as a 1-sigma covariance in (east, north) arcsec, from Horizons' 3-sigma ellipse
    (SMAA, SMIA, Theta from +RA toward +Dec); falls back to the RA/Dec marginals without correlation"""
    smaa, smia, th = pl.get("smaa"), pl.get("smia"), pl.get("theta")
    ok = lambda v: v is not None and isinstance(v, (int, float)) and np.isfinite(v)  # noqa: E731
    if ok(smaa) and ok(smia) and ok(th):
        t = math.radians(th)
        u = np.array([math.cos(t), math.sin(t)]); w = np.array([-math.sin(t), math.cos(t)])
        return (smaa / 3.0) ** 2 * np.outer(u, u) + (smia / 3.0) ** 2 * np.outer(w, w), "Horizons 3-sigma ellipse"
    if ok(pl.get("ra3")) and ok(pl.get("dec3")):
        return np.diag([(pl["ra3"] / 3.0) ** 2, (pl["dec3"] / 3.0) ** 2]), "RA/Dec marginals (no ellipse)"
    return None, "no uncertainty from Horizons"


def _parse_sexa(s, hours):
    v = [float(x) for x in re.findall(r"[0-9]+(?:\.[0-9]*)?", str(s))]
    if len(v) < 3:
        return float("nan")
    x = v[0] + v[1] / 60.0 + v[2] / 3600.0
    x = x * 15.0 if hours else x
    return -x if str(s).strip().startswith("-") else x


def gaia_star_distance(gep, year, xi, eta, ra0, dec0):
    """distance (arcsec) to the nearest Gaia source, and the matching radius that applies to it (3" for stars with
    proper motion; 3" plus 10 mas/yr times the years from 2016 for those without)"""
    if len(gep) == 0:
        return 99.0, 3.0
    gx, gy = tangent(gep[:, 0], gep[:, 1], ra0, dec0)
    d = np.hypot(gx - xi, gy - eta)
    rad = np.where(gep[:, 3] > 0.5, 3.0, 3.0 + abs(year - 2016.0) * 0.010)
    j = int(np.argmin(d - rad))
    return float(d[j]), float(rad[j])


def amp_from_flux(flux_ap, sigma_px):
    return flux_ap / (2.0 * math.pi * sigma_px ** 2 * (1.0 - math.exp(-4.5 / sigma_px ** 2)))


def measure_one(cfg, out, pl, gaia_fn=gaia_cone, show=True, known_fn=None):
    from astropy.io import fits
    rng = np.random.default_rng(12345)  # fixed, so a rerun gives the same numbers
    tag = "%s_%s" % (pl["survey"], pl["plate"])
    tgt = os.path.join(out, "cutouts", tag + "__target.fits")
    hdu = fits.open(tgt)[0]
    h = hdu.header; img = hdu.data.astype(float)
    info = plate_info(h)
    if plate_info(h)["plate"] != pl["plate"]:
        raise RuntimeError("the target cutout is plate %s, not the planned %s" % (info["plate"], pl["plate"]))
    if abs((info["mid"] - dt.datetime.fromisoformat(pl["mid"])).total_seconds()) > 1.0:
        raise RuntimeError("plate time differs between the plan and the cutout header (%s vs %s)" % (pl["mid"], info["mid"]))
    res = dict(plate=tag, mid=info["mid"].isoformat(), exp_min=info["exp_min"], site=info["site"], mpc=info["mpc"],
               V=pl["V"], ra3=pl["ra3"], dec3=pl["dec3"], smaa=pl.get("smaa"), smia=pl.get("smia"),
               theta=pl.get("theta"), pred_ra=pl["pred_ra"], pred_dec=pl["pred_dec"], flag=info["flag"],
               time_note=info["time_note"], reportable=info["reportable"], header_dateobs=info["header_dateobs"],
               clock_sigma_min=info.get("clock_sigma_min"))
    cov_o, cov_src = ellipse_cov(pl)
    res["uncertainty_source"] = cov_src
    if cov_o is None:
        res["status"] = "refused: no uncertainty for the prediction"
        return res, []
    year = decyear(info["mid"])
    ra0, dec0 = pl["pred_ra"], pl["pred_dec"]
    size_deg = max(h["NAXIS1"], h["NAXIS2"]) * h["XPIXELSZ"] * h.get("PLTSCALE", 67.2) / 1000.0 / 3600.0
    gcat = gaia_fn(ra0, dec0, size_deg * 0.75)
    gep = gaia_at_epoch(gcat, year)
    det, noise, sigma_px = detect(img, cfg.get("nsig", 4.0))
    r_img = residual_image(img)
    res["n_det"] = len(det)
    fit = fit_astrometry(det, h, gep, ra0, dec0)
    if fit is None:
        res["status"] = "astrometry failed (too few Gaia matches)"
        return res, []
    res.update(n_ref=fit["n"], rms_arcsec=round(fit["rms"], 3), order=fit["order"], psf_sigma_px=round(sigma_px, 3))
    ra, dec = apply_astrometry(fit, det[:, 0], det[:, 1])
    zp = phot_zero(fit)
    dxi, deta = tangent(ra, dec, ra0, dec0)
    bx, by = search_half_box(cfg, pl)
    res["box_arcsec"] = [round(bx, 1), round(by, 1)]
    if not box_on_cutout(h, ra0, dec0, max(bx, by)):
        res["flag"] = (res["flag"] + "; " if res["flag"] else "") + "search box runs off the cutout (part not searched)"
        res["reportable"] = False
    scale = math.sqrt(abs(fit["cx"][1] * fit["cy"][2] - fit["cx"][2] * fit["cy"][1]))  # arcsec per pixel
    res["scale_arcsec_px"] = round(scale, 4)
    md = sorted(fit["matched_det"])
    star_conc = np.nanmedian(det[md, 4]) if md else float("nan")
    tr_err = fit["rms"] * math.sqrt(fit["npar"] / max(fit["n"], 1))  # error of the transformation itself
    # ---- comparison plates: only those taken when the object was far from here (moved > min_move arcsec), so a
    # same-night partner plate (POSS-I red/blue) never counts as another date; each is measured by forced photometry
    rate = pl.get("rate", float("nan"))
    min_move = cfg.get("min_move_arcsec", 30.0)
    comps = []; same_night = []
    seen = {}  # plate id -> the survey name it was first used under (the archive serves some plates under two names)
    for fn in sorted(os.listdir(os.path.join(out, "cutouts"))):
        if not fn.startswith(tag + "__cmp_") or not fn.endswith(".fits"):
            continue
        ch = fits.open(os.path.join(out, "cutouts", fn))[0]
        name = fn.split("__cmp_")[1][:-5]
        try:
            cinfo = comparison_info(ch.header)
        except Exception as e:  # noqa
            log("    comparison %s skipped: %s" % (fn, e)); continue
        if cinfo["plate"] in seen:
            same_night.append("%s: the same plate %s as %s; counted once" % (name, cinfo["plate"], seen[cinfo["plate"]]))
            continue
        seen[cinfo["plate"]] = name
        reg_t, reg_c = str(h.get("REGION", "")).strip(), str(ch.header.get("REGION", "")).strip()
        if re.match(r"^X[EO]\d{3}$", reg_t) and re.match(r"^X[EO]\d{3}$", reg_c) and reg_t[2:] == reg_c[2:]:
            same_night.append("%s %s (the POSS-I partner plate of this field, same night)" % (name, cinfo["plate"])); continue
        dth = abs((cinfo["mid"] - info["mid"]).total_seconds()) / 3600.0
        if not np.isfinite(rate) or rate * dth < min_move:
            same_night.append("%s %s (%.1f h apart)" % (name, cinfo["mid"].isoformat()[:16], dth)); continue
        cimg = ch.data.astype(float)
        cdet, cnoise, cspx = detect(cimg, cfg.get("nsig", 4.0))
        cfit = fit_astrometry(cdet, ch.header, gaia_at_epoch(gcat, decyear(cinfo["mid"])), ra0, dec0)
        if cfit is None:
            same_night.append("%s %s (astrometry failed; not used)" % (name, cinfo["mid"].isoformat()[:10])); continue
        cr = residual_image(cimg)
        cnoise_f = forced_noise(cr, cdet, cspx, rng)
        if not np.isfinite(cnoise_f):
            same_night.append("%s %s (noise not measurable; not used)" % (name, cinfo["mid"].isoformat()[:10])); continue
        comps.append(dict(name=name, plate_id=cinfo["plate"], date=cinfo["mid"].isoformat()[:10], h=ch.header, fit=cfit, r=cr, det=cdet,
                          note=("; header date implausible" if "implausible" in cinfo.get("flag", "") else ""),
                          spx=cspx, zp=phot_zero(cfit), noise=cnoise_f,
                          cscale=math.sqrt(abs(cfit["cx"][1] * cfit["cy"][2] - cfit["cx"][2] * cfit["cy"][1]))))
    # ---- sources in the box with a neighbor closer than blend_fwhm star FWHMs: fit them together (a one-source fit
    # is pulled toward the neighbor); keep the one-source position to size the effect
    blend_px = cfg.get("blend_fwhm", 2.0) * 2.3548 * sigma_px
    single = {}
    for i in range(len(det)):
        if abs(dxi[i]) > bx or abs(deta[i]) > by or blend_px <= 0:
            continue
        dd = np.hypot(det[:, 0] - det[i, 0], det[:, 1] - det[i, 1])
        nbr = [int(j) for j in np.where((dd > 0.5) & (dd < blend_px))[0]]
        if not nbr:
            continue
        amp_of = lambda j: det[j, 9] if np.isfinite(det[j, 9]) else det[j, 6]  # noqa: E731
        pj = psf_fit_joint(r_img, det[i, 0] - 1.0, det[i, 1] - 1.0,
                           [(det[j, 0] - 1.0, det[j, 1] - 1.0, amp_of(j)) for j in nbr], sigma_px)
        if pj is None:
            continue
        single[i] = (float(det[i, 0]), float(det[i, 1]), [round(float(dd[j]) * scale, 1) for j in nbr])
        det[i, 0], det[i, 1], det[i, 9] = pj[0] + 1.0, pj[1] + 1.0, pj[2]
    if single:
        ra, dec = apply_astrometry(fit, det[:, 0], det[:, 1])
        dxi, deta = tangent(ra, dec, ra0, dec0)
    # ---- candidates in the box
    cands = []
    for i in range(len(det)):
        if abs(dxi[i]) > bx or abs(deta[i]) > by:
            continue
        dstar, srad = gaia_star_distance(gep, year, dxi[i], deta[i], ra0, dec0)
        mag = float(zp[0] * (-2.5 * math.log10(det[i, 2])) + zp[1]) if (zp is not None and det[i, 2] > 0) else float("nan")
        conc_rel = det[i, 4] / star_conc if np.isfinite(star_conc) and star_conc > 0 else float("nan")
        gx, gy = tangent(gep[:, 0], gep[:, 1], ra0, dec0) if len(gep) else (np.array([99.0]), np.array([99.0]))
        dg = np.hypot(gx - dxi[i], gy - deta[i])
        nb = (dg > 3.0) & (dg < 12.0) & (gep[:, 2] < (mag - 1.5 if np.isfinite(mag) else 17.0)) if len(gep) else np.array([False])
        sx = math.hypot(det[i, 7] * scale, tr_err) if np.isfinite(det[i, 7]) else fit["rms"]
        sy = math.hypot(det[i, 8] * scale, tr_err) if np.isfinite(det[i, 8]) else fit["rms"]
        cands.append(dict(x=round(det[i, 0], 2), y=round(det[i, 1], 2), ra=float(ra[i]), dec=float(dec[i]),
                          dra_arcsec=round(float(dxi[i]), 2), ddec_arcsec=round(float(deta[i]), 2),
                          snr=round(float(det[i, 3]), 1), mag_g_scale=round(mag, 2), conc_rel=round(float(conc_rel), 2),
                          fwhm_px=round(float(det[i, 5]), 2), flux=float(det[i, 2]),
                          amp=float(det[i, 9]) if np.isfinite(det[i, 9]) else float(det[i, 6]),
                          gaia_star_within=round(dstar, 2), gaia_radius=round(srad, 2),
                          sig_ra_arcsec=sx, sig_dec_arcsec=sy, err_source="formal (provisional)",
                          bright_star_within_12=bool(np.any(nb)), static_test=[]))
        if i in single:
            sr, sd = apply_astrometry(fit, np.array([single[i][0]]), np.array([single[i][1]]))
            s_xi, s_eta = tangent(sr, sd, ra0, dec0)
            cands[-1]["blend"] = dict(neighbors_arcsec=single[i][2],
                                      shift_from_one_source_fit=[round(float(dxi[i] - s_xi[0]), 2), round(float(deta[i] - s_eta[0]), 2)])

    def mahal(c):
        C = cov_o + np.diag([c["sig_ra_arcsec"] ** 2, c["sig_dec_arcsec"] ** 2])
        d = np.array([c["dra_arcsec"], c["ddec_arcsec"]])
        return float(math.sqrt(max(d @ np.linalg.solve(C, d), 0.0)))
    # ---- error bars from this plate: references of similar brightness, and stars injected at the same brightness
    for c in cands:
        c["dist_sigma"] = mahal(c)
        # every source in the box gets plate-based error bars: the formal error alone can be several times too small
        if c["gaia_star_within"] < c["gaia_radius"] or (np.isfinite(c["conc_rel"]) and c["conc_rel"] > 2.0):
            continue
        ix, iy, ninj = injection_errors(r_img, det, sigma_px, max(c["amp"], 1e-6), rng)  # fitted, not the noisy peak
        a_x = math.hypot(ix * scale, tr_err) if np.isfinite(ix) else float("nan")
        a_y = math.hypot(iy * scale, tr_err) if np.isfinite(iy) else float("nan")
        sim = np.abs(fit["gmag"] - c["mag_g_scale"]) < 1.0 if np.isfinite(c["mag_g_scale"]) else np.zeros(fit["n"], bool)
        b_x = b_y = float("nan")
        if sim.sum() >= 8:
            b_x = float(1.4826 * np.median(np.abs(fit["res_x"][sim])))
            b_y = float(1.4826 * np.median(np.abs(fit["res_y"][sim])))
        f_x, f_y = c["sig_ra_arcsec"], c["sig_dec_arcsec"]
        vx = [v for v in (f_x, a_x, b_x) if np.isfinite(v)]
        vy = [v for v in (f_y, a_y, b_y) if np.isfinite(v)]
        c["sig_ra_arcsec"], c["sig_dec_arcsec"] = max(vx), max(vy)
        c["err_parts"] = dict(formal=[round(f_x, 3), round(f_y, 3)], injected=[round(a_x, 3), round(a_y, 3), ninj],
                              similar_refs=[round(b_x, 3), round(b_y, 3), int(sim.sum())])
        c["err_source"] = "largest of formal, injected stars (%d), similar-brightness references (%d)" % (ninj, int(sim.sum()))
        if "blend" in c:  # plus half the shift that fitting the neighbor jointly made, as a deblending error
            hx, hy = [0.5 * abs(v) for v in c["blend"]["shift_from_one_source_fit"]]
            c["sig_ra_arcsec"] = math.hypot(c["sig_ra_arcsec"], hx); c["sig_dec_arcsec"] = math.hypot(c["sig_dec_arcsec"], hy)
            c["err_parts"]["deblend"] = [round(hx, 3), round(hy, 3)]
            c["err_source"] += "; plus half the shift from the joint fit with its neighbor(s)"
        c["dist_sigma"] = mahal(c)
    # ---- static test by forced photometry on the other-date plates
    for c in cands:
        for cp in comps:
            try:
                xc, yc = invert_astrometry(cp["fit"], cp["h"], c["ra"], c["dec"])
            except Exception:  # noqa
                c["static_test"].append([cp["name"], cp["date"], "off"]); continue
            x0, y0 = xc - 1.0, yc - 1.0
            amp = forced_amp(cp["r"], x0, y0, cp["spx"])
            if not np.isfinite(amp) or not np.isfinite(cp["noise"]):
                c["static_test"].append([cp["name"], cp["date"], "off"]); continue
            snr = amp / cp["noise"]
            exp_snr = float("nan")
            if cp["zp"] is not None and np.isfinite(c["mag_g_scale"]):
                fl = 10 ** (-((c["mag_g_scale"] - cp["zp"][1]) / cp["zp"][0]) / 2.5)
                exp_snr = amp_from_flux(fl, cp["spx"]) / cp["noise"]
            if snr >= 3.0:
                p = psf_fit(cp["r"], x0, y0, cp["spx"])
                off = math.hypot(p[0] - x0, p[1] - y0) * cp["cscale"] if p is not None else float("nan")
                verdict = "present" if (np.isfinite(off) and off <= 1.5) else "neighbor"
            elif snr < 2.0 and np.isfinite(exp_snr) and exp_snr >= 5.0:
                verdict = "absent"
            else:
                verdict = "too shallow" if (np.isfinite(exp_snr) and exp_snr < 5.0) else "unclear"
            c["static_test"].append([cp["name"], cp["date"], verdict, round(float(snr), 1),
                                     round(float(exp_snr), 1) if np.isfinite(exp_snr) else None])
    # ---- verdicts
    for c in cands:
        why = []
        if c["gaia_star_within"] < c["gaia_radius"]:
            why.append("Gaia star")
        pres = [t for t in c["static_test"] if t[2] == "present"]
        if pres:
            why.append("static (present on %s)" % ", ".join("%s %s" % (t[0], t[1]) for t in pres))
        if np.isfinite(c["conc_rel"]) and c["conc_rel"] > 2.0:
            why.append("too concentrated for a star image (flaw?)")
        c["rejected"] = "; ".join(why)
        notes = []
        if c["bright_star_within_12"]:
            notes.append("brighter star within 12\" (centroid may be pulled)")
        if "blend" in c:
            notes.append("blended: fitted jointly with a source %s\" away" % "/".join("%.1f" % v for v in c["blend"]["neighbors_arcsec"]))
        if not any(t[2] == "absent" for t in c["static_test"]):
            notes.append("not shown absent on any other-date plate")
        c["note"] = "; ".join(notes)
    # known asteroids and comets at that moment (SkyBoT), other than the target itself; the cone holds the whole box
    # plus 600", so every object whose place is known to 200" or better (3 sigma 600") and could reach into the box is in it
    known_checked = False
    if cfg.get("check_known", True) and (known_fn is not None or gaia_fn is gaia_cone):
        kfn = known_fn or known_objects
        try:
            kn = kfn(info["site"], jd_from_datetime(info["mid"]), ra0, dec0, math.hypot(bx, by) + KNOWN_CONE_MARGIN)
            keys = name_keys(list(cfg.get("names", [cfg["designation"]])) + [pl.get("target_name", "")])
            pv = pl.get("V")
            others, target, possible = known_object_verdicts(cands, kn, keys, info["exp_min"],
                                                             pv if isinstance(pv, (int, float)) else float("nan"), cov_o)
            res["known_objects_in_box"] = [o["name"] for o in others if abs(o["dra_arcsec"]) <= bx and abs(o["ddec_arcsec"]) <= by]
            res["known_objects"] = others
            res["skybot_target"] = target or "the target is not in SkyBoT's list"
            if possible:
                res["known_possible_target"] = possible
                res["flag"] = (res["flag"] + "; " if res["flag"] else "") + \
                    "the target is not in SkyBoT's list under its names, but %s fits the prediction; it may be the target " \
                    "under another name (check by hand)" % ", ".join(o["name"] for o in possible)
                res["reportable"] = False
            known_checked = True
        except Exception as e:  # noqa
            res["known_objects_in_box"] = "check failed: %s" % str(e)[:200]
            res["flag"] = (res["flag"] + "; " if res["flag"] else "") + "known-object check failed"
    else:
        res["flag"] = (res["flag"] + "; " if res["flag"] else "") + "known-object check not run"
    if not known_checked:
        res["reportable"] = False
    for c in cands:
        for kk in ("sig_ra_arcsec", "sig_dec_arcsec", "dist_sigma"):
            c[kk] = round(float(c[kk]), 3)
    good = sorted([c for c in cands if not c["rejected"]], key=lambda c: c["dist_sigma"])
    inside = [c for c in good if c["dist_sigma"] <= 3.0]
    # a known object rejected inside the prediction's 3-sigma region: the target could be among them
    kin = [c for c in cands if "known object" in c["rejected"] and c["dist_sigma"] <= 3.0]
    if kin:
        res["flag"] = (res["flag"] + "; " if res["flag"] else "") + "a source within 3 sigma of the prediction was " \
            "rejected as a known object (%s); check by hand" % "; ".join(c["rejected"] for c in kin)
        res["reportable"] = False
    res["n_in_box"] = len(cands); res["n_candidates"] = len(good); res["n_within_3sigma"] = len(inside)
    res["rate_arcsec_h"] = round(pl.get("rate", float("nan")), 3)
    res["half_exposure_shift_arcsec"] = round(pl.get("rate", 0.0) * info["exp_min"] / 120.0, 2)
    res["comparison_dates"] = ["%s %s %s (noise %.0f%s)" % (c["name"], c["plate_id"], c["date"], c["noise"], c["note"]) for c in comps]
    res["n_comparison_plates"] = len(comps)  # distinct plates
    res["not_used_as_comparison"] = same_night
    if not comps:
        res["flag"] = (res["flag"] + "; " if res["flag"] else "") + "no other-date plate to rule out static sources"
        res["reportable"] = False
    if len(inside) == 1:
        b = inside[0]
        res.update(best_ra=b["ra"], best_dec=b["dec"], best_dra=b["dra_arcsec"], best_ddec=b["ddec_arcsec"],
                   best_mag=b["mag_g_scale"], best_snr=b["snr"], best_sig_ra=b["sig_ra_arcsec"],
                   best_sig_dec=b["sig_dec_arcsec"], best_dist_sigma=b["dist_sigma"], best_note=b["note"])
        res["result"] = "unique candidate within 3 sigma"
        if b.get("known_uncertain"):
            res["flag"] = (res["flag"] + "; " if res["flag"] else "") + \
                "the candidate could be known object %s, whose place is uncertain; check by hand" % ", ".join(b["known_uncertain"])
            res["reportable"] = False
    elif len(inside) > 1:
        res["result"] = "ambiguous: %d candidates within 3 sigma" % len(inside)
    else:
        res["result"] = "no candidate within 3 sigma"
    if len(inside) != 1:
        res["reportable"] = False  # nothing to report
    res["status"] = "ok"
    if show:
        make_picture(out, tag, img, h, fit, pl, cands, bx, by)
    return res, cands


def make_picture(out, tag, img, h, fit, pl, cands, bx, by):
    from PIL import Image, ImageDraw
    v = img - np.median(img)
    s = np.percentile(v, 99.5) or 1.0
    a = np.arcsinh(np.clip(v, 0, None) / (0.1 * s)) / np.arcsinh(10.0)
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)[::-1]  # FITS y up -> picture y down
    im = Image.fromarray(a).convert("RGB")
    sc = 3 if im.size[0] < 500 else 2
    im = im.resize((im.size[0] * sc, im.size[1] * sc), Image.NEAREST)
    d = ImageDraw.Draw(im)
    ny = img.shape[0]

    def topix(ra_, dec_):
        """picture position of ra_, dec_ under the refitted astrometry (the header solution if that fails)"""
        try:
            x_, y_ = invert_astrometry(fit, h, float(ra_), float(dec_))
        except Exception:  # noqa
            px_, py_ = dss_sky2pix(h, ra_, dec_)
            x_, y_ = float(px_[0]), float(py_[0])
        return (x_ - 1) * sc, (ny - y_) * sc
    corners = [(-bx, -by), (bx, -by), (bx, by), (-bx, by)]
    pts = [topix(*untangent(cx_, cy_, pl["pred_ra"], pl["pred_dec"])) for cx_, cy_ in corners]
    d.polygon(pts, outline=(80, 160, 255))
    cov_o, _ = ellipse_cov(pl)
    if cov_o is not None:
        w_, v_ = np.linalg.eigh(cov_o)
        ep = []
        for t_ in np.linspace(0, 2 * math.pi, 73):
            e_ = 3.0 * (math.sqrt(w_[0]) * math.cos(t_) * v_[:, 0] + math.sqrt(w_[1]) * math.sin(t_) * v_[:, 1])
            ep.append(topix(*untangent(e_[0] + 1.0 * math.cos(t_), e_[1] + 1.0 * math.sin(t_), pl["pred_ra"], pl["pred_dec"])))
        d.line(ep, fill=(255, 210, 60))
    X, Y = topix(pl["pred_ra"], pl["pred_dec"])
    d.line([X - 12, Y, X - 5, Y], fill=(80, 160, 255)); d.line([X + 5, Y, X + 12, Y], fill=(80, 160, 255))
    for c in cands:
        X = (c["x"] - 1) * sc; Y = (ny - c["y"]) * sc
        col = (255, 80, 80) if c["rejected"] else (80, 255, 120)
        d.ellipse([X - 9, Y - 9, X + 9, Y + 9], outline=col)
    im.save(os.path.join(out, "pictures", tag + ".png"))


def step_measure(cfg, out):
    plan = read_plan(out)
    st = os.path.join(out, "fetch_status.txt")  # (folders from version 1.0.0 have none)
    if os.path.exists(st) and not open(st).read().startswith("COMPLETE"):
        raise FetchIncomplete("the last --fetch of this subject did not finish, or the plan is newer (%s); run --fetch "
                              "first" % st)
    rows = []; allc = {}
    for pl in plan:
        try:
            r, c = measure_one(cfg, out, pl)
        except Exception as e:  # noqa
            r, c = dict(plate="%s_%s" % (pl["survey"], pl["plate"]), status="error: %s" % e), []
        rows.append(r); allc[r["plate"]] = c
        log("  %-26s %s  refs %s rms %s\"  in box %s, not rejected %s, within 3 sigma %s: %s%s" % (
            r["plate"], r.get("mid", "")[:16], r.get("n_ref"), r.get("rms_arcsec"), r.get("n_in_box"),
            r.get("n_candidates"), r.get("n_within_3sigma"), r.get("result", r.get("status")),
            (" -> %+.2f\" %+.2f\" (+-%.2f\", %.2f\") mag %.1f, %.1f sigma" % (
                r["best_dra"], r["best_ddec"], r["best_sig_ra"], r["best_sig_dec"], r["best_mag"], r["best_dist_sigma"]))
            if "best_dra" in r else ""))
    write_json(os.path.join(out, "results.json"), dict(version=VERSION, subject=cfg["designation"], plates=rows, candidates=allc))
    keys = ["plate", "mid", "exp_min", "site", "mpc", "V", "pred_ra", "pred_dec", "ra3", "dec3", "smaa", "smia", "theta",
            "n_ref", "rms_arcsec", "n_in_box", "n_candidates", "n_within_3sigma", "result", "best_ra", "best_dec",
            "best_dra", "best_ddec", "best_sig_ra", "best_sig_dec", "best_dist_sigma", "best_mag", "best_snr",
            "best_note", "known_objects_in_box", "comparison_dates", "status", "reportable", "flag", "time_note",
            "header_dateobs"]
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    log("-> results.csv, results.json, pictures/")

# --------------------------------------------------------------------------------------------- self-test

def self_test():
    """synthetic plates in a real DSS header layout, with DSS-like correlated noise. Covers real POSS-I and UK Schmidt
    header formats; Hill's date and clock tables; a same-night partner plate; a faint static source; a Gaia star
    without proper motion; a neighbor on the other-date plate; a long tilted error ellipse with an impostor near the
    center; a known asteroid; SkyBoT's answer format and the known-object rules; the JSON writer; --object; missing
    uncertainties; header errors; blended neighbors; archive errors while planning; and the stated error bars against
    the real scatter."""
    import tempfile
    from astropy.io import fits
    rng = np.random.default_rng(7)
    ok = True

    def check(cond, what):
        nonlocal ok
        log("  %-4s %s" % ("ok" if cond else "FAIL", what))
        ok = ok and bool(cond)

    def make_header(ra_c, dec_c, n=240, date="15/03/54", ut="06:00:00", exp=45.0):
        h = fits.Header()
        h["NAXIS"] = 2; h["NAXIS1"] = n; h["NAXIS2"] = n
        ra_h = ra_c / 15.0; hh = int(ra_h); mm = int((ra_h - hh) * 60); ss = ((ra_h - hh) * 60 - mm) * 60
        h["PLTRAH"] = hh; h["PLTRAM"] = mm; h["PLTRAS"] = ss
        h["PLTDECSN"] = "-" if dec_c < 0 else "+"; ad = abs(dec_c)
        h["PLTDECD"] = int(ad); h["PLTDECM"] = int((ad - int(ad)) * 60); h["PLTDECS"] = ((ad - int(ad)) * 60 - h["PLTDECM"]) * 60
        h["XPIXELSZ"] = 25.284450; h["YPIXELSZ"] = 25.284450; h["PLTSCALE"] = 67.2
        h["PPO3"] = 176616.98338; h["PPO6"] = 177193.56115606
        h["AMDX1"] = 67.226; h["AMDX2"] = -0.23; h["AMDX3"] = 1.09; h["AMDX4"] = 1.2e-6; h["AMDX8"] = 2.3e-6
        h["AMDY1"] = 67.233; h["AMDY2"] = 0.226; h["AMDY3"] = -0.527; h["AMDY4"] = -7.3e-6; h["AMDY8"] = 2.2e-6
        X0 = h["PPO3"] / h["XPIXELSZ"]; Y0 = h["PPO6"] / h["YPIXELSZ"]
        h["CNPIX1"] = int(X0 - n / 2); h["CNPIX2"] = int(Y0 - n / 2)
        h["DATE-OBS"] = date; h["UT"] = ut; h["EXPOSURE"] = exp; h["TELESCOP"] = "Palomar 48-inch Schmidt"
        h["REGION"] = "XP000"; h["PLATEID"] = "T1"
        return h

    def render(h, ra, dec, mags, fwhm_pix=2.2, flaw=None, corr=0.74):
        """stars plus noise correlated like a DSS scan (Gaussian kernel 0.74 px; lag-1 correlation ~0.63)"""
        n = h["NAXIS1"]
        w = rng.normal(0, 1, (n, n))
        if corr:
            w = ndimage.gaussian_filter(w, corr); w /= w.std()
        img = 5000 + 40 * w
        x, y = dss_sky2pix(h, ra, dec)
        yy, xx = np.mgrid[1:n + 1, 1:n + 1]
        s_ = fwhm_pix / 2.3548
        for xi_, yi_, m in zip(x, y, mags):
            if 0 < xi_ < n + 1 and 0 < yi_ < n + 1:
                amp = 6000 * 10 ** (-0.4 * (m - 16))  # V = 19 at S/N ~7, as on real 1950s plates
                x0, y0 = int(xi_), int(yi_)
                ya, yb = max(0, y0 - 8), min(n, y0 + 8); xa, xb = max(0, x0 - 8), min(n, x0 + 8)
                img[ya:yb, xa:xb] += amp * np.exp(-((xx[ya:yb, xa:xb] - xi_) ** 2 + (yy[ya:yb, xa:xb] - yi_) ** 2) / (2 * s_ * s_))
        if flaw is not None:
            img[flaw[1] - 1, flaw[0] - 1] += 2500
        return img

    # ---------------- 1. header reading (real DSS header formats; Palomar's log; Hill's tables)
    tmp = tempfile.mkdtemp()
    tab = os.path.join(tmp, "dates.csv")
    with open(tab, "w") as f:
        f.write("plate,label,header_date,offset_days,source\nXE733,1029.0,1954-04-02,1.0,block rule\n"
                "XE676,1600.0,1956-05-07,1.0,Palomar log\nXE723,1619.0,1958-12-10,,unresolved\n")
    load_poss1_dates(tab)
    load_poss1_clock(os.path.join(tmp, "none.csv"))  # no per-plate clock table for the first cases (survey median)

    def hdr(**kw):
        x = fits.Header()
        for k_, v_ in kw.items():
            x["DATE-OBS" if k_ == "DATE_OBS" else k_] = v_
        return x
    P = dict(TELESCOP="Palomar Schmidt")
    cases = [  # header, expected start (UT), expected reportable
        (hdr(DATE_OBS="1956-05-07T05:07:00", EXPOSURE=45.0, REGION="XE676", PLATEID="0DGZ", PLTLABEL="E1600", **P),
         dt.datetime(1956, 5, 8, 5, 3, 24), True),  # Palomar's log: 21:04 PST on the evening of 7 May
        (hdr(DATE_OBS="1954-04-02T05:82:00", EXPOSURE=45.0, REGION="XE733", PLATEID="0EGW", PLTLABEL="E1029", **P),
         dt.datetime(1954, 4, 3, 5, 48, 24), True),
        (hdr(DATE_OBS="1954-04-02T06:65:00", EXPOSURE=12.0, REGION="XO733", PLATEID="A3RO", PLTLABEL="O1029", **P),
         dt.datetime(1954, 4, 3, 6, 38, 12), True),
        (hdr(DATE_OBS="1958-12-10T08:20:00", EXPOSURE=45.0, REGION="XE723", PLATEID="0X", PLTLABEL="E1619", **P), None, False),
        (hdr(DATE_OBS="1954-04-02T05:82:00", EXPOSURE=4.5, REGION="XE733", PLATEID="0EGW", PLTLABEL="E1029", **P), None, False),
        (hdr(DATE_OBS="1983-04-17T13:00", EXPOSURE=60.0, TELESCOP="UK Schmidt - Doubl", REGION="S789", PLATEID="024P",
             PLTLABEL="J8488"), dt.datetime(1983, 4, 17, 12, 30, 0), False),  # UK Schmidt: header time = mid-exposure
        (hdr(DATE_OBS="1954-04-02T05:82:00", EXPOSURE=45.0, REGION="XO733", PLATEID="A3RO", PLTLABEL="O1030", **P), None, False),
        (hdr(DATE_OBS="11/03/76", UT="17:38:00.00", EXPOSURE=70.0, EPOCH=1976.1932373047, TELESCOP="UK 48-inch Schmidt",
             REGION="S134", PLATEID="0025", PLTLABEL="J 2098"), dt.datetime(1976, 3, 11, 17, 3, 0), False),
        (hdr(DATE_OBS="11/03/76", UT="17:38:00.00", EXPOSURE=70.0, EPOCH=1976.2032, TELESCOP="UK 48-inch Schmidt",
             REGION="S134", PLATEID="0025", PLTLABEL="J 2098"), None, False),  # EPOCH 3.7 days off
    ]
    for hh_, want, rep in cases:
        inf_ = plate_info(hh_)
        good = (inf_["reportable"] == rep) and (want is None or abs((inf_["start"] - want).total_seconds()) < 1)
        check(good, "header %-20s %-6s -> start %s, reportable %s %s" % (
            hh_["DATE-OBS"], hh_["PLTLABEL"], inf_["start"].isoformat(), inf_["reportable"], inf_["flag"]))
    if [x for x in cases[-1:] if "EPOCH" not in plate_info(x[0])["flag"]]:
        check(False, "EPOCH 3.7 days off is flagged")
    bad = [("minutes > 59 on a non-POSS-I plate", dict(DATE_OBS="1983-04-17T13:75", EXPOSURE=60.0, TELESCOP="UK Schmidt")),
           ("no EXPOSURE", dict(DATE_OBS="1983-04-17T13:10", TELESCOP="UK Schmidt")),
           ("no time of day", dict(DATE_OBS="1983-04-17", EXPOSURE=60.0, TELESCOP="UK Schmidt")),
           ("ISO time and UT disagree", dict(DATE_OBS="1983-04-17T13:10:00", UT="14:10:00", EXPOSURE=60.0, TELESCOP="UK Schmidt")),
           ("unknown telescope", dict(DATE_OBS="1983-04-17T13:10", EXPOSURE=60.0, TELESCOP="Oschin 48-inch"))]
    for what, kw in bad:
        try:
            plate_info(hdr(REGION="S1", PLATEID="1", PLTLABEL="J1", **kw))
            check(False, "%s refused" % what)
        except ValueError:
            check(True, "%s refused" % what)
    # Hill's per-plate clock: used when the table has the plate (a blue plate takes its red partner's); a plate whose
    # mid-exposure disagrees with Hill's time by more than 1 min is not reportable
    clk = os.path.join(tmp, "clock.csv")
    with open(clk, "w") as f:
        f.write("plate,on_line,tau_median,tau_q16,tau_q84,from_prior_centre,ut\n"
                "XE733,1,-0.60,-1.22,0.02,0.0,1954-04-03 06:11:06.000000000\n"
                "XE676,0,-1.11,-1.73,-0.48,0.0,1956-05-08 05:40:00.000000000\n")
    load_poss1_clock(clk)
    ck = [(hdr(DATE_OBS="1954-04-02T05:82:00", EXPOSURE=45.0, REGION="XE733", PLATEID="0EGW", PLTLABEL="E1029", **P),
           dt.datetime(1954, 4, 3, 5, 48, 36), True),
          (hdr(DATE_OBS="1954-04-02T06:65:00", EXPOSURE=12.0, REGION="XO733", PLATEID="A3RO", PLTLABEL="O1029", **P),
           dt.datetime(1954, 4, 3, 6, 38, 24), True),
          (hdr(DATE_OBS="1956-05-07T05:07:00", EXPOSURE=45.0, REGION="XE676", PLATEID="0DGZ", PLTLABEL="E1600", **P),
           dt.datetime(1956, 5, 8, 5, 3, 5, 400000), False)]
    for hh_, want, rep_ in ck:
        inf_ = plate_info(hh_)
        check(abs((inf_["start"] - want).total_seconds()) < 1 and inf_["reportable"] == rep_ and "per-plate clock" in inf_["time_note"],
              "per-plate clock: %s %s -> start %s, reportable %s %s" % (hh_["REGION"], hh_["PLTLABEL"], inf_["start"].isoformat(),
                                                                         inf_["reportable"], inf_["flag"]))
    badclk = os.path.join(tmp, "badclock.csv")
    with open(badclk, "w") as f:
        f.write("plate,on_line,tau_median,tau_q16,tau_q84,from_prior_centre,ut\nXE733,1,-0.60,x,0.02,0.0,1954-04-03 06:11:06\n")
    try:
        load_poss1_clock(badclk)
        check(False, "a clock table with an unreadable row is refused")
    except ValueError:
        check(True, "a clock table with an unreadable row is refused")
    load_poss1_clock(os.path.join(tmp, "none.csv"))
    # a comparison plate with a telescope not known for reporting (POSS-II) is still used, dated by EPOCH
    hp2 = hdr(DATE_OBS="1991-06-10T08:30:00", EXPOSURE=50.0, EPOCH=1991.4389, TELESCOP="Oschin Schmidt - D",
              REGION="XP564", PLATEID="A1B2", PLTLABEL="F1234")
    try:
        plate_info(hp2); check(False, "POSS-II (Oschin) is not a reporting telescope")
    except ValueError:
        check(True, "POSS-II (Oschin) is not a reporting telescope")
    ci = comparison_info(hp2)
    check(ci["plate"] == "XP564_A1B2" and not ci["reportable"] and abs(decyear(ci["mid"]) - 1991.4389) < 1e-6,
          "POSS-II plate used as a comparison, dated by EPOCH (%s, %s)" % (ci["plate"], ci["mid"].isoformat()[:16]))
    ci2 = comparison_info(hdr(DATE_OBS="1991-06-10T08:30:00", EXPOSURE=50.0, TELESCOP="Oschin Schmidt - D", REGION="XP564",
                              PLATEID="A1B2", PLTLABEL="F1234"))
    check(ci2["mid"] == dt.datetime(1991, 6, 10, 8, 30) and not ci2["reportable"],
          "POSS-II plate without EPOCH dated by DATE-OBS (%s)" % ci2["mid"].isoformat()[:16])
    try:
        comparison_info(hdr(DATE_OBS="1991-06-10", EXPOSURE=50.0, TELESCOP="Oschin Schmidt - D", REGION="XP564",
                            PLATEID="A1B2", PLTLABEL="F1234"))
        check(False, "a comparison plate with no usable date or time is refused")
    except ValueError:
        check(True, "a comparison plate with no usable date or time is refused")
    bad_date = comparison_info(hdr(DATE_OBS="1952-08-22T05:53:00", EXPOSURE=55.0, TELESCOP="Oschin Schmidt - D", REGION="XP799",
                                   PLATEID="A0TP", PLTLABEL="SF05757", EMULSION="IIIaF"))
    good_date = comparison_info(hdr(DATE_OBS="1991-06-16T05:43:00", EXPOSURE=110.0, TELESCOP="Oschin Schmidt - D", REGION="XI799",
                                    PLATEID="A0SS", PLTLABEL="SN04021", EMULSION="IVN"))
    check("implausible" in bad_date["flag"] and "implausible" not in good_date["flag"],
          "a 1952 date on a POSS-II IIIaF plate is flagged; 1991 is not (%s)" % bad_date["flag"])
    load_poss1_dates(os.path.join(tmp, "none.csv"))  # an empty table for the rest of the test
    # ---------------- 2. SkyBoT's answer and the known-object rules
    # SkyBoT's real answers (VOTable), cut to the columns used here: a cone of 300" around RA 190.35549, Dec -3.96212 at
    # JD 2434887.688819 from Palomar (POSS-I plate XE675, Quaoar); and a field with nothing in it (RA 10, Dec +85)
    vo_head = ('<?xml version="1.0" encoding="UTF-8" ?>\n<vot:VOTABLE version="1.3" xsi:schemaLocation="http://www.ivoa.net/'
               'xml/VOTable/v1.3 http://www.ivoa.net/xml/VOTable/VOTable-1.3.xsd" xmlns:xsi="http://www.w3.org/2001/'
               'XMLSchema-instance" xmlns:vot="http://www.ivoa.net/xml/VOTable/v1.3" xmlns:stc="http://www.ivoa.net/xml/STC">\n'
               '<vot:INFO ID="status" name="QUERY_STATUS" value="OK"/>\n<vot:INFO ID="ticket" name="Ticket" value="%s"/>\n'
               '<vot:INFO ID="reqTarget" name="-c" value="%s"/>\n<vot:INFO ID="reqEpoch" name="Epoch" value="%s"/>\n'
               '<vot:INFO ID="supplier" name="Supplier" value="Provided by LTE/OBSPM/CNRS"/>\n<vot:RESOURCE name="Skybot.ConeSearch">'
               '<vot:DESCRIPTION>SkyBoT - List of the targets located in the FOV</vot:DESCRIPTION>\n')
    vo_fields = ('<vot:FIELD ID="num" name="Num" datatype="char" arraysize="6"/>\n'
                 '<vot:FIELD ID="name" name="Name" datatype="char" arraysize="32"/>\n'
                 '<vot:FIELD ID="ra" name="RA" ref="Ephemeris" datatype="char" arraysize="13" unit="h:m:s"/>\n'
                 '<vot:FIELD ID="de" name="DEC" ref="Ephemeris" datatype="char" arraysize="13" unit="d:m:s"/>\n'
                 '<vot:FIELD ID="class" name="Class" datatype="char" arraysize="24"/>\n'
                 '<vot:FIELD ID="magV" name="Mv" datatype="float" width="13" precision="2"/>\n'
                 '<vot:FIELD ID="errpos" name="ErrPos" datatype="float" width="10" precision="3" unit="arcsec"/>\n'
                 '<vot:FIELD ID="angdist" name="d" datatype="float" width="8" precision="3" unit="arcsec"/>\n'
                 '<vot:FIELD ID="dracosdec" name="dRAcosDec" datatype="double" width="12" unit="arcsec/h"/>\n'
                 '<vot:FIELD ID="ddec" name="dDEC" datatype="double" width="12" unit="arcsec/h"/>\n')
    vo_rows = [("50000", "Quaoar", "12 41 25.2785", "-03 57 43.369", "KBO&gt;Classical&gt;Main", "19.04", "3.637", "0.642",
                "-1.4334", "0.643"),
               ("409671", "2005 YY236", "12 41 23.4548", "-03 57 41.769", "MB&gt;Outer", "21.51", "0.427", "27.938", "-0.0715",
                "0.977"),
               ("-", "2008 UX294", "12 41 31.4411", "-03 53 35.424", "MB&gt;Middle", "22.57", "13.986", "264.584", "-8.5744",
                "-2.665")]
    q_cone, q_ep = "190.35548999999997-3.9621200000000001,rs=300", "2434887.688819"

    def votable(rows, nrows=None, res_info='<vot:INFO name="QUERY_STATUS" value="OK"/>\n', fields=None, data=None, cone=q_cone,
                ep=q_ep):
        tr = "".join("<vot:TR>%s</vot:TR>\n" % "".join("<vot:TD>%s</vot:TD>" % x for x in r) for r in rows)
        nr = "" if nrows == "absent" else ' nrows="%d"' % (len(rows) if nrows is None else nrows)
        return (vo_head % ("188053159621697502", cone, ep) + res_info +
                '<vot:TABLE ID="SkybotConeSearch_results" name="SkybotConeSearch"%s>\n' % nr + (fields or vo_fields)
                + (data or "<vot:DATA><vot:TABLEDATA>\n" + tr + "</vot:TABLEDATA>\n</vot:DATA>\n") + "</vot:TABLE>\n"
                "</vot:RESOURCE>\n</vot:VOTABLE>\n").encode()
    sky_ans = votable(vo_rows)
    sky_none = (vo_head % ("188053157801019808", "10+85,rs=120", q_ep) + '<vot:INFO name="QUERY_STATUS" value="ERROR">No '
                'solar system object was found in the requested FOV: RA(deg,hms) DEC(deg,dms): 10 (00:40:00.00) 85 '
                '(+85:00:00.00) EPOCH: 1954-05-25 04:31:53.96</vot:INFO>\n<vot:TABLE ID="SkybotConeSearch_results" '
                'name="SkybotConeSearch" nrows="0"><vot:DATA><vot:TABLEDATA></vot:TABLEDATA>\n</vot:DATA>\n</vot:TABLE>\n'
                '</vot:RESOURCE>\n</vot:VOTABLE>\n').encode()
    urls = []
    g = globals(); saved = g["http_get"]
    try:
        g["http_get"] = lambda url, **k_: (urls.append(url), sky_ans)[1]
        kn = known_objects("675", 2434887.688819, 190.35549, -3.96212, 300.0)
    finally:
        g["http_get"] = saved
    sky_err = (vo_head % ("1", q_cone, q_ep)).replace(  # the error SkyBoT gave once for a 1994 plate (the same query worked later)
        '<vot:INFO ID="status" name="QUERY_STATUS" value="OK"/>', '<vot:INFO ID="status" name="QUERY_STATUS" value="ERROR">'
        'SkyBoT comet conesearch -&gt; Bad request: Epoch 2434887.688819 not available...</vot:INFO>', 1).encode() + \
        b"</vot:RESOURCE></vot:VOTABLE>"
    seq = [sky_err, sky_ans]; n_sky = []; wait0 = g["SKYBOT_RETRY_WAIT"]
    try:
        g["SKYBOT_RETRY_WAIT"] = 0.0
        g["http_get"] = lambda url, **k_: (n_sky.append(1), seq[min(len(n_sky), 2) - 1])[1]
        kn_r = known_objects("675", 2434887.688819, 190.35549, -3.96212, 300.0); n_retry_sky = len(n_sky); del n_sky[:]
        g["http_get"] = lambda url, **k_: (n_sky.append(1), sky_err)[1]
        try:
            known_objects("675", 2434887.688819, 190.35549, -3.96212, 300.0); r_sky = None
        except RuntimeError as e:
            r_sky = str(e)
    finally:
        g["http_get"], g["SKYBOT_RETRY_WAIT"] = saved, wait0
    check(len(kn_r) == 3 and n_retry_sky == 2 and r_sky is not None and "not available" in r_sky and len(n_sky) == 3,
          "SkyBoT: an error answer is asked again (SkyBoT sometimes fails a query that works later); after 3 the check fails")
    qa = urllib.parse.parse_qs(urllib.parse.urlsplit(urls[0]).query) if urls else {}
    check(urls and urls[0].startswith(SKYBOT_URL) and qa.get("-ep") == ["2434887.688819"] and qa.get("-observer") == ["675"]
          and qa.get("-rs") == ["300"] and qa.get("-filter") == ["0"] and qa.get("-mime") == ["votable"],
          "SkyBoT asked for the plate's time, site (-observer; SkyBoT ignores -loc) and cone, with no filter on errors")
    dk = [math.hypot(k["xi"], k["eta"]) for k in kn]
    check([k["label"] for k in kn] == ["50000 Quaoar", "409671 2005 YY236", "2008 UX294"] and kn[0]["V"] == 19.04
          and abs(kn[2]["err"] - 13.986) < 1e-9 and kn[0]["cls"] == "KBO>Classical>Main" and kn[2]["dra"] == -8.5744
          and max(abs(d_ - float(r_[7])) for d_, r_ in zip(dk, vo_rows)) < 0.01,
          "SkyBoT rows read: names, V, error, class, motion, and distances as SkyBoT gives them (%s)"
          % ", ".join("%.3f\"" % d_ for d_ in dk))
    check(abs(kn[0]["xi"] + 0.585) < 0.01 and abs(kn[0]["eta"] - 0.263) < 0.01,
          "SkyBoT offsets have the right signs: Quaoar %+.3f\" east, %+.3f\" north of the center" % (kn[0]["xi"], kn[0]["eta"]))
    check(parse_skybot(sky_none, expect=(2434887.688819, 10.0, 85.0, 120)) == [],
          "SkyBoT: 'No solar system object was found' is 'no objects'")
    bad_vo = [("an HTML page", b"<html><body>Service unavailable</body></html>"),
              ("JSON", b'[{"Name": "x"}]'),
              ("a failed query", (vo_head % ("1", "x", q_ep)).replace('value="OK"', 'value="ERROR"', 1).encode()
               + b"</vot:RESOURCE></vot:VOTABLE>"),
              ("a resource error other than 'none found'", votable([], res_info='<vot:INFO name="QUERY_STATUS" value="ERROR">'
                                                                         'Epoch out of range</vot:INFO>\n')),
              ("'none found' with rows", votable(vo_rows[:1], res_info='<vot:INFO name="QUERY_STATUS" value="ERROR">No solar '
                                                 'system object was found</vot:INFO>\n')),
              ("an OK answer with no rows", votable([])),
              ("an answer without nrows", votable(vo_rows, nrows="absent")),
              ("fewer rows than nrows", votable(vo_rows, nrows=4)),
              ("rows not as TABLEDATA", votable(vo_rows, data="<vot:DATA><vot:BINARY><vot:STREAM encoding=\"base64\">AAAA"
                                                "</vot:STREAM></vot:BINARY></vot:DATA>\n")),
              ("no position-error column", votable([r[:6] for r in vo_rows], fields="".join(vo_fields.split("\n")[:6]) + "\n")),
              ("a row without a place", votable([("1", "x", "", "", "", "", "", "", "", "")])),
              ("an answer for another epoch", votable(vo_rows, ep="2434888.688819")),
              ("an answer for another cone", votable(vo_rows, cone="190.36548999999997-3.9621200000000001,rs=300"))]
    for what, raw_ in bad_vo:
        try:
            parse_skybot(raw_, expect=(2434887.688819, 190.35549, -3.96212, 300))
            check(False, "SkyBoT: %s is an error, not 'no objects'" % what)
        except RuntimeError:
            check(True, "SkyBoT: %s is an error, not 'no objects'" % what)
    nk = [name_keys([x]) for x in ("50000 Quaoar (2002 LM60)", "(2026 ED138)", "2026", "2026ED138", "185643 (2040 P-L)",
                                   "(50000) Quaoar")]
    check(nk == [({"50000"}, {"quaoar", "2002 lm60"}), (set(), {"2026 ed138"}), ({"2026"}, set()), (set(), {"2026 ed138"}),
                 ({"185643"}, {"2040 p-l"}), ({"50000"}, {"quaoar"})],
          "names read as numbers, names and designations (Horizons' '50000 Quaoar (2002 LM60)'; '2026 ED138' is not 2026)")
    check(_is_target(dict(num="50000", name="Quaoar"), name_keys(["2002 LM60", "50000 Quaoar (2002 LM60)"]))
          and not _is_target(dict(num="", name="2026 BC12"), name_keys(["2026"]))
          and not _is_target(dict(num="150000", name="Beta"), name_keys(["50000"]))
          and not _is_target(dict(num="", name="C/2026 A1"), name_keys(["2026"])),
          "the target is found by its exact number or name: '50000' is not '150000', '2026' is not '2026 BC12'")

    def kobj(name, x, y, V, err, num="", cls="MB>Outer", dra=float("nan"), ddec=float("nan")):
        return dict(num=num, name=name, label=(num + " " + name).strip(), xi=x, eta=y, V=V, err=err, cls=cls, dra=dra,
                    ddec=ddec, ra=float("nan"), dec=float("nan"))
    kc = [dict(dra_arcsec=0.0, ddec_arcsec=0.0, mag_g_scale=19.0),      # 0 the target's image
          dict(dra_arcsec=50.0, ddec_arcsec=0.0, mag_g_scale=18.0),     # 1 at a well-placed known asteroid
          dict(dra_arcsec=-50.0, ddec_arcsec=0.0, mag_g_scale=19.0),    # 2 at a known asteroid predicted 4.5 mag fainter
          dict(dra_arcsec=0.0, ddec_arcsec=50.0, mag_g_scale=19.0),     # 3 40" from an object placed to 15" (3 sigma 45")
          dict(dra_arcsec=100.0, ddec_arcsec=-100.0, mag_g_scale=19.0),  # 4 12" from a well-placed object
          dict(dra_arcsec=-100.0, ddec_arcsec=100.0, mag_g_scale=19.0),  # 5 3" from a comet predicted much fainter
          dict(dra_arcsec=150.0, ddec_arcsec=0.0, mag_g_scale=19.0),    # 6 20" from a moving asteroid's mid-exposure place,
          dict(dra_arcsec=130.0, ddec_arcsec=20.0, mag_g_scale=19.0)]   # 7 5" from its path; 20" off its path
    for c_ in kc:
        c_.update(rejected="", note="")
    objs = [kobj("Quaoar", 0.3, -0.2, 19.0, 3.6, num="50000", cls="KBO>Classical>Main"), kobj("Alpha", 52.0, 3.0, 18.5, 0.5, "12345"),
            kobj("2010 AB1", -48.0, 1.0, 23.5, 0.4), kobj("2011 CD2", 0.0, 90.0, 19.0, 15.0),
            kobj("2012 EF3", 30.0, 30.0, 19.0, 300.0), kobj("2013 GH4", 112.0, -100.0, 18.0, 1.0),
            kobj("C/1954 A1", -103.0, 100.0, 25.0, 2.0, cls="Comet"), kobj("2015 XY1", 130.0, 0.0, 19.5, 1.0, dra=40.0, ddec=0.0),
            kobj("2016 ZZ9", 20.0, 0.0, 24.5, 50.0), kobj("Beta", 400.0, 400.0, 18.0, 1.0, "150000")]
    pcov = np.diag([(2.0 / 3.0) ** 2, (2.0 / 3.0) ** 2])  # a prediction good to 2" (3 sigma)
    others, target, possible = known_object_verdicts(kc, objs, name_keys(["Quaoar", "50000"]), 45.0, 19.0, pcov)
    rj = [c_["rejected"] for c_ in kc]
    check([t_["name"] for t_ in target] == ["50000 Quaoar"] and len(others) == 9 and not possible,
          "the target is set aside (and '50000' is not '150000 Beta')")
    check(rj == ["", "known object 12345 Alpha", "", "", "", "", "known object 2015 XY1", ""]
          and "too faint" in kc[2]["note"],
          "asteroids placed to 5\": rejected within max(10\", 3 x error) of their path during the exposure; not one "
          "2 mag fainter than the source, not 12\" away")
    check(kc[3].get("known_uncertain") == ["2011 CD2"] and kc[5].get("known_uncertain") == ["C/1954 A1"]
          and not any(c_.get("known_uncertain") for c_ in kc[:3] + kc[4:5] + kc[6:]),
          "asteroids placed to 5-200\" and comets reject nothing and mark the sources they could be (an asteroid not if "
          "far too faint); an asteroid placed to 300\" only listed")
    ck_ = name_keys(["60558", "Echeclus", "60558 Echeclus (2000 EC98)"])  # SkyBoT's real rows for Echeclus, 1954 May 25
    check(_is_target(dict(num="P/Echeclus", name="174P"), ck_) and _is_target(dict(num="60558", name="Echeclus"), ck_)
          and not _is_target(dict(num="P/NEAT", name="166P"), ck_),
          "the target's comet entry ('P/Echeclus 174P' for (60558) Echeclus) is the target too")
    kc2 = [dict(dra_arcsec=0.0, ddec_arcsec=0.0, mag_g_scale=19.0, rejected="", note="")]
    others2, target2, possible2 = known_object_verdicts(kc2, objs, name_keys(["2002 LM60"]), 45.0, 19.0, pcov)
    check(not target2 and [p_["name"] for p_ in possible2] == ["50000 Quaoar"] and not kc2[0]["rejected"],
          "a target listed by SkyBoT under another name is taken as possibly the target: it rejects nothing")
    # ---------------- the DSS archive's pages: 'no plate here' against error pages (real answers of 31 Oct 2026:
    # POSS-I red at RA 100, Dec -70, where POSS-I has no plates; and a request with Dec 95)
    dss_none = (b'<HTML>\n'
                b'<HEAD> <TITLE>ERROR</TITLE> </HEAD>\n'
                b'<BODY>\n'
                b'<H1>ERROR</H1>\n'
                b'<P><HR><P>\n'
                b'Something went wrong; the image file was not created.\n'
                b'<PRE>\n'
                b'data +06 40 00.00 -70 00 00.00 1.00 1.00\n'
                b'Calibration and image data not available for field data...\n'
                b'\n'
                b'\n'
                b'\t\t\t        GetImage\n'
                b'\n'
                b'\t\t\t       Version 3.0\n'
                b'\n'
                b'\t\t        Copyright (c) 1993-2005,\n'
                b'      Association of Universities for Research in Astronomy, Inc.\n'
                b'\t\t          All Rights Reserved.\n'
                b'\n'
                b'Reading plate list: /ifs/archive/ops/mast/public/dss/headers/poss1_red.v30.lis...\n'
                b'List version 3\n'
                b'\n'
                b'\n'
                b'Field                  RA(J2000)   Dec(J2000)  Region PlId   Epoch    X     Y   Width  Hght Dsc\n'
                b'----                 ------------ ------------ ------ ---- -------- ----- ----- ----- ----- ---\n'
                b'\n'
                b'FATAL ERROR:\n'
                b'\n'
                b'\t"Aborting GetImage..."\n'
                b'\n'
                b'Program aborted.\n'
                b'\n'
                b'\n'
                b'\n'
                b'  REQUEST_URI: /cgi-bin/dss_search?v=poss1_red&r=100.000000&d=-70.000000&h=1.00&w=1.00&e=J2000&f=fits&c=none\n'
                b'  UNIQUE_ID: auVY9VQzi-ZSJGI8zSk8UAAAAc4\n'
                b'</PRE>\n'
                b'<P><HR><P>\n'
                b'<ADDRESS><A HREF="mailto:archive@stsci.edu">archive@stsci.edu</A></ADDRESS>\n'
                b'\n'
                b'</BODY>\n'
                b'</HTML>\n')
    dss_bad = (b'<HTML>\n'
               b'<HEAD><TITLE>Error</TITLE></HEAD>\n'
               b'<BODY>\n'
               b'<H1>Error</H1>\n'
               b'<P><HR><P>\n'
               b'<PRE>\n'
               b'RA out of range: 100.000000\n'
               b'(0 < RA <= 24h or 360&deg;)\n'
               b'\n'
               b'<a href="/dss/dss_help.html#coordinates">Help on coordinates...</a>\n'
               b'\n'
               b'</PRE>\n'
               b'<P><HR><P>\n'
               b'<ADDRESS><A HREF="mailto:archive@stsci.edu">archive@stsci.edu</A></ADDRESS>\n'
               b'\n'
               b'</BODY>\n'
               b'</HTML>\n')
    dss_lost = dss_none.replace(b"----                 ------------ ------------ ------ ---- -------- ----- ----- ----- ----- ---\n",
                                b"----                 ------------ ------------ ------ ---- -------- ----- ----- ----- ----- ---\n"
                                b"XE111                06 40 00.00 -70 00 00.00 XE111  0ABC 1954.123 10000 10000   300   300  1\n")
    dss_busy = b"<HTML><BODY><H1>Busy</H1>The jukebox is busy; please try again later.</BODY></HTML>"
    check(dss_page(dss_none)[0] == "none" and all(dss_page(x)[0] == "error" for x in (dss_bad, dss_lost, dss_busy, b"")),
          "DSS: the 'no plate here' page is told apart from error pages (busy, bad request, a listed plate not made)")
    calls = []
    g = globals(); saved = (g["http_get"], g["DSS_RETRY_WAIT"])
    fake_fits = io.BytesIO()
    hf = make_header(169.85, -10.95, n=40)
    fits.PrimaryHDU(np.zeros((40, 40), np.int16), hf).writeto(fake_fits)
    try:
        g["DSS_RETRY_WAIT"] = 0.0
        g["http_get"] = lambda url, **k_: (calls.append(url), dss_none)[1]
        r_none = fetch_dss("poss1_red", 100.0, -70.0, 1.0); n_none = len(calls); del calls[:]
        answers = [dss_busy, dss_busy, fake_fits.getvalue()]
        g["http_get"] = lambda url, **k_: (calls.append(url), answers[len(calls) - 1])[1]
        r_retry = fetch_dss("poss1_red", 169.85, -10.95, 1.0); n_retry = len(calls); del calls[:]
        g["http_get"] = lambda url, **k_: (calls.append(url), dss_busy)[1]
        try:
            fetch_dss("poss1_red", 169.85, -10.95, 1.0); r_err = None
        except RuntimeError as e:
            r_err = str(e)
        n_err = len(calls)
    finally:
        g["http_get"], g["DSS_RETRY_WAIT"] = saved
    check(r_none is None and n_none == 1, "DSS: 'no plate here' is no plate, asked once")
    bare = io.BytesIO(); fits.PrimaryHDU(np.zeros((10, 10), np.int16)).writeto(bare)
    try:
        g["http_get"] = lambda url, **k_: bare.getvalue()
        try:
            fetch_dss("poss1_red", 169.85, -10.95, 1.0); r_bare = "no error"
        except RuntimeError as e:
            r_bare = str(e)
    finally:
        g["http_get"] = saved[0]
    check("plate solution" in r_bare, "DSS: a FITS answer without a plate solution is an error, not 'no plate'")
    check(r_retry is not None and n_retry == 3 and r_err is not None and "error page" in r_err and n_err == 3,
          "DSS: an error page is asked again, and after 3 an error, never 'no plate' (%s)" % (r_err or "")[:60])
    # ---------------- the JSON writer and --object
    jp = os.path.join(tmp, "nan.json")
    write_json(jp, dict(a=float("nan"), b=[float("inf"), np.float64("nan"), np.int64(5), 1.5], c=(1, "x")))

    def strict(c_):
        raise ValueError(c_)
    try:
        jr = json.loads(open(jp).read(), parse_constant=strict)
        check(jr == dict(a=None, b=[None, None, 5, 1.5], c=[1, "x"]), "results.json is standard JSON: NaN and infinity as null")
    except ValueError as e:
        check(False, "results.json is standard JSON: NaN and infinity as null (%s)" % e)
    oc = [object_subject("2026ED138", [1950, 1958]), object_subject("(50000) Quaoar", [1950, 1958], ["poss1_red"]),
          object_subject("Quaoar", [1975, 2000]), object_subject("2026", [1954, 1954]), object_subject("2040 P-L", [1960, 1960]),
          object_subject("2026 ed138", [1954, 1954]), object_subject("(2026 ED138)", [1954, 1954]),
          object_subject("1125 t-2", [1954, 1954])]
    check(oc[0][0]["horizons_command"] == "DES=2026 ED138;" and oc[0][0]["names"] == ["2026 ED138"]
          and oc[0][1] == "2026_ED138_1950-1958" and "surveys" not in oc[0][0]
          and oc[1][0]["horizons_command"] == "50000;" and oc[1][0]["names"] == ["50000", "Quaoar"]
          and oc[1][0]["surveys"] == ["poss1_red"] and oc[2][0]["horizons_command"] == "Quaoar;"
          and oc[3][0]["horizons_command"] == "2026;" and oc[4][0]["horizons_command"] == "DES=2040 P-L;"
          and oc[5][0]["horizons_command"] == oc[6][0]["horizons_command"] == "DES=2026 ED138;"
          and oc[5][1] == "2026_ED138_1954-1954" and oc[7][0]["horizons_command"] == "DES=1125 T-2;",
          "--object: designation, number and name read (%s)" % "; ".join(o_[0]["horizons_command"] for o_ in oc))
    nbad = 0
    for args_ in (("C/2019 Y4", [1950, 1958]), ("2026 ED138", [1958, 1950]), ("2026 ED138", [1950, 1958], ["poss3"]),
                  ("2026 A1", [1950, 1958])):
        try:
            object_subject(*args_)
        except ValueError:
            nbad += 1
    check(nbad == 4, "--object refuses a comet designation, reversed years, an unknown survey and '2026 A1'")
    mm = []
    for cf_, tg_ in ((dict(designation="x", names=["Quaoar", "50000"], horizons_command="50000;"), "50000 Quaoar (2002 LM60)"),
                     (dict(designation="x", names=["2026", "ed138"], horizons_command="2026;"), "2026 Cottrell (1955 FF)"),
                     (dict(designation="x", names=["Eris"], horizons_command="Eris;"), "136199 Eris (2003 UB313)")):
        try:
            check_target_names(cf_, tg_); mm.append(True)
        except SubjectMismatch:
            mm.append(False)
    check(mm == [True, False, True], "the plan stops when the subject's names are not names of the object Horizons found")
    # ---------------- 3. the measuring scenario
    ra_c, dec_c = 169.85, -10.95
    h1 = make_header(ra_c, dec_c)
    N = 140
    gra = ra_c + rng.uniform(-0.055, 0.055, N) / math.cos(math.radians(dec_c))
    gdec = dec_c + rng.uniform(-0.055, 0.055, N)
    gcat = np.column_stack([gra, gdec, rng.normal(0, 15, N), rng.normal(0, 15, N), rng.uniform(15.5, 19.5, N)])
    pra, pdec = dss_pix2sky(h1, (h1["NAXIS1"] + 1) / 2.0, (h1["NAXIS2"] + 1) / 2.0)
    pra = float(pra); pdec = float(pdec)
    ep1 = gaia_at_epoch(gcat, 1954.2); ep2 = gaia_at_epoch(gcat, 1989.3)
    e1x, e1y = tangent(ep1[:, 0], ep1[:, 1], pra, pdec); e2x, e2y = tangent(ep2[:, 0], ep2[:, 1], pra, pdec)
    used = []

    def clear(x, y, r=15.0):
        return (min(np.min(np.hypot(e1x - x, e1y - y)), np.min(np.hypot(e2x - x, e2y - y))) > r and
                all(math.hypot(x - u[0], y - u[1]) > 25 for u in used) and abs(x) < 95 and abs(y) < 95)
    # a long thin error ellipse tilted at -40 deg: 3-sigma 90" x 3"
    TH = -40.0; ux, uy = math.cos(math.radians(TH)), math.sin(math.radians(TH)); vx, vy = -uy, ux

    def pick(cands_):
        for c_ in cands_:
            if clear(*c_):
                used.append(c_); return c_
        raise RuntimeError("self-test: no clear spot")
    OBJ = pick([(a_ * ux, a_ * uy) for a_ in list(np.arange(40, 66, 1.0)) + list(np.arange(-40, -66, -1.0))])  # on the line
    IMP = pick([(a_ * ux + p_ * vx, a_ * uy + p_ * vy) for p_ in (12, -12, 14, -14, 16, -16) for a_ in range(-15, 16, 3)])
    STA = pick([(a_, b_) for a_ in np.arange(-80, 81, 7.0) for b_ in np.arange(-80, 81, 7.0)])
    NOPM = pick([(a_, b_) for a_ in np.arange(-80, 81, 7.0) for b_ in np.arange(-80, 81, 7.0)])
    KNO = pick([(a_, b_) for a_ in np.arange(-80, 81, 7.0) for b_ in np.arange(-80, 81, 7.0)])
    sky = lambda s_: tuple(float(v) for v in untangent(s_[0], s_[1], pra, pdec))  # noqa: E731
    o_ = sky(OBJ); i_ = sky(IMP); s0 = sky(STA); n_ = sky(NOPM); k_ = sky(KNO)
    nb_ = sky((OBJ[0] + 3.2, OBJ[1]))  # a neighbor 3.2" from the object's place, on the other-date plate only
    gcat = np.vstack([gcat, [n_[0], n_[1], np.nan, np.nan, 19.0]])  # a Gaia star without proper motion
    ep1 = gaia_at_epoch(gcat, 1954.2); ep2 = gaia_at_epoch(gcat, 1989.3)
    log("  injected: object %+.0f\" %+.0f\" (on the ellipse's long axis), impostor %+.0f\" %+.0f\", static %+.0f\" %+.0f\", "
        "no-PM Gaia star %+.0f\" %+.0f\", known asteroid %+.0f\" %+.0f\"" % (OBJ + IMP + STA + NOPM + KNO))
    img1 = render(h1, np.r_[ep1[:-1, 0], o_[0], i_[0], s0[0], n_[0], k_[0]], np.r_[ep1[:-1, 1], o_[1], i_[1], s0[1], n_[1], k_[1]],
                  np.r_[ep1[:-1, 2], 18.6, 18.6, 18.9, 19.0, 18.5], flaw=(60, 150))
    h2 = make_header(ra_c, dec_c, date="02/04/89", ut="15:10:00", exp=60.0); h2["PLATEID"] = "T2"
    h2["AMDX3"] = 3.5; h2["AMDY3"] = -2.1  # this plate's header solution is a few arcsec off
    # and it is a POSS-II plate (telescope not known for reporting): usable only as a comparison, dated by EPOCH
    h2["TELESCOP"] = "Oschin Schmidt - D"; h2["EPOCH"] = decyear(dt.datetime(1989, 4, 2, 15, 10))
    img2 = render(h2, np.r_[ep2[:-1, 0], s0[0], nb_[0]], np.r_[ep2[:-1, 1], s0[1], nb_[1]], np.r_[ep2[:-1, 2], 18.9, 17.6])
    h3 = make_header(ra_c, dec_c, date="15/03/54", ut="06:40:00", exp=12.0); h3["PLATEID"] = "T3"
    o3 = sky((OBJ[0] + 0.8, OBJ[1]))  # a slow mover: 0.8" between the partner plates
    img3 = render(h3, np.r_[ep1[:-1, 0], o3[0], s0[0]], np.r_[ep1[:-1, 1], o3[1], s0[1]], np.r_[ep1[:-1, 2], 18.6, 18.9])
    out = subject_paths(tmp, "selftest")
    fits.PrimaryHDU(img1, h1).writeto(os.path.join(out, "cutouts", "poss1_red_XP000_T1__target.fits"))
    fits.PrimaryHDU(img2, h2).writeto(os.path.join(out, "cutouts", "poss1_red_XP000_T1__cmp_poss2ukstu_red.fits"))
    fits.PrimaryHDU(img3, h3).writeto(os.path.join(out, "cutouts", "poss1_red_XP000_T1__cmp_poss1_blue.fits"))
    # the archive serves some plates under two survey names: the same other-date plate again, to be counted once
    fits.PrimaryHDU(img2, h2).writeto(os.path.join(out, "cutouts", "poss1_red_XP000_T1__cmp_phase2_gsc1.fits"))
    inf = plate_info(h1)
    check(inf["start"] == dt.datetime(1954, 3, 15, 6, 0, 0) and inf["mid"] == dt.datetime(1954, 3, 15, 6, 22, 30)
          and inf["site"] == "675" and inf["mpc"] == "261", "synthetic plate time and site")
    xx, yy = dss_sky2pix(h1, o_[0], o_[1]); rr, dd = dss_pix2sky(h1, xx, yy)
    check(math.hypot((float(rr[0]) - o_[0]) * 3600 * math.cos(math.radians(pdec)), (float(dd[0]) - o_[1]) * 3600) < 0.01,
          "plate solution inverse")
    check(box_on_cutout(h1, pra, pdec, 100.0) and not box_on_cutout(h1, pra, pdec, 300.0)
          and not box_on_cutout(h1, pra + 1.0, pdec, 10.0), "a search box inside, partly off and off the cutout")
    pl = dict(survey="poss1_red", plate="XP000_T1", pred_ra=pra, pred_dec=pdec, V=18.9,
              ra3=math.hypot(90 * ux, 3 * vx), dec3=math.hypot(90 * uy, 3 * vy), smaa=90.0, smia=3.0, theta=TH,
              rate=3.0, mid=inf["mid"].isoformat())
    known = lambda site, jd, ra, dec, rad: [kobj("selftest", OBJ[0] + 0.4, OBJ[1], 18.6, 1.0, cls="KBO"),  # noqa: E731
                                            kobj("Test", KNO[0], KNO[1], 18.5, 0.5, num="12345")]
    cfg = dict(designation="selftest", names=["selftest"], box_factor=1.5, nsig=4.0)
    res, cands = measure_one(cfg, out, pl, gaia_fn=lambda ra, dec, r: gcat, known_fn=known, show=True)
    log("  astrometry: %s refs, rms %s\"; in box %s, not rejected %s, within 3 sigma %s: %s" % (
        res.get("n_ref"), res.get("rms_arcsec"), res.get("n_in_box"), res.get("n_candidates"), res.get("n_within_3sigma"),
        res.get("result")))

    def at(s_, r=2.5):
        m_ = [c for c in cands if math.hypot(c["dra_arcsec"] - s_[0], c["ddec_arcsec"] - s_[1]) < r]
        return m_[0] if m_ else None
    for nm, s_ in (("object", OBJ), ("impostor", IMP), ("static", STA), ("no-PM star", NOPM), ("known asteroid", KNO)):
        c = at(s_)
        log("   %-14s %s" % (nm, "not detected" if c is None else "%+.2f\" %+.2f\"  %.1f sigma  snr %.1f  -> %s  %s" % (
            c["dra_arcsec"], c["ddec_arcsec"], c["dist_sigma"], c["snr"], c["rejected"] or "CANDIDATE",
            [t[2] for t in c["static_test"]])))
    c = at(OBJ)
    check(c is not None and not c["rejected"], "the object is kept (same-night partner and neighbor do not reject it)")
    check(res.get("result") == "unique candidate within 3 sigma" and c is not None and
          abs(res.get("best_dra", 99) - c["dra_arcsec"]) < 1e-6, "the object is the unique candidate within 3 sigma")
    if c is not None:
        err = math.hypot(c["dra_arcsec"] - OBJ[0], c["ddec_arcsec"] - OBJ[1])
        check(err <= 3.0 * math.hypot(c["sig_ra_arcsec"], c["sig_dec_arcsec"]),
              "object position %.2f\" from the truth (stated %.2f\", %.2f\")" % (err, c["sig_ra_arcsec"], c["sig_dec_arcsec"]))
    c = at(IMP)
    check(c is not None and c["dist_sigma"] > 3.0, "the impostor near the center is outside the ellipse (%.1f sigma)"
          % (c["dist_sigma"] if c else float("nan")))
    c = at(STA)
    check(c is not None and "static" in c["rejected"], "the faint static source is rejected by forced photometry")
    check(res.get("n_comparison_plates") == 1 and len(c["static_test"] if c else []) == 1
          and any("counted once" in s_ for s_ in res.get("not_used_as_comparison", [])),
          "a plate served under two survey names counts as one comparison (%s)" % res.get("comparison_dates"))
    c = at(NOPM)
    check(c is not None and "Gaia star" in c["rejected"], "the Gaia star without proper motion is rejected")
    c = at(KNO)
    check(c is not None and "known object" in c["rejected"], "the known asteroid is rejected")
    st_ = res.get("skybot_target")
    check(isinstance(st_, list) and len(st_) == 1 and st_[0]["name"] == "selftest" and res.get("known_objects_in_box")
          == ["12345 Test"], "SkyBoT's entry for the target is kept apart (%s)" % (st_,))
    prov = [c for c in cands if "Gaia star" not in c["rejected"] and "flaw" not in c["rejected"]
            and not (c.get("err_parts") and np.all(np.isfinite(c["err_parts"]["injected"][:2])))]
    check(not prov, "every source in the box gets plate-based error bars, however far (%d left with formal only)" % len(prov))
    fl = [c for c in cands if abs(c["x"] - 60) < 1.5 and abs(c["y"] - 150) < 1.5]
    check(bool(fl) and "flaw" in fl[0]["rejected"], "the plate flaw is found and rejected")
    pl2 = dict(pl, ra3=float("nan"), dec3=float("nan"), smaa=float("nan"), smia=float("nan"), theta=float("nan"))
    r2, _ = measure_one(cfg, out, pl2, gaia_fn=lambda ra, dec, r: gcat, known_fn=known, show=False)
    check(r2["status"].startswith("refused"), "no uncertainty from Horizons -> refused (%s)" % r2["status"])
    known_u = lambda site, jd, ra, dec, rad: known(site, jd, ra, dec, rad) + [kobj("2014 XY9", OBJ[0] + 20.0, OBJ[1], 18.7, 100.0)]  # noqa: E731
    ru, cu = measure_one(cfg, out, pl, gaia_fn=lambda ra, dec, r: gcat, known_fn=known_u, show=False)
    check(ru.get("result") == "unique candidate within 3 sigma" and "could be known object 2014 XY9" in ru.get("flag", "")
          and ru.get("reportable") is False, "a candidate that could be a known object with an uncertain place stays the "
          "candidate, flagged and not reportable")
    known_b = lambda site, jd, ra, dec, rad: known(site, jd, ra, dec, rad) + [kobj("Other", OBJ[0], OBJ[1], 18.6, 0.5, "23456")]  # noqa: E731
    rb_, _ = measure_one(cfg, out, pl, gaia_fn=lambda ra, dec, r: gcat, known_fn=known_b, show=False)
    check(rb_.get("result") == "no candidate within 3 sigma" and "rejected as a known object" in rb_.get("flag", "")
          and rb_.get("reportable") is False, "a known object rejected inside the prediction's 3-sigma region flags the plate, "
          "and a plate with no candidate is not reportable")
    known_c = lambda site, jd, ra, dec, rad: [kobj("Alias", OBJ[0] + 0.4, OBJ[1], 18.6, 1.0, "99999", cls="KBO"),  # noqa: E731
                                              kobj("Test", KNO[0], KNO[1], 18.5, 0.5, num="12345")]
    rc_, _ = measure_one(cfg, out, pl, gaia_fn=lambda ra, dec, r: gcat, known_fn=known_c, show=False)
    check(rc_.get("result") == "unique candidate within 3 sigma" and "under another name" in rc_.get("flag", "")
          and rc_.get("reportable") is False and [o_["name"] for o_ in rc_.get("known_possible_target", [])] == ["99999 Alias"],
          "the target listed by SkyBoT under a name the subject doesn't give is not rejected; the plate is flagged")
    check("no other-date plate" not in res.get("flag", "") and "runs off" not in res.get("flag", ""),
          "with a comparison plate and the box on the cutout, neither is flagged (%s)" % res.get("flag"))
    # ---- blends: a candidate 1.05 FWHM from a brighter fixed source
    # (a) the joint fit itself, on plain images with the real scans' 3.8 px FWHM: unbiased, where a one-source fit is
    #     pulled ~2 px toward the neighbor
    sg = 3.8 / 2.3548; sep = 1.05 * 3.8
    yy_, xx_ = np.mgrid[0:60, 0:60]
    d1, dj = [], []
    for _ in range(12):
        x0_, y0_ = 26.0 + rng.uniform(-0.5, 0.5), 30.0 + rng.uniform(-0.5, 0.5)
        w = ndimage.gaussian_filter(rng.normal(0, 1, (60, 60)), 0.74); w /= w.std()
        im = 40 * w + 415 * np.exp(-((xx_ - x0_) ** 2 + (yy_ - y0_) ** 2) / (2 * sg * sg)) \
            + 600 * np.exp(-((xx_ - x0_ - sep) ** 2 + (yy_ - y0_) ** 2) / (2 * sg * sg))
        p1_ = psf_fit(im, x0_, y0_, sg); pj_ = psf_fit_joint(im, x0_, y0_, [(x0_ + sep, y0_, 600.0)], sg)
        if p1_ is not None:
            d1.append(p1_[0] - x0_)
        if pj_ is not None:
            dj.append(pj_[0] - x0_)
    check(len(dj) >= 10 and abs(np.mean(dj)) < 0.25 and (not d1 or np.mean(d1) > 1.0),
          "joint fit of a source 1.05 FWHM from a brighter one: bias %+.2f px (one-source fit %s)" % (
              np.mean(dj) if dj else float("nan"), ("%+.2f px" % np.mean(d1)) if d1 else "fails"))
    # (b) the measuring step uses it: a candidate 6.5" from a brighter fixed neighbor is fitted jointly, its error
    #     includes the deblending, it stays within 3 sigma of the truth, and the neighbor is rejected as static
    OB2 = pick([(a_, b_) for a_ in np.arange(-70, 71, 7.0) for b_ in np.arange(-70, 71, 7.0)
                if clear(a_ + 6.5, b_, 12.0) and clear(a_, b_, 15.0)])
    o2 = sky(OB2); n2 = sky((OB2[0] + 6.5, OB2[1]))
    out4 = subject_paths(tmp, "selftest_blend")
    im4 = render(h1, np.r_[ep1[:-1, 0], o2[0], n2[0]], np.r_[ep1[:-1, 1], o2[1], n2[1]], np.r_[ep1[:-1, 2], 18.9, 18.5])
    im5 = render(h2, np.r_[ep2[:-1, 0], n2[0]], np.r_[ep2[:-1, 1], n2[1]], np.r_[ep2[:-1, 2], 18.5])
    fits.PrimaryHDU(im4, h1).writeto(os.path.join(out4, "cutouts", "poss1_red_XP000_T1__target.fits"))
    fits.PrimaryHDU(im5, h2).writeto(os.path.join(out4, "cutouts", "poss1_red_XP000_T1__cmp_poss2ukstu_red.fits"))
    plb = dict(pl, pred_ra=o2[0], pred_dec=o2[1], smaa=6.0, smia=6.0, theta=0.0, ra3=6.0, dec3=6.0)
    rb, cb = measure_one(dict(cfg, check_known=False), out4, plb, gaia_fn=lambda ra, dec, r: gcat[:-1], show=False)
    cj = [c for c in cb if math.hypot(c["dra_arcsec"], c["ddec_arcsec"]) < 3.0]
    nj = [c for c in cb if math.hypot(c["dra_arcsec"] - 6.5, c["ddec_arcsec"]) < 2.5]
    cj = cj[0] if cj else None; nj = nj[0] if nj else None
    ej = math.hypot(cj["dra_arcsec"], cj["ddec_arcsec"]) if cj else 99.0
    check(cj is not None and "blend" in cj and "deblend" in cj.get("err_parts", {}) and not cj["rejected"]
          and ej <= 3.0 * math.hypot(cj["sig_ra_arcsec"], cj["sig_dec_arcsec"]),
          "a candidate 6.5\" from a brighter neighbor is fitted jointly: %.2f\" from the truth (stated %.2f\", %.2f\"; %s)"
          % (ej, cj["sig_ra_arcsec"] if cj else float("nan"), cj["sig_dec_arcsec"] if cj else float("nan"),
             cj.get("blend") if cj else "not found"))
    check(nj is not None and "static" in nj["rejected"], "the blended neighbor is rejected as static")
    # ---- archive errors while planning must stop the run, not read as "no plates"
    out3 = subject_paths(tmp, "selftest_plan")
    g = globals(); saved = (g["horizons_track"], g["fetch_dss"])
    try:
        g["horizons_track"] = lambda *a_, **k_: [dict(t=dt.datetime(1954, 1, 1) + dt.timedelta(days=5 * j), ra=150.0 + 1.5 * j,
                                                   dec=10.0, elong=150.0) for j in range(6)]

        def boom(*a_, **k_):
            raise IOError("GET failed after 3 tries: (HTTP Error 500: Internal Server Error)")
        g["fetch_dss"] = boom
        stopped = False
        try:
            step_plan(dict(designation="selftest", horizons_command="x;", years=[1954, 1954], surveys=["poss1_red"]), out3)
        except PlanIncomplete:
            stopped = True
        st = open(os.path.join(out3, "plan_status.txt")).read() if os.path.exists(os.path.join(out3, "plan_status.txt")) else ""
        check(stopped and not os.path.exists(os.path.join(out3, "plan.json")) and st.startswith("INCOMPLETE"),
              "archive errors stop the plan (no plan.json, status INCOMPLETE) instead of reporting no plates")
    finally:
        g["horizons_track"], g["fetch_dss"] = saved
    # ---- an error at the check that the object is on a plate also stops the plan, and so does a Horizons failure
    out6 = subject_paths(tmp, "selftest_plan2")
    saved = (g["horizons_track"], g["fetch_dss"], g["horizons_at"])
    try:
        def h_fail(*a_, **k_):
            raise RuntimeError("GET failed after 3 tries: (HTTP Error 503)")
        g["horizons_track"] = h_fail
        try:
            step_plan(dict(designation="selftest", horizons_command="x;", years=[1954, 1954], surveys=["poss1_red"]), out6)
            h_stopped = False
        except PlanIncomplete:
            h_stopped = True
        check(h_stopped and open(os.path.join(out6, "plan_status.txt")).read().startswith("INCOMPLETE"),
              "a JPL Horizons failure stops the plan as incomplete")
        g["horizons_track"] = lambda *a_, **k_: [dict(t=dt.datetime(1954, 3, 10) + dt.timedelta(days=5 * j), ra=169.85,
                                                   dec=-10.95, elong=150.0) for j in range(2)]
        g["horizons_at"] = lambda *a_, **k_: dict(ra=pra, dec=pdec, dra=1.0, ddec=0.5, V=19.0, ra3=2.0, dec3=2.0, smaa=2.0,
                                                  smia=1.0, theta=0.0)
        n_calls = []

        def dss_then_error(*a_, **k_):
            n_calls.append(1)
            if len(n_calls) > 1:
                raise RuntimeError("DSS poss1_red: the archive answered with an error page, not an image, 3 times")
            return fits.PrimaryHDU(img1, h1)
        g["fetch_dss"] = dss_then_error
        stopped = False
        try:
            step_plan(dict(designation="selftest", horizons_command="x;", years=[1954, 1954], surveys=["poss1_red"]), out6)
        except PlanIncomplete:
            stopped = True
        st = open(os.path.join(out6, "plan_status.txt")).read() if os.path.exists(os.path.join(out6, "plan_status.txt")) else ""
        check(stopped and len(n_calls) == 2 and not os.path.exists(os.path.join(out6, "plan.json")) and st.startswith("INCOMPLETE"),
              "an archive error while checking that the object is on a plate stops the plan")
        g["fetch_dss"] = lambda *a_, **k_: fits.PrimaryHDU(img1, h1)
        with open(os.path.join(out6, "fetch_status.txt"), "w") as f:
            f.write("COMPLETE earlier\n")
        step_plan(dict(designation="selftest", horizons_command="x;", years=[1954, 1954], surveys=["poss1_red"]), out6)
        check(os.path.exists(os.path.join(out6, "plan.json")) and
              open(os.path.join(out6, "fetch_status.txt")).read().startswith("INCOMPLETE"),
              "a new plan needs a new fetch before measuring")
    finally:
        g["horizons_track"], g["fetch_dss"], g["horizons_at"] = saved
    # ---- fetching: the cutout holds the whole search box, a kept cutout that doesn't is fetched again, other-date
    # plates come from every survey, and an incomplete plan stops fetching
    out5 = subject_paths(tmp, "selftest_fetch")
    with open(os.path.join(out5, "plan.json"), "w") as f:
        json.dump([dict(survey="poss1_red", plate="XP000_T1", pred_ra=pra, pred_dec=pdec, ra3=2.0, dec3=2.0)], f)
    calls = []

    def fake_dss(sv, ra, dec, size, path=None):
        calls.append((sv, size))
        return fits.PrimaryHDU(img1, h1) if sv == "poss1_red" else None
    cf = dict(cutout_arcmin=10.0, box_factor=1.5, min_box_arcsec=400.0, surveys=["poss1_red", "poss1_blue"])
    saved = g["fetch_dss"]
    try:
        g["fetch_dss"] = fake_dss
        step_fetch(cf, out5); run1 = list(calls); del calls[:]
        step_fetch(cf, out5); run2 = list(calls); del calls[:]
        step_fetch(dict(cf, min_box_arcsec=20.0), out5); run3 = list(calls); del calls[:]
        step_fetch(dict(cf, min_box_arcsec=2000.0), out5); run4 = list(calls); del calls[:]
        def dss_err(sv, ra, dec, size, path=None):
            if sv == "quickv":
                raise RuntimeError("DSS quickv: the archive answered with an error page, not an image, 3 times")
            return fake_dss(sv, ra, dec, size, path)
        g["fetch_dss"] = dss_err
        os.remove(os.path.join(out5, "cutouts", "poss1_red_XP000_T1__target.fits"))
        try:
            step_fetch(cf, out5); f_stopped = False
        except FetchIncomplete:
            f_stopped = True
        fst = open(os.path.join(out5, "fetch_status.txt")).read()
        try:
            step_measure(cf, out5); m_stopped = False
        except FetchIncomplete:
            m_stopped = True
        g["fetch_dss"] = fake_dss
        step_fetch(cf, out5); del calls[:]
        fst2 = open(os.path.join(out5, "fetch_status.txt")).read()

        def dss_stop(sv, ra, dec, size, path=None):
            raise KeyboardInterrupt
        g["fetch_dss"] = dss_stop
        os.remove(os.path.join(out5, "cutouts", "poss1_red_XP000_T1__target.fits"))
        try:
            step_fetch(cf, out5)
        except KeyboardInterrupt:
            pass
        try:
            step_measure(cf, out5); i_stopped = False
        except FetchIncomplete:
            i_stopped = True
        g["fetch_dss"] = fake_dss
        step_fetch(cf, out5); del calls[:]
        with open(os.path.join(out5, "plan_status.txt"), "w") as f:
            f.write("INCOMPLETE test\n")
        try:
            step_fetch(cf, out5); stopped = False
        except PlanIncomplete:
            stopped = True
    finally:
        g["fetch_dss"] = saved
    check(f_stopped and fst.startswith("INCOMPLETE") and "quickv" in fst and m_stopped and fst2.startswith("COMPLETE"),
          "a failed download stops the fetch and the measuring until a later --fetch gets everything")
    check(i_stopped, "an interrupted fetch is not measured")
    check(run1 and run1[0][0] == "poss1_red" and run1[0][1] >= 2 * 400.0 / 60.0 + 4.0 - 1e-9,
          "the target cutout holds the whole search box (%.1f')" % (run1[0][1] if run1 else float("nan")))
    check(sorted(sv for sv, _ in run1[1:]) == sorted(SURVEYS), "other-date plates are asked from every survey (%s)"
          % [sv for sv, _ in run1[1:]])
    check(run2 and run2[0][0] == "poss1_red" and len(run2) == len(SURVEYS) + 1 and len(run3) == len(SURVEYS),
          "a kept cutout is fetched again only when it doesn't hold the search box")
    check(not run4 and stopped, "a box larger than the DSS limit is not fetched; an incomplete plan stops fetching")
    # ---------------- 4. stated error bars against the real scatter, with correlated noise
    pulls = []; errs = []
    picks = [(a_, b_) for a_ in np.arange(-84, 85, 21.0) for b_ in np.arange(-84, 85, 21.0)]
    picks = [p_ for p_ in picks if min(np.min(np.hypot(e1x - p_[0], e1y - p_[1])), 99) > 12.0][:24]
    out2 = subject_paths(tmp, "selftest_err")
    plw = dict(pl, smaa=400.0, smia=400.0, theta=0.0, ra3=400.0, dec3=400.0)
    for draw in range(3):
        mags = rng.uniform(18.4, 19.0, len(picks))
        pr = [sky(p_) for p_ in picks]
        im = render(h1, np.r_[ep1[:-1, 0], [q[0] for q in pr]], np.r_[ep1[:-1, 1], [q[1] for q in pr]], np.r_[ep1[:-1, 2], mags])
        for fn in os.listdir(os.path.join(out2, "cutouts")):
            os.remove(os.path.join(out2, "cutouts", fn))
        fits.PrimaryHDU(im, h1).writeto(os.path.join(out2, "cutouts", "poss1_red_XP000_T1__target.fits"))
        rw, cw = measure_one(dict(cfg, check_known=False), out2, plw, gaia_fn=lambda ra, dec, r: gcat[:-1], show=False)
        for p_ in picks:
            m_ = [c for c in cw if math.hypot(c["dra_arcsec"] - p_[0], c["ddec_arcsec"] - p_[1]) < 3.0 and not c["rejected"]]
            if not m_:
                continue
            c = m_[0]
            pulls += [(c["dra_arcsec"] - p_[0]) / c["sig_ra_arcsec"], (c["ddec_arcsec"] - p_[1]) / c["sig_dec_arcsec"]]
            errs.append(math.hypot(c["dra_arcsec"] - p_[0], c["ddec_arcsec"] - p_[1]))
    pulls = np.array(pulls)
    check("no other-date plate" in rw.get("flag", "") and "runs off" in rw.get("flag", ""),
          "no other-date plate and a box off the cutout are both flagged (%s)" % rw.get("flag"))
    if len(errs) < 0.8 * 3 * len(picks):
        check(False, "faint objects found: %d of %d" % (len(errs), 3 * len(picks)))
    else:
        ps = float(np.std(pulls)); pm = float(np.mean(pulls))
        check(0.7 < ps < 1.4 and abs(pm) < 0.3, "error bars (correlated noise): %d of %d faint objects found; median error "
              "%.2f\"; pulls mean %+.2f, sd %.2f (want 0, 0.7-1.4)" % (len(errs), 3 * len(picks), float(np.median(errs)), pm, ps))
    log("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return ok


# ------------------------------------------------------------------------------------------------- main

def object_subject(obj, years, surveys=None):
    """a subject for --object: '50000', '(50000) Quaoar' or '50000 Quaoar' (numbered), '2026 ED138', '2026ED138' or
    '2040 P-L' (designation), or a name ('Quaoar'). Returns (cfg, output folder name)."""
    s = re.sub(r"\s+", " ", str(obj)).strip()
    mp = re.match(r"^\((.*)\)$", s)
    if mp and _desig(mp.group(1)):  # '(2026 ED138)', as Horizons prints it
        s = mp.group(1).strip()
    des = _desig(s)
    m = re.match(r"^\(?(\d+)\)?(?: ([A-Za-z][A-Za-z' -]*))?$", s)  # a number, then only a name
    if des:  # checked first: '2026 ED138' and '2040 P-L' are designations, not number 2026 or 2040 with a name
        cfg = dict(designation=des, names=[des], horizons_command="DES=%s;" % des)
    elif m:
        names = [m.group(1)] + ([m.group(2)] if m.group(2) else [])
        cfg = dict(designation=" ".join(names), names=names, horizons_command="%s;" % m.group(1))
    elif re.match(r"^[A-Za-z][A-Za-z' -]{1,40}$", s):
        cfg = dict(designation=s, names=[s], horizons_command="%s;" % s)
    else:
        raise ValueError("--object: give a number, a provisional designation or a name (got %r)" % obj)
    y0, y1 = int(years[0]), int(years[1])
    if not (1889 < y0 <= y1 < 2060):
        raise ValueError("--years: two years, first <= last, between 1890 and 2059 (got %s %s)" % (years[0], years[1]))
    cfg.update(years=[y0, y1], box_factor=1.5, min_box_arcsec=60.0, cutout_arcmin=10.0,
               note="made by --object (precovery.py %s)" % VERSION)
    if surveys:
        bad = [v for v in surveys if v not in SURVEYS]
        if bad:
            raise ValueError("--surveys: unknown %s (known: %s)" % (", ".join(bad), ", ".join(SURVEYS)))
        cfg["surveys"] = list(surveys)
    return cfg, safe_name("%s_%d-%d" % (cfg["designation"].replace(" ", "_"), y0, y1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", nargs="?", help="a name in targets/precovery/ (without .json), or a path to a .json file")
    ap.add_argument("--object", metavar="NAME", help="search for this object instead of a subject file: a number "
                    "(50000), a provisional designation (2026 ED138) or a name (Quaoar); needs --years")
    ap.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"), help="with --object: the years to search")
    ap.add_argument("--surveys", nargs="+", metavar="SURVEY", help="with --object: the DSS surveys to search "
                    "(default: all of %s)" % ", ".join(SURVEYS))
    ap.add_argument("--plan", action="store_true", help="only find the plates")
    ap.add_argument("--fetch", action="store_true", help="only download the cutouts")
    ap.add_argument("--measure", action="store_true", help="only measure the downloaded cutouts")
    ap.add_argument("--self-test", action="store_true", help="run the offline self-test")
    ap.add_argument("--probe", nargs=2, type=float, metavar=("RA", "DEC"),
                    help="ask every survey which plate covers RA DEC (deg) and print its date; checks the DSS link")
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
                    help="folder that holds targets/ and output/ (default: the folder above tools/)")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    missing = [table_path(n) for n in HILL_TABLES if not os.path.exists(table_path(n))]
    if missing:
        sys.exit("D. Hill's POSS-I tables are missing (they ship with this tool in targets/precovery/): %s"
                 % ", ".join(os.path.normpath(m) for m in missing))
    if a.probe:
        for sv in SURVEYS:
            try:
                hdu = fetch_dss(sv, a.probe[0], a.probe[1], 1.0)
                if hdu is None:
                    log("  %-16s no plate here (the archive says this survey has none)" % sv)
                    continue
                hh = hdu.header
                raw = "  ".join("%s=%r" % (k, hh.get(k)) for k in ("DATE-OBS", "UT", "EXPOSURE", "EPOCH", "PLTLABEL",
                                                                     "REGION", "PLATEID", "BANDPASS", "TELESCOP"))
                log("  %-16s raw: %s" % (sv, raw))
                inf = plate_info(hh)
                log("  %-16s plate %-14s %s  exp %5.1f min  %-28s site %s mpc %s %s" % (
                    sv, inf["plate"], inf["start"].isoformat(), inf["exp_min"], inf["telescope"], inf["site"],
                    inf["mpc"], inf["flag"]))
            except Exception as e:  # noqa
                log("  %-16s error: %s" % (sv, e))
            time.sleep(0.5)
        return
    if a.object:
        if a.subject:
            ap.error("give a subject or --object, not both")
        if not a.years:
            ap.error("--object needs --years FIRST LAST")
        try:
            cfg, name = object_subject(a.object, a.years, a.surveys)
        except ValueError as e:
            ap.error(str(e))
        out = subject_paths(a.root, name)
        if a.plan or not (a.fetch or a.measure):  # what was planned: give this file to rerun it, or edit it
            write_json(os.path.join(out, name + ".json"), cfg)
    else:
        if not a.subject:
            ap.error("give a subject, or --object NAME --years FIRST LAST")
        if a.years or a.surveys:
            ap.error("--years and --surveys go with --object; a subject file sets them itself")
        sf = a.subject if a.subject.endswith(".json") else os.path.join(a.root, "targets", "precovery", a.subject + ".json")
        cfg = json.load(open(sf))
        name = os.path.splitext(os.path.basename(sf))[0]
        out = subject_paths(a.root, name)
    log("precovery.py %s  subject %s  -> %s" % (VERSION, cfg["designation"], out))
    every = not (a.plan or a.fetch or a.measure)
    if a.plan or every:
        try:
            step_plan(cfg, out)
        except PlanIncomplete:
            sys.exit(3)
        except SubjectMismatch as e:
            log("STOPPED: %s" % e)
            sys.exit(2)
    try:
        if a.fetch or every:
            step_fetch(cfg, out)
        if a.measure or every:
            step_measure(cfg, out)
    except PlanIncomplete as e:
        log("STOPPED: %s" % e)
        sys.exit(3)


if __name__ == "__main__":
    main()
