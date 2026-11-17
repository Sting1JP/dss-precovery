"""mpc_report.py - the report file for the Minor Planet Center (ADES PSV) for the positions precovery.py marks reportable.

The names in the header blocks come from a settings file (see ades_settings.example.json). The file never has a
`prog` field: the MPC adds the program code itself. Only nights with at least two reportable positions of the object
are written; a single position is held back and listed. When the IAU's ADES tools (pip package iau-ades) are
installed, the file is converted to XML and checked against the MPC's submission schema.
"""
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap

COLUMNS = ["permID", "provID", "mode", "stn", "obsTime", "ra", "dec", "rmsRA", "rmsDec", "mag", "rmsMag", "band",
           "photCat", "rmsTime", "astCat", "remarks"]
WITH_CORR = COLUMNS[:9] + ["rmsCorr"] + COLUMNS[9:]  # when a position comes with an RA-Dec correlation
OBSERVATORY_NAMES = {"261": "Palomar Mountain-DSS"}
SETTINGS_KEYS = {"submitter", "measurers", "telescope", "observatory_names", "comment_lines"}
TELESCOPE_KEYS = ("name", "aperture", "design", "detector")
MAX_COMMENT = 100   # characters in one comment line's text
WRAP = 93           # generated comment text is wrapped to this, so a whole line stays within 100 characters
NIGHT_START_UT_H = {"261": 20.0}  # local noon (Palomar, UTC-8) in UT: a night runs from one local noon to the next
PLACEHOLDER = "A. N. Observer"   # the name in ades_settings.example.json
BLUE_MIN_RMSTIME_S = 60.0
_DESIG = re.compile(r"^(?:(1[89]\d\d|20\d\d) ?([A-Z]{2}\d*)|(\d{4}) (P-L|T-[123]))$", re.I)


class SettingsError(ValueError):
    """the settings file is missing something, or holds something a report must not carry"""


def _walk(x, path="settings"):
    if isinstance(x, dict):
        for k, v in x.items():
            yield path + "." + str(k), k, v
            yield from _walk(v, path + "." + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield "%s[%d]" % (path, i), None, v
            yield from _walk(v, "%s[%d]" % (path, i))


def check_settings(s):
    """refuses a settings dict with unknown keys, missing names, a 'prog' key, an email address, or a comment line
    longer than 100 characters; returns it"""
    if not isinstance(s, dict):
        raise SettingsError("the settings file must hold one JSON object")
    for where, key, val in _walk(s):
        if key is not None and str(key).strip().lower() == "prog":
            raise SettingsError("%s: no 'prog' in a report; the MPC applies the program code itself" % where)
        if isinstance(val, str) and "@" in val:
            raise SettingsError("%s: no email address in a report (%r)" % (where, val))
        if isinstance(val, str) and (re.search(r"[\x00-\x1f\x7f|]", val) or val != val.strip()):
            raise SettingsError("%s: no line breaks, control characters, '|' or leading or trailing spaces (%r)" % (where, val))
        if isinstance(val, str) and val.strip() == PLACEHOLDER:
            raise SettingsError("%s: still the example's placeholder name; put the real names in the settings file" % where)
    unknown = sorted(set(s) - SETTINGS_KEYS)
    if unknown:
        raise SettingsError("unknown settings: %s (known: %s)" % (", ".join(unknown), ", ".join(sorted(SETTINGS_KEYS))))
    if not (isinstance(s.get("submitter"), str) and s["submitter"].strip()):
        raise SettingsError("'submitter' must be a name")
    m = s.get("measurers")
    if not (isinstance(m, list) and m and all(isinstance(v, str) and v.strip() for v in m)):
        raise SettingsError("'measurers' must be a list of names")
    t = s.get("telescope")
    if not (isinstance(t, dict) and set(t) == set(TELESCOPE_KEYS) and all(str(t[k]).strip() for k in TELESCOPE_KEYS)):
        raise SettingsError("'telescope' must give exactly %s" % ", ".join(TELESCOPE_KEYS))
    on = s.get("observatory_names", {})
    if not (isinstance(on, dict) and all(isinstance(v, str) and v.strip() for v in on.values())):
        raise SettingsError("'observatory_names' must map MPC codes to names")
    cl = s.get("comment_lines", [])
    if not (isinstance(cl, list) and all(isinstance(v, str) for v in cl)):
        raise SettingsError("'comment_lines' must be a list of lines")
    bad = [v for v in cl if len(v) > MAX_COMMENT or not v.strip()]
    if bad:
        raise SettingsError("comment lines must be 1 to %d characters: %r" % (MAX_COMMENT, bad[0][:60]))
    return s


def read_settings(path):
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError) as e:
        raise SettingsError("cannot read the settings file %s (%s)" % (path, e))
    return check_settings(s)


def object_ids(target_name):
    """(permID, provID, display name) from JPL Horizons' target name, e.g. '120178 (2003 OP32)',
    '50000 Quaoar (2002 LM60)' or '(2026 ED138)'"""
    s = re.sub(r"\s+", " ", str(target_name or "")).strip()
    perm = prov = name = ""
    for part in re.findall(r"\(([^)]*)\)", s):
        m = _DESIG.match(part.strip())
        if m:
            prov = ("%s %s" % (m.group(1), m.group(2)) if m.group(1) else "%s %s" % (m.group(3), m.group(4))).upper()
    rest = re.sub(r"\([^)]*\)", " ", s).strip()
    m2 = _DESIG.match(rest)
    m = re.match(r"^(\d+)(?: (.+))?$", rest)
    if m2:  # a bare designation such as '2003 OP32' or '2040 P-L' is not a number with a name
        if not prov:
            prov = ("%s %s" % (m2.group(1), m2.group(2)) if m2.group(1) else "%s %s" % (m2.group(3), m2.group(4))).upper()
    elif m:
        perm, name = m.group(1), (m.group(2) or "").strip()
    if perm:
        display = "(%s) %s" % (perm, name or prov) if (name or prov) else perm
    else:
        display = prov
    return perm, prov, display


def obs_time(mid_iso):
    """mid-exposure ISO time -> ADES obsTime, to the nearest second"""
    t = dt.datetime.fromisoformat(str(mid_iso)[:26])
    t = (t + dt.timedelta(microseconds=500000)).replace(microsecond=0)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _up10(seconds):
    return int(math.ceil(seconds / 10.0 - 1e-9) * 10)


def rms_time(r):
    """rmsTime (s): the plate clock's uncertainty (clock_sigma_min), rounded up to 10 s. A plate that takes its red
    partner's clock (clock_from_partner; by default every blue plate), which may not hold for it, gets the larger of
    60 s, that uncertainty, and the correction itself."""
    sig = r.get("clock_sigma_min")
    if not _finite(sig) or sig <= 0:
        return None
    t = _up10(sig * 60.0)
    partner = r.get("clock_from_partner")
    if partner is None:
        partner = str(r.get("region", "")).startswith("XO")
    if partner:
        tau = r.get("clock_min")
        tau = abs(tau) * 60.0 if isinstance(tau, (int, float)) and math.isfinite(tau) else 0.0
        t = max(int(BLUE_MIN_RMSTIME_S), t, _up10(tau))
    return t


def _fmt_exp(x):
    return ("%d" % int(round(x))) if abs(x - round(x)) < 1e-6 else ("%.1f" % x)


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def plate_row(r, perm, prov):
    """one report row (dict of column -> text) from a measured plate (a row of results.json). Read: mpc, mid (the
    time of the position: mid-exposure), best_ra, best_dec, best_sig_ra (RA cos Dec), best_sig_dec, best_rms_corr
    (optional), best_band, best_band_mag, best_rms_mag, region, label, emulsion, exp_min, clock_sigma_min, clock_min,
    clock_from_partner. Positions and their errors must not include the clock: it goes into rmsTime."""
    mag = r.get("best_band_mag")
    has_mag = row_has_mag(r)
    region = str(r.get("region", ""))
    emul = str(r.get("emulsion", "") or ("103aE" if region.startswith("XE") else "103aO"))
    rt = rms_time(r)
    return dict(permID=perm, provID=prov, mode="PHO", stn=str(r["mpc"]), obsTime=obs_time(r["mid"]),
                ra="%.6f" % r["best_ra"], dec="%.6f" % r["best_dec"], rmsRA="%.2f" % r["best_sig_ra"],
                rmsDec="%.2f" % r["best_sig_dec"],
                rmsCorr=("%.3f" % r["best_rms_corr"]) if _finite(r.get("best_rms_corr")) else "",
                mag=("%.1f" % mag) if has_mag else "",
                rmsMag=("%.1f" % r["best_rms_mag"]) if has_mag else "", band=r.get("best_band", "") if has_mag else "",
                photCat="Gaia3" if has_mag else "", rmsTime=str(rt) if rt is not None else "", astCat="Gaia3",
                remarks="POSS-I %s (DSS %s), %s, %s min" % (r.get("label", ""), region, emul, _fmt_exp(r["exp_min"])))


def select_rows(plates, require_pair=False):
    """(rows to write, held back): the reportable plates with one candidate, grouped by night; a night with a single
    position is held back, and with require_pair a night without both a red (XE) and a blue (XO) plate too.
    held_back entries are (plate dict, reason)."""
    rep = [r for r in plates if r.get("status") == "ok" and r.get("reportable") is True and "best_ra" in r]
    rep.sort(key=lambda r: r["mid"])
    by_night = {}
    for r in rep:
        start = NIGHT_START_UT_H.get(str(r.get("mpc")), 12.0)
        night = (dt.datetime.fromisoformat(r["mid"][:26]) - dt.timedelta(hours=start)).date()
        by_night.setdefault((str(r.get("mpc")), night), []).append(r)
    nights = [by_night[k] for k in sorted(by_night, key=lambda k: by_night[k][0]["mid"])]
    keep, held = [], []
    for n in nights:
        regions = set(str(r.get("region", ""))[:2] for r in n)
        if len(n) < 2:
            held += [(n[0], "a single position that night")]
        elif require_pair and not {"XE", "XO"} <= regions:
            held += [(r, "no red and blue pair that night") for r in n]
        else:
            keep += n
    return keep, held


def _wrap(text):
    return textwrap.wrap(text, WRAP, break_long_words=False, break_on_hyphens=False)


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def comment_lines(rows, display, settings):
    """the comment block: what was measured and how, wrapped to lines of 100 characters or fewer"""
    labels = [r.get("label", "") for r in rows]
    regions = [r.get("region", "") for r in rows]
    text = ["Precovery of %s on POSS-I plates %s." % (display, _join(labels)),
            "Measured on Digitized Sky Survey scans (STScI), DSS plates %s." % _join(regions),
            "Each plate refit to Gaia DR3 stars propagated to the plate epoch with proper motions.",
            "obsTime = mid-exposure. Start read as hours and hundredths; UT date from Hill (2026)."]
    for r in rows:
        tau = r.get("clock_min")
        if not _finite(tau):
            continue
        src = str(r.get("clock_source", ""))
        partner = r.get("clock_from_partner")
        if partner is None:
            partner = str(r.get("region", "")).startswith("XO")
        what = "the survey median of Hill (2026)" if "median" in src else "the per-plate value of Hill (2026)"
        if partner:
            text.append("%s: clock %+.2f min, taken from its red partner plate (%s)." % (r.get("label"), tau, what))
        else:
            text.append("%s: clock %+.2f min (%s)." % (r.get("label"), tau, what))
    for r in rows:
        nb = r.get("best_blend")
        if nb:
            text.append("%s: fitted jointly with a source %s\" away that blends with the image." % (
                r.get("label"), "/".join("%.1f" % v for v in nb)))
    text.append("rmsRA (RA*cos Dec) and rmsDec from the plate solution and artificial images placed on the plate; "
                "rmsTime from the plate clock.")
    if any(row_has_mag(r) for r in rows):
        text.append("mag: fitted peak, calibrated on isolated Gaia DR3 stars within 1 mag on the same plate, with G "
                    "and BP-RP converted to Cousins R (red plates) and Johnson B (blue plates) with the relations of "
                    "the Gaia DR3 documentation; no color term.")
    out = []
    for t in text:
        out += _wrap(t)
    out += list(settings.get("comment_lines", []))
    return out


def row_has_mag(r):
    """a magnitude and its rmsMag, both finite"""
    ok = lambda v: isinstance(v, (int, float)) and math.isfinite(v)  # noqa: E731
    return ok(r.get("best_band_mag")) and ok(r.get("best_rms_mag")) and bool(r.get("best_band"))


def format_psv(header, rows):
    """header lines and rows (dicts) -> the file's text; columns padded to a common width, the last one not. The
    rmsCorr column is there only if a row has a value for it."""
    cols = WITH_CORR if any(r.get("rmsCorr") for r in rows) else COLUMNS
    w = {c: max([len(c)] + [len(r[c]) for r in rows]) for c in cols}
    line = lambda vals: "|".join(v.ljust(w[c]) if c != cols[-1] else v for c, v in zip(cols, vals))  # noqa: E731
    out = list(header) + [line(cols)] + [line([r[c] for c in cols]) for r in rows]
    return "\n".join(out) + "\n"


def header_lines(stn, settings, version, with_mag, comments):
    names = dict(OBSERVATORY_NAMES); names.update(settings.get("observatory_names", {}))
    if stn not in names:
        raise SettingsError("no observatory name for MPC code %s (add it to 'observatory_names')" % stn)
    t = settings["telescope"]
    h = ["# version=2022", "# observatory", "! mpcCode %s" % stn, "! name %s" % names[stn],
         "# submitter", "! name %s" % settings["submitter"], "# measurers"]
    h += ["! name %s" % m for m in settings["measurers"]]
    h += ["# telescope"] + ["! %s %s" % (k, t[k]) for k in TELESCOPE_KEYS]
    h += ["# software", "! astrometry dss-precovery %s" % version]
    if with_mag:
        h.append("! photometry dss-precovery %s" % version)
    if comments:
        h += ["# comment"] + ["! line %s" % c for c in comments]
    return h


def read_psv(path):
    """(header lines, rows as dicts) from a PSV file"""
    head, rows, cols = [], [], None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            s = raw.rstrip("\n")
            if not s.strip():
                continue
            if s.startswith("#") or s.startswith("!"):
                head.append(s); continue
            vals = [v.strip() for v in s.split("|")]
            if cols is None:
                cols = vals; continue
            rows.append(dict(zip(cols, vals)))
    return head, rows


def validate(path):
    """the IAU's ADES tools, if installed: PSV to XML, then the MPC's submission schema.
    Returns ('ok' | 'failed' | 'not installed', text)."""
    import importlib.util
    try:
        found = importlib.util.find_spec("ades.psvtoxml") is not None and importlib.util.find_spec("ades.valsubmit") is not None
    except ImportError:
        found = False
    if not found:
        return "not installed", "iau-ades is not installed (pip install iau-ades), so the file was not validated"
    path = os.path.abspath(path)
    run = lambda args, td: subprocess.run([sys.executable, "-m"] + args, cwd=td, capture_output=True, text=True,  # noqa: E731
                                          encoding="utf-8", errors="replace", timeout=300)
    with tempfile.TemporaryDirectory() as td:
        xml = os.path.join(td, "report.xml")
        try:
            p = run(["ades.psvtoxml", path, xml], td)
            if p.returncode != 0 or not os.path.exists(xml):
                return "failed", ("psvtoxml: " + (p.stdout + p.stderr).strip())[-2000:]
            p = run(["ades.valsubmit", xml], td)
        except (subprocess.TimeoutExpired, OSError) as e:
            return "failed", "the validator did not finish (%s)" % e
        vf = os.path.join(td, "valsubmit.file")
        verdict = open(vf, encoding="utf-8", errors="replace").read().strip() if os.path.exists(vf) else ""
        detail = (p.stdout + p.stderr).strip()
        if verdict.startswith("submit is OK"):
            return "ok", verdict
        return "failed", ((verdict or "no verdict from valsubmit") + ("\n" + detail if detail else ""))[-3000:]


def write_report(plates, target_name, settings, version, path, require_pair=False, comments=None):
    """writes the report for the plates (rows of results.json) to path, if any night has two or more reportable
    positions (with require_pair: a red and a blue one). comments: the comment lines to use instead of the generated
    ones (which describe the star-image method); the settings' own lines follow either way. Returns dict(path or None,
    written: [plate], held_back: [(plate, reason)], no_mag: [plate], validation: (status, text))."""
    check_settings(settings)
    perm, prov, display = object_ids(target_name)
    keep, held = select_rows(plates, require_pair)
    if keep and not (perm or prov):
        held += [(r, "no number or designation for the object") for r in keep]
        keep = []
    stns = sorted(set(str(r["mpc"]) for r in keep))
    if len(stns) > 1:
        raise ValueError("positions from more than one observatory code (%s) in one report" % ", ".join(stns))
    out = dict(path=None, invalid_path=None, written=[r["plate"] for r in keep],
               held_back=[(r["plate"], why) for r, why in held], validation=("not run", ""),
               no_mag=[r["plate"] for r in keep if not row_has_mag(r)])
    if not keep:
        return out
    rows = [plate_row(r, perm, prov) for r in keep]
    lines = comment_lines(keep, display, settings) if comments is None else list(comments) + list(settings.get("comment_lines", []))
    head = header_lines(stns[0], settings, version, any(row_has_mag(r) for r in keep), lines)
    for h in head:
        if h.startswith("! line ") and len(h) - 7 > MAX_COMMENT:
            raise ValueError("comment line longer than %d characters: %r" % (MAX_COMMENT, h[:60]))
    text = format_psv(head, rows)
    if re.search(r"(^|\|)\s*prog\s*(\||$)", text, re.M) or "@" in text:
        raise ValueError("the report would carry a prog field or an email address; not written")
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    out["validation"] = validate(path + ".part")
    if out["validation"][0] == "failed":  # kept for inspection under a name that can't be taken for the report
        bad = re.sub(r"\.psv$", "", path) + ".invalid.psv"
        os.replace(path + ".part", bad)
        out["invalid_path"] = bad
        out["written"] = []
        return out
    os.replace(path + ".part", path)
    out["path"] = path
    return out
