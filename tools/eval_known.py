#!/usr/bin/env python3
"""Score a known-answer run against the pass criteria in known_answers/README.md.
usage: python3 tools/eval_known.py output/precovery/<subject> [--ukst]   (--ukst: the UK Schmidt verdict rules)
Reads plan.json and results.json, and prints one block per plate and the verdict."""
import json, math, sys, os

d = sys.argv[1] if len(sys.argv) > 1 else "output/precovery/quaoar_1954"
UKST = "--ukst" in sys.argv
conclusive = []; mid_better = []; pulls7 = []
plan = {"%s_%s" % (p["survey"], p["plate"]): p for p in json.load(open(os.path.join(d, "plan.json")))}
res = json.load(open(os.path.join(d, "results.json")))
rows = res["plates"]; cands = res["candidates"]
print("tool version %s, %d plates in plan, %d measured" % (res.get("version"), len(plan), len(rows)))
c1 = len(plan) > 0
fails = []; inconclusive = []; recs = []
for r in rows:
    tag = r["plate"]; p = plan.get(tag, {})
    print("\n%s  mid %s  exp %s min  V %.2f  status %s" % (tag, r.get("mid"), r.get("exp_min"), r.get("V", float("nan")), r.get("status")))
    if r.get("status") != "ok":
        fails.append(tag + ": not measured (%s)" % r.get("status")); continue
    cl = [c for c in cands.get(tag, []) if not c["rejected"]]
    # the target = the source nearest the prediction (the known orbit is good to about 1.5" or better)
    allc = sorted(cands.get(tag, []), key=lambda c: math.hypot(c["dra_arcsec"], c["ddec_arcsec"]))
    near = allc[0] if allc else None
    clear = near is not None and near["gaia_star_within"] >= 5.0
    if near is None or math.hypot(near["dra_arcsec"], near["ddec_arcsec"]) > 5.0:
        print("  nothing detected within 5\" of the prediction")
        fails.append(tag + ": target not detected"); continue
    print("  nearest source: %+.2f\" %+.2f\"  snr %.1f  mag %.2f  rejected=%r  gaia star within %.1f\"" % (
        near["dra_arcsec"], near["ddec_arcsec"], near["snr"], near["mag_g_scale"], near["rejected"], near["gaia_star_within"]))
    if not clear:
        print("  not clear sky (a Gaia star within 5\"): criteria 2-3 not applied"); continue
    # criterion 2: top non-rejected candidate
    top = sorted(cl, key=lambda c: c["dist_sigma"])
    unique = r.get("result") == "unique candidate within 3 sigma" and abs(r.get("best_dra", 99) - near["dra_arcsec"]) < 1e-6
    ok2 = unique or (bool(top) and top[0] is near)
    print("  C2 result field: %r; target unique within 3 sigma: %s" % (r.get("result"), unique))
    if not ok2:
        fails.append(tag + ": target is not the top candidate (rejected=%r)" % near["rejected"])
    # criterion 3
    so_r, so_d = near["sig_ra_arcsec"], near["sig_dec_arcsec"]
    sj_r, sj_d = r["ra3"] / 3.0, r["dec3"] / 3.0
    lr, ld = 3 * math.hypot(so_r, sj_r), 3 * math.hypot(so_d, sj_d)
    tot = math.hypot(near["dra_arcsec"], near["ddec_arcsec"])
    ok3 = abs(near["dra_arcsec"]) <= lr and abs(near["ddec_arcsec"]) <= ld and tot <= 2.0
    print("  C2 top candidate: %s   C3 residual RA %+.2f (limit %.2f), Dec %+.2f (limit %.2f), total %.2f (limit 2.0): %s" % (
        ok2, near["dra_arcsec"], lr, near["ddec_arcsec"], ld, tot, "pass" if ok3 else "FAIL"))
    if not ok3:
        fails.append(tag + ": position residual outside tolerance")
    # criterion 4: timing convention
    vr, vd = p.get("dra_rate", float("nan")), p.get("ddec_rate", float("nan"))
    v = math.hypot(vr, vd)
    half_h = r["exp_min"] / 120.0
    if not (math.isfinite(v) and v > 0):
        print("  C4 no motion rate for this plate in plan.json: timing not tested")
        pulls7 += [near["dra_arcsec"] / math.hypot(so_r, sj_r), near["ddec_arcsec"] / math.hypot(so_d, sj_d)]
        continue
    ur, ud = vr / v, vd / v
    along = near["dra_arcsec"] * ur + near["ddec_arcsec"] * ud
    tool_mid = "MID-exposure" in str(r.get("time_note", ""))  # the tool read the header time as mid-exposure
    # the other reading: header = mid (if the tool used start), or header = start (if the tool used mid)
    along_alt = along - v * half_h if tool_mid else along + v * half_h
    lab_tool, lab_alt = ("UT=mid", "UT=start") if tool_mid else ("UT=start", "UT=mid")
    # JPL's error along the track from its full 3-sigma ellipse (SMAA, SMIA, Theta from +RA toward +Dec), else the
    # RA/Dec marginals
    sm, sn, th = r.get("smaa"), r.get("smia"), r.get("theta")
    if all(isinstance(q, (int, float)) and math.isfinite(q) for q in (sm, sn, th)):
        c_ = math.cos(math.radians(th)) * ur + math.sin(math.radians(th)) * ud  # cosine to the ellipse's long axis
        sj_along = math.sqrt((sm / 3.0 * c_) ** 2 + (sn / 3.0) ** 2 * max(0.0, 1.0 - c_ * c_))
    else:
        sj_along = math.hypot(sj_r * ur, sj_d * ud)
    sig_along = math.hypot(math.hypot(so_r * ur, so_d * ud), sj_along)
    shift = v * half_h
    if shift < 2 * sig_along:
        verdict4 = "inconclusive (shift %.2f\" < 2 sigma %.2f\")" % (shift, 2 * sig_along)
    elif abs(along) < abs(along_alt):
        verdict4 = "the tool's reading (%s) confirmed" % lab_tool; conclusive.append(tag)
    else:
        verdict4 = "the other reading (%s) fits better: FAIL" % lab_alt; conclusive.append(tag); mid_better.append(tag)
        fails.append(tag + ": timing convention")
    pulls7 += [near["dra_arcsec"] / math.hypot(so_r, sj_r), near["ddec_arcsec"] / math.hypot(so_d, sj_d)]
    print("  C4 along-track residual: %s (tool) %+.2f\", %s %+.2f\" (half-exposure shift %.2f\", sigma %.2f\"): %s" % (
        lab_tool, along, lab_alt, along_alt, shift, sig_along, verdict4))
    nfalse = len([c for c in cl if c is not near])
    print("  C6 other non-rejected candidates in the box: %d" % nfalse)
    recs.append(dict(tag=tag, mid=r["mid"], ra=near["ra"], dec=near["dec"], along=along, along_alt=along_alt, v=v,
                     vr=vr, vd=vd, nfalse=nfalse))
# criterion 5: same night (red and blue plates within 1 day)
import datetime as dt
for i in range(len(recs)):
    for j in range(i + 1, len(recs)):
        a, b = recs[i], recs[j]
        ta = dt.datetime.fromisoformat(a["mid"]); tb = dt.datetime.fromisoformat(b["mid"])
        h = (tb - ta).total_seconds() / 3600.0
        if abs(h) > 24:
            continue
        cd = math.cos(math.radians(a["dec"]))
        mr = ((b["ra"] - a["ra"] + 180.0) % 360.0 - 180.0) * 3600 * cd; md = (b["dec"] - a["dec"]) * 3600
        er = mr - a["vr"] * h; ed = md - a["vd"] * h
        ok = math.hypot(er, ed) <= 2.0
        print("\nC5 same night %s / %s: %.2f h apart; moved %+.2f\" %+.2f\", predicted %+.2f\" %+.2f\"; difference %.2f\" (limit 2.0): %s"
              % (a["tag"], b["tag"], h, mr, md, a["vr"] * h, a["vd"] * h, math.hypot(er, ed), "pass" if ok else "FAIL"))
        if not ok:
            fails.append("same-night pair %s/%s" % (a["tag"], b["tag"]))
print("\nC1 plates found: %s" % c1)
if pulls7:
    print("C7 residual / stated error, all plates and both coordinates: n %d, rms %.2f, max |%.2f|" % (
        len(pulls7), math.sqrt(sum(x * x for x in pulls7) / len(pulls7)), max(abs(x) for x in pulls7)))
timing = "pass" if (conclusive and not mid_better) else ("FAIL" if mid_better else "inconclusive")
print("C4 overall: %s (conclusive plates: %d)" % (timing, len(conclusive)))
if not c1:
    print("VERDICT: INCONCLUSIVE (no plates)")
elif fails:
    print("VERDICT: FAIL\n  " + "\n  ".join(fails))
elif not recs:
    print("VERDICT: INCONCLUSIVE (no clear-sky plate)")
elif UKST and timing == "inconclusive":
    print("VERDICT: POSITIONS PASS, TIMING INCONCLUSIVE on %d clear-sky plates (UK Schmidt positions stay unreportable)" % len(recs))
else:
    print("VERDICT: PASS on %d clear-sky plates" % len(recs))
