#!/usr/bin/env python3
"""Sample ~320 InsectSound1000 clips/species with recording-id hold-out and download missing WAVs."""
import json, os, zlib, struct, time, urllib.request, http.cookiejar
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

INDEX = "/workspace/insect-sound/cd/index.jsonl"
OUT = "/workspace/insect-sound/raw"
MANIFEST = "/workspace/insect-sound/manifest_10x.json"
URL = "https://www.openagrar.de/servlets/MCRFileNodeServlet/openagrar_derivate_00055778/InsectSound1000.zip"
TARGET = 640
SPECIES = [
    "Aphidoletes_aphidimyza", "Bombus_terrestris", "Bradysia_difformis",
    "Coccinella_septempunctata", "Episyrphus_balteatus", "Halyomorpha_halys",
    "Myzus_persicae", "Nezara_viridula", "Palomena_prasina",
    "Rhaphigaster_nebulos", "Trialeurodes_vaporariorum", "Tuta_absoluta",
]

def load_index():
    by = defaultdict(lambda: defaultdict(list))
    with open(INDEX) as f:
        for line in f:
            rec = json.loads(line)
            base = os.path.basename(rec["n"])
            sp = next(s for s in SPECIES if f"_{s}_" in base)
            rid = base.split(f"_{sp}_")[0]
            rec = dict(rec)
            rec["species"] = sp
            rec["base"] = base
            rec["recording_id"] = rid
            by[sp][rid].append(rec)
    return by

def largest_remainder(weights, total, caps):
    """Integer allocation summing to total, each <= cap, proportional to weights."""
    keys = list(weights)
    wsum = sum(weights[k] for k in keys)
    if wsum <= 0 or total <= 0:
        return {k: 0 for k in keys}
    raw = {k: total * weights[k] / wsum for k in keys}
    alloc = {k: min(caps[k], int(raw[k])) for k in keys}
    # distribute remainder by fractional part, respecting caps
    used = sum(alloc.values())
    order = sorted(keys, key=lambda k: (raw[k] - int(raw[k])), reverse=True)
    guard = 0
    while used < total and guard < total * 3:
        guard += 1
        progressed = False
        for k in order:
            if alloc[k] < caps[k] and used < total:
                alloc[k] += 1
                used += 1
                progressed = True
            if used >= total:
                break
        if not progressed:
            break
    return alloc

def spaced(recs, n):
    recs = sorted(recs, key=lambda r: r["base"])
    if n >= len(recs):
        return recs
    if n == 1:
        return [recs[len(recs) // 2]]
    idxs = []
    seen = set()
    for i in range(n):
        j = round(i * (len(recs) - 1) / (n - 1))
        if j not in seen:
            seen.add(j)
            idxs.append(j)
    # if rounding collapsed, fill
    if len(idxs) < n:
        for j in range(len(recs)):
            if j not in seen:
                seen.add(j)
                idxs.append(j)
            if len(idxs) >= n:
                break
    return [recs[j] for j in idxs[:n]]

def build():
    by = load_index()
    manifest = []
    summary = {}
    for sp in SPECIES:
        sessions = {rid: recs for rid, recs in by[sp].items()}
        sizes = {rid: len(recs) for rid, recs in sessions.items()}
        total_files = sum(sizes.values())
        target = min(TARGET, total_files)
        target_frac = 0.15 * total_files
        def pick(cands):
            return min(cands, key=lambda rid: (abs(sizes[rid] - target_frac), rid))
        test_id = pick(list(sizes))
        rest = [rid for rid in sizes if rid != test_id]
        val_id = pick(rest)
        train_ids = [rid for rid in rest if rid != val_id]
        role = {test_id: "test", val_id: "val"}
        for rid in train_ids:
            role[rid] = "train"
        caps = sizes
        alloc = largest_remainder(sizes, target, caps)
        counts = {"train": 0, "val": 0, "test": 0}
        for rid, n in alloc.items():
            if n <= 0:
                continue
            for rec in spaced(sessions[rid], n):
                q = dict(rec)
                q["split"] = role[rid]
                manifest.append(q)
                counts[role[rid]] += 1
        summary[sp] = {
            "archive_files": total_files,
            "n_ids_archive": len(sizes),
            "sampled": sum(counts.values()),
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
            "train_ids": sorted(train_ids),
            "val_ids": [val_id],
            "test_ids": [test_id],
            "alloc_per_id": {rid: alloc[rid] for rid in alloc if alloc[rid]},
        }
        print(f"{sp} sampled {sum(counts.values())} train {counts['train']} val {counts['val']} test {counts['test']} test_id {test_id}", flush=True)
    # leak check
    seen = defaultdict(set)
    for rec in manifest:
        seen[rec["species"], rec["recording_id"]].add(rec["split"])
    bad = [k for k, v in seen.items() if len(v) > 1]
    if bad:
        raise SystemExit("leak " + str(bad[:5]))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"summary": summary, "files": manifest}, open(MANIFEST, "w"))
    print("TOTAL", len(manifest), {s: sum(1 for r in manifest if r["split"]==s) for s in ("train","val","test")}, flush=True)
    return manifest

def make_opener():
    cj = http.cookiejar.MozillaCookieJar("/tmp/oa_cookies.txt")
    cj.load(ignore_discard=True, ignore_expires=True)
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def fetch_range(opener, start, end, retries=6):
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
        try:
            resp = opener.open(req, timeout=180)
            if resp.status != 206:
                raise RuntimeError(f"status {resp.status}")
            data = resp.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"short {len(data)}")
            return data
        except Exception as e:
            last = e
            time.sleep(min(20, 1.2 * (attempt + 1)))
    raise RuntimeError(last)

def extract_one(rec):
    dest = os.path.join(OUT, rec["base"])
    if os.path.exists(dest) and os.path.getsize(dest) == rec["u"]:
        return "cached", rec["c"]
    opener = make_opener()
    off, csize = rec["o"], rec["c"]
    budget = 1024
    blob = fetch_range(opener, off, off + 30 + budget + csize - 1)
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError("bad sig " + rec["base"])
    fnlen, exlen = struct.unpack_from("<HH", blob, 26)
    method = struct.unpack_from("<H", blob, 8)[0]
    if method != 8:
        raise RuntimeError("method")
    data_rel = 30 + fnlen + exlen
    if data_rel + csize > len(blob):
        comp = fetch_range(opener, off + data_rel, off + data_rel + csize - 1)
    else:
        comp = blob[data_rel:data_rel + csize]
    raw = zlib.decompress(comp, -15)
    if len(raw) != rec["u"] or raw[:4] != b"RIFF":
        raise RuntimeError("bad wav " + rec["base"])
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, dest)
    return "ok", csize

def main():
    manifest = build()
    todo = []
    cached = 0
    for rec in manifest:
        dest = os.path.join(OUT, rec["base"])
        if os.path.exists(dest) and os.path.getsize(dest) == rec["u"]:
            cached += 1
        else:
            todo.append(rec)
    print(f"cached {cached} to_download {len(todo)}", flush=True)
    ok = err = 0
    t0 = time.time()
    bytes_c = 0
    errors = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(extract_one, rec): rec for rec in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = futs[fut]
            try:
                status, csz = fut.result()
                ok += 1
                bytes_c += csz
            except Exception as e:
                err += 1
                errors.append((rec["base"], str(e)))
                print("ERR", rec["base"], e, flush=True)
            if i % 50 == 0 or i == len(todo):
                print(f"progress {i}/{len(todo)} ok={ok} err={err} MB={bytes_c/1e6:.1f} s={time.time()-t0:.0f}", flush=True)
    print("DONE", ok, "errors", err, "cached", cached, flush=True)
    if errors:
        json.dump(errors, open("/workspace/insect-sound/download_errors_5seed.json", "w"), indent=2)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
