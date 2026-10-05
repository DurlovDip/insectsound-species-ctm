#!/usr/bin/env python3
"""Download a balanced InsectSound1000 subset via HTTP range requests on the official ZIP."""
import json, os, zlib, struct, random, time, urllib.request, http.cookiejar
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

INDEX = "/workspace/insect-sound/cd/index.jsonl"
OUT = "/workspace/insect-sound/raw"
MANIFEST = "/workspace/insect-sound/manifest.json"
URL = "https://www.openagrar.de/servlets/MCRFileNodeServlet/openagrar_derivate_00055778/InsectSound1000.zip"
N_PER = 64
SEED_SAMPLE = 0
SEED_SPLIT = 1

SPECIES = [
    "Aphidoletes_aphidimyza", "Bombus_terrestris", "Bradysia_difformis",
    "Coccinella_septempunctata", "Episyrphus_balteatus", "Halyomorpha_halys",
    "Myzus_persicae", "Nezara_viridula", "Palomena_prasina",
    "Rhaphigaster_nebulos", "Trialeurodes_vaporariorum", "Tuta_absoluta",
]

def build_manifest():
    by = defaultdict(list)
    with open(INDEX) as f:
        for line in f:
            rec = json.loads(line)
            base = os.path.basename(rec["n"])
            sp = next(s for s in SPECIES if f"_{s}_" in base)
            rec["species"] = sp
            rec["base"] = base
            by[sp].append(rec)
    rng = random.Random(SEED_SAMPLE)
    selected = []
    for sp in SPECIES:
        groups = defaultdict(list)
        for it in by[sp]:
            prefix = it["base"].split(f"_{sp}_")[0]
            groups[prefix].append(it)
        keys = list(groups)
        rng.shuffle(keys)
        # Evenly spaced indices within each recording, then round-robin.
        pools = []
        for k in keys:
            g = sorted(groups[k], key=lambda x: x["base"])
            # target about N_PER samples total, but keep a deep pool
            step = max(1, len(g) // 40)
            take = g[::step]
            if len(take) < 8:
                take = g
            pools.append(take)
        picked = []
        while len(picked) < N_PER:
            progress = False
            for take in pools:
                if take:
                    picked.append(take.pop(0))
                    progress = True
                    if len(picked) >= N_PER:
                        break
            if not progress:
                break
        if len(picked) < N_PER:
            raise SystemExit(f"only {len(picked)} for {sp}")
        for p in picked:
            q = dict(p)
            q["species"] = sp
            selected.append(q)
        print(f"sample {sp}: {len(picked)} from {len(keys)} recording-ids", flush=True)

    rng2 = random.Random(SEED_SPLIT)
    by_sp = defaultdict(list)
    for p in selected:
        by_sp[p["species"]].append(p)
    manifest = []
    for sp in SPECIES:
        items = by_sp[sp][:]
        rng2.shuffle(items)
        n = len(items)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        for it in items[:n_train]:
            q = dict(it); q["split"] = "train"; manifest.append(q)
        for it in items[n_train:n_train + n_val]:
            q = dict(it); q["split"] = "val"; manifest.append(q)
        for it in items[n_train + n_val:]:
            q = dict(it); q["split"] = "test"; manifest.append(q)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f)
    from collections import Counter
    c = Counter(m["split"] for m in manifest)
    print("splits", dict(c), "files", len(manifest), flush=True)
    return manifest

def make_opener():
    cj = http.cookiejar.MozillaCookieJar("/tmp/oa_cookies.txt")
    cj.load(ignore_discard=True, ignore_expires=True)
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def fetch_range(opener, start, end, retries=6):
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            URL, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"}
        )
        try:
            resp = opener.open(req, timeout=180)
            if resp.status != 206:
                raise RuntimeError(f"status {resp.status}")
            data = resp.read()
            expect = end - start + 1
            if len(data) != expect:
                raise RuntimeError(f"short read {len(data)} != {expect}")
            return data
        except Exception as e:
            last = e
            time.sleep(min(20, 1.5 * (attempt + 1)))
    raise RuntimeError(f"range {start}-{end} failed: {last}")

def extract_one(rec):
    dest = os.path.join(OUT, rec["base"])
    if os.path.exists(dest) and os.path.getsize(dest) == rec["u"]:
        return rec["base"], "cached", rec["c"]
    opener = make_opener()
    off = rec["o"]
    csize = rec["c"]
    # one shot: local header + generous name/extra + payload
    # if extra is larger than budget, do a second request
    budget = 1024
    blob = fetch_range(opener, off, off + 30 + budget + csize - 1)
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError("bad local sig " + rec["base"])
    fnlen, exlen = struct.unpack_from("<HH", blob, 26)
    method = struct.unpack_from("<H", blob, 8)[0]
    if method != 8:
        raise RuntimeError(f"method {method} {rec['base']}")
    data_rel = 30 + fnlen + exlen
    if data_rel + csize > len(blob):
        # extra longer than budget; fetch exact payload
        data_off = off + data_rel
        comp = fetch_range(opener, data_off, data_off + csize - 1)
    else:
        comp = blob[data_rel:data_rel + csize]
    if len(comp) != csize:
        raise RuntimeError(f"csize mismatch {rec['base']}")
    raw = zlib.decompress(comp, -15)
    if len(raw) != rec["u"] or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RuntimeError(f"bad wav {rec['base']} len={len(raw)}")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, dest)
    return rec["base"], "ok", csize

def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))
        print("loaded manifest", len(manifest), flush=True)
    else:
        manifest = build_manifest()
    # download
    todo = []
    for rec in manifest:
        dest = os.path.join(OUT, rec["base"])
        if not (os.path.exists(dest) and os.path.getsize(dest) == rec["u"]):
            todo.append(rec)
    print(f"to download {len(todo)} / {len(manifest)}", flush=True)
    ok = 0
    bytes_c = 0
    errors = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(extract_one, rec): rec for rec in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = futs[fut]
            try:
                name, status, csz = fut.result()
                ok += 1
                bytes_c += csz
            except Exception as e:
                errors.append((rec["base"], str(e)))
                print("ERR", rec["base"], e, flush=True)
            if i % 25 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"progress {i}/{len(todo)} ok={ok} err={len(errors)} compressed_MB={bytes_c/1e6:.1f} elapsed_s={dt:.0f}", flush=True)
    print("DONE", ok, "errors", len(errors), flush=True)
    if errors:
        with open("/workspace/insect-sound/download_errors.json", "w") as f:
            json.dump(errors, f, indent=2)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
