import os
import json
import urllib.request
import concurrent.futures

BASE_URL = "https://auk-project.github.io/"
ASSETS_DIR = "/Users/vanch/mlx-AuK/assets/demo_assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

with open("/tmp/all_families.json", "r", encoding="utf-8") as f:
    families = json.load(f)

urls_to_download = {}

for fam in families:
    for grp in fam.get("groups", []):
        for s in grp.get("samples", []):
            audio = s.get("audio", {})
            if isinstance(audio, dict):
                src = audio.get("src")
                if src:
                    full_url = urllib.parse.urljoin(BASE_URL, src)
                    local_filename = os.path.basename(src)
                    local_path = os.path.join(ASSETS_DIR, local_filename)
                    urls_to_download[full_url] = local_path

print(f"Total unique input audio assets to download: {len(urls_to_download)}")

def download_file(item):
    url, path = item
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return url, True, "cached"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            with open(path, "wb") as out:
                out.write(data)
        return url, True, f"{len(data)} bytes"
    except Exception as e:
        return url, False, str(e)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(download_file, urls_to_download.items()))

success_cnt = 0
for url, ok, msg in results:
    if ok:
        success_cnt += 1
    else:
        print(f"Failed: {url} -> {msg}")

print(f"Downloaded {success_cnt}/{len(urls_to_download)} audio assets to {ASSETS_DIR}")
