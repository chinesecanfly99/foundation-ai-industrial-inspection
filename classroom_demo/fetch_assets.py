"""Fetch only previously public artifacts, checking their model checksums."""
import hashlib
import io
import urllib.request
from pathlib import Path
from zipfile import ZipFile

URL='https://github.com/chinesecanfly99/foundation-ai-industrial-inspection/releases/download/v0.2.0/demo-assets-v0.2.0.zip'
ZIP_SHA='faa82d2228abb7b2f2803e14ed89d5e286a13758c5886d7b81e2bcf482f07d98'

def fetch(url):
    with urllib.request.urlopen(url,timeout=180) as response:
        return response.read(130_000_001)

def save(path,data,digest=None):
    if digest is not None and not hashlib.sha256(data).hexdigest().startswith(digest):
        raise ValueError(f'Unexpected checksum: {path}')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)

expected={'bottle_patch':'82a62f71f0acb36cdb178452cb7595f4475fa626789fc42656b81e8e0a99cbb8',
          'bottle_pixel':'b5fdb04651fd98d245de32406178953b48a1db54491f64abd9fae8045824fc21',
          'tile_patch':'2c26fb958ca2951daecd77cfb422a53dc077d4309892d741e079ea73ee934e05',
          'tile_pixel':'41ccf4b82e7f57e7feb6f2f02dd83fd6026337dbfc10b19c5804aae2ba522e4b'}

def main():
    blob=fetch(URL)
    if hashlib.sha256(blob).hexdigest()!=ZIP_SHA:
        raise ValueError('Unexpected public release checksum')
    with ZipFile(io.BytesIO(blob)) as z:
        for model,digest in expected.items():
            for name in ('model.npz','model_card.json'):
                rel=f'models/{model}/{name}'
                key=next(k for k in z.namelist() if k==rel or k.endswith('/'+rel))
                save(rel,z.read(key),digest if name.endswith('.npz') else None)
        weights=next(k for k in z.namelist() if k.endswith('/resnet18-f37072fd.pth'))
        save('torch_cache/hub/checkpoints/resnet18-f37072fd.pth',z.read(weights),'f37072fd')

if __name__=='__main__':main()
