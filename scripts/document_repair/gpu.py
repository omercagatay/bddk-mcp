"""Stage 3b, optional: recognition of the masked region images on a GPU host over ssh.

Only runs when a host alias is given (``--gpu-host`` or ``BDDK_REPAIR_GPU_HOST``). The alias is an entry of the
operator's ssh configuration; host name, user and key stay there. What leaves this machine: the masked PNG images
rendered for this document in this run and two name lists, sent to ``<workdir>/docs/<doc>/`` on that host. What comes
back: one Markdown file per image and ``meta.jsonl``. ``recognize.py`` must already be installed at
``<workdir>/recognize.py`` on the host (copied there once by the operator); it is never uploaded by this tool.
This is ssh, not the HTTP outbound path, and the host is not in the HTTP outbound allowlist.
"""

from __future__ import annotations

import io
import re
import shlex
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DOC_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
REMOTE_PATH_RE = re.compile(r"/?[A-Za-z0-9_][A-Za-z0-9_./-]*")
POLL_SECONDS = 30
WAIT_SECONDS = 6 * 3600  # ponytail: fixed deadline; make it an option if a document ever needs longer
MAX_OUTPUT_BYTES = 1 << 20  # one recognition output; LightOnOCR stops at 4096 new tokens


def _check(value: str, pattern: re.Pattern, what: str) -> str:
    if not pattern.fullmatch(value) or ".." in value.split("/"):
        raise ValueError(f"unsafe {what}: {value!r}")
    return value


def _ssh(host, command, **kw):
    """One remote command; every remote word is quoted (and validated before)."""
    proc = subprocess.run([*SSH, "--", host, command], **kw)
    if proc.returncode == 255:
        raise RuntimeError(f"ssh to {host!r} failed (exit 255)")
    return proc


def _upload_tar(masked: Path, names: list[str], lists: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in names:
            tar.add(masked / f"{name}.png", arcname=f"{name}.png", recursive=False)
        for fname, text in lists.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(fname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _unpack_outputs(data: bytes, doc: str, outdir: Path) -> int:
    """Write the expected output files only: <doc>_pNNN_K_<v>.md and meta.jsonl, regular files, bounded size."""
    want = re.compile(rf"{re.escape(doc)}_p\d{{3}}_\d+_[abcd]\.md|meta\.jsonl")
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        for member in tar:
            name = PurePosixPath(member.name).name
            if not member.isfile() or not want.fullmatch(name) or member.size > 64 * MAX_OUTPUT_BYTES:
                continue
            (outdir / name).write_bytes(tar.extractfile(member).read())
            n += 1
    return n


def run(host, doc, masked: Path, lists: dict[str, list[str]], outdir: Path, workdir: str, python: str) -> float:
    """Send this run's masked images, run recognize.py (4B on a, c, d; 1B on b) in the background, wait, fetch."""
    host = _check(host, HOST_RE, "ssh host alias")
    doc = _check(doc, DOC_RE, "document name")
    workdir = _check(workdir, REMOTE_PATH_RE, "remote working directory")
    python = _check(python, REMOTE_PATH_RE, "remote python")
    rd = f"{workdir}/docs/{doc}"
    q = shlex.quote
    names = [n for k in ("a", "b", "c", "d") for n in lists.get(k, [])]
    l4b = "".join(n + "\n" for k in ("a", "c", "d") for n in lists.get(k, []))
    l1b = "".join(n + "\n" for n in lists.get("b", []))
    print(f"GPU step: sending {len(names)} masked images of {doc} to {host}:{rd}/img", flush=True)
    # a fresh directory per run: outputs left by an earlier run of the same name are never reused
    _ssh(host, f"rm -rf {q(rd)} && mkdir -p {q(rd + '/img')} {q(rd + '/out')}", check=True)
    payload = _upload_tar(masked, names, {"l4b.txt": l4b, "l1b.txt": l1b})
    _ssh(host, f"tar xf - -C {q(rd + '/img')}", input=payload, check=True)
    rec = f"{q(python)} -I {q(workdir + '/recognize.py')}"
    img, out = q(rd + "/img"), q(rd + "/out")
    script = (
        f"{rec} 4b {img} {q(rd + '/img/l4b.txt')} {out} > {q(rd + '/r4b.log')} 2>&1; "
        f"{rec} 1b {img} {q(rd + '/img/l1b.txt')} {out} > {q(rd + '/r1b.log')} 2>&1; "
        f"echo DONE > {q(rd + '/done.flag')}"
    )
    _ssh(host, f"setsid nohup sh -c {q(script)} > /dev/null 2>&1 < /dev/null &", check=True)
    t0 = time.time()
    while _ssh(host, f"test -f {q(rd + '/done.flag')}").returncode != 0:
        if time.time() - t0 > WAIT_SECONDS:
            raise TimeoutError(f"no done.flag in {rd} on {host} after {WAIT_SECONDS} s")
        time.sleep(POLL_SECONDS)
    data = _ssh(host, f"tar cf - -C {out} .", capture_output=True, check=True).stdout
    n = _unpack_outputs(data, doc, outdir)
    print(f"GPU step: {n} files back from {host}; logs stay in {rd}/r4b.log and r1b.log there", flush=True)
    return time.time() - t0
