#!/usr/bin/env python3
"""check_compat.py — conflitos entre pins contra requires_dist do PyPI."""
import json, re, sys
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"

def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

def parse_requirements(path: str):
    pins = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = re.match(r"^([A-Za-z0-9._\-]+)==([A-Za-z0-9.+\-]+)$", line)
            if m:
                pins[normalize(m.group(1))] = m.group(2)
    return pins

_cache = {}

def fetch_dependencies(name: str, version: str):
    key = (name, version)
    if key in _cache:
        return _cache[key]
    url = PYPI_JSON.format(name=name, version=version)
    deps = None
    erro = None
    for _ in range(3):
        req = Request(url, headers={"User-Agent": "check-compat/1.1"})
        try:
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            requires = data.get("info", {}).get("requires_dist") or []
            deps = [Requirement(r) for r in requires]
            break
        except (URLError, HTTPError, json.JSONDecodeError, ValueError, OSError) as e:
            erro = e
    if deps is None:
        print(f"  ⚠️  Não consegui ler metadados de {name}=={version}: {erro}")
    _cache[key] = deps
    return deps

def check(pins):
    conflitos = falhas = arestas = 0
    print(f"\n🔍 Verificando {len(pins)} pins contra os metadados do PyPI...\n")
    for pkg, ver in sorted(pins.items()):
        deps = fetch_dependencies(pkg, ver)
        if deps is None:
            falhas += 1
            continue
        for dep in deps:
            dep_name = normalize(dep.name)
            if dep_name not in pins:
                continue
            if dep.marker and not dep.marker.evaluate():
                continue
            arestas += 1
            pinned_ver = pins[dep_name]
            try:
                ok = dep.specifier.contains(Version(pinned_ver), prereleases=True)
            except InvalidVersion:
                print(f"  ⚠️  versão ilegível: {dep_name}=={pinned_ver}")
                falhas += 1
                continue
            if not ok:
                print(f"  ❌  {pkg}=={ver}  exige  {dep_name}{dep.specifier}")
                print(f"      mas o pin é       {dep_name}=={pinned_ver}\n")
                conflitos += 1
    print(f"  🔗 {arestas} restrição(ões) verificada(s) entre pins.")
    if conflitos:
        print(f"  ⛔ {conflitos} conflito(s) encontrado(s).\n")
        return 1
    if falhas:
        print(f"  ⚠️  {falhas} pacote(s) sem metadado — este 'verde' não cobre tudo.\n")
        return 2
    print("  ✅  Nenhum conflito encontrado entre os pins.\n")
    return 0

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docker/requirements.txt"
    sys.exit(check(parse_requirements(path)))
