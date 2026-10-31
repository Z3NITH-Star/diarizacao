#!/usr/bin/env python3
"""
check_compat.py — Verifica conflitos entre pins do requirements.txt
contra os metadados reais de dependência no PyPI (requires_dist).

Uso:
    python check_compat.py docker/requirements.txt

Sai com:
    0  nenhum conflito
    1  conflito(s) entre pins
    2  algum metadado não pôde ser lido (verde não é confiável nesse caso)

Requer: pip install packaging
"""

import json
import re
import sys
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"


def normalize(name: str) -> str:
    """PEP 503: pyannote.audio == pyannote-audio == pyannote_audio."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: str) -> dict[str, str]:
    """Extrai {nome: versão} de linhas com pin == (ignora comentários e opções)."""
    pins: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = re.match(r"^([A-Za-z0-9._\-]+)==([A-Za-z0-9.+\-]+)$", line)
            if m:
                pins[normalize(m.group(1))] = m.group(2)
    return pins


_cache: dict[tuple[str, str], list[Requirement] | None] = {}


def fetch_dependencies(name: str, version: str) -> list[Requirement] | None:
    """Metadados de name==version. None = falhou (não é o mesmo que 'sem deps')."""
    key = (name, version)
    if key in _cache:
        return _cache[key]

    url = PYPI_JSON.format(name=name, version=version)
    deps = None
    erro = None
    for tentativa in range(3):
        req = Request(url, headers={"User-Agent": "check-compat/1.1"})
        try:
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            requires = data.get("info", {}).get("requires_dist") or []
            deps = [Requirement(r) for r in requires]
            break
        except (URLError, HTTPError, json.JSONDecodeError, ValueError, OSError) as e:
            # OSError cobre ConnectionResetError/timeout, que o urlopen deixa escapar.
            erro = e

    if deps is None:
        print(f"  ⚠️  Não consegui ler metadados de {name}=={version}: {erro}")

    _cache[key] = deps
    return deps


def check(pins: dict[str, str]) -> int:
    conflitos = 0
    falhas = 0
    arestas = 0

    print(f"\n🔍 Verificando {len(pins)} pins contra os metadados do PyPI...\n")

    for pkg, ver in sorted(pins.items()):
        deps = fetch_dependencies(pkg, ver)
        if deps is None:
            falhas += 1
            continue

        for dep in deps:
            dep_name = normalize(dep.name)
            if dep_name not in pins:          # só checa o que também está pinado
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
                print(f"      mas o pin é       {dep_name}=={pinned_ver}")
                conflitos += 1
                print()

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
