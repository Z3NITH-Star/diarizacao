"""
diarize.py — sanity-check + skeleton (Meta A v2).

Modos:
  --check    Valida a stack sem áudio (M3). Imprime versões + aviso de
             segurança. Exit 0 em sucesso, 1 em falha.
  --audio    Diarização com áudio real (M4+). Retorna 4 (não implementado)
             para evitar falso sucesso em CI.

Segurança: patch torch.load(weights_only=False) ativo. Ver README.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# Patch de segurança: torch.load(weights_only=False)
# Necessário para pyannote.audio==3.4.0 em torch>=2.6 (default virou True).
# DESATIVA a proteção contra pickle malicioso.
# Mitigação: usar APENAS checkpoints oficiais do HF com HF_TOKEN válido.
# =============================================================================
import torch as _torch

_orig_torch_load = _torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


_torch.load = _patched_torch_load
# =============================================================================


def run_check() -> int:
    """M3: valida a stack sem áudio. Sem traceback em sucesso."""
    try:
        import torch
        import torchaudio
        import pyannote.audio
        import whisperx

        bar = "=" * 64
        print(bar)
        print("CHECK DE STACK — diarize.py")
        print(bar)
        print(f"python:          {sys.version.split()[0]}")
        print(f"torch:           {torch.__version__}")
        print(f"torchaudio:      {torchaudio.__version__}")
        print(f"pyannote.audio:  {pyannote.audio.__version__}")
        print(f"whisperx:        {getattr(whisperx, '__version__', 'unknown')}")
        print(f"cuda available:  {torch.cuda.is_available()}")
        print(bar)
        print()
        print("AVISO DE SEGURANÇA:")
        print("  Patch torch.load(weights_only=False) ATIVO.")
        print("  Risco: pickle malicioso pode executar código arbitrário.")
        print("  Mitigação: usar APENAS checkpoints oficiais do Hugging Face")
        print("             (pyannote/speaker-diarization-3.1) com HF_TOKEN")
        print("             válido e acesso aprovado.")
        print(bar)
        print("=== --check OK ===")
        return 0
    except Exception as exc:
        print(f"FALHA no check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def run_diarize(
    audio_path: str,
    hf_token: str,
    model_name: str = "pyannote/speaker-diarization-3.1",
    device: str = "cpu",
) -> int:
    """M4+: diarização com áudio real. Pendente até M3 verde."""
    audio = Path(audio_path)
    if not audio.is_file():
        print(f"ERRO: arquivo de áudio não encontrado: {audio_path}", file=sys.stderr)
        return 2
    if not hf_token:
        print("ERRO: HF_TOKEN não configurado no ambiente.", file=sys.stderr)
        return 3

    print(f"DIARIZAÇÃO — áudio: {audio_path}", file=sys.stderr)
    print(f"Modelo: {model_name}", file=sys.stderr)
    print(f"Device: {device}", file=sys.stderr)
    print(
        "ERRO: implementação M4 pendente. Use --check para validar stack.",
        file=sys.stderr,
    )
    return 4  # <-- BUG 5 FIX: não retorna 0 (falso sucesso em CI)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Diarização whisperx + pyannote.audio")
    parser.add_argument("--check", action="store_true",
                        help="Valida a stack sem áudio (M3).")
    parser.add_argument("--audio", type=str, help="Caminho do áudio (M4+).")
    parser.add_argument("--model", type=str,
                        default="pyannote/speaker-diarization-3.1",
                        help="Modelo HF para diarização.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device de inferência.")
    args = parser.parse_args(argv)

    if args.check:
        return run_check()
    if args.audio:
        return run_diarize(args.audio, os.environ.get("HF_TOKEN", ""),
                           args.model, args.device)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
