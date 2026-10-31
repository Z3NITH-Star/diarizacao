"""
diarize.py - Script base para diarização com whisperx + pyannote.audio.

Modos:
  --check    Valida a stack sem áudio (M3). Imprime versões + aviso de
             segurança. Não deve gerar traceback em sucesso.
  --audio    Diarização com áudio real (M4+). Requer HF_TOKEN e caminho
             válido. Implementação completa pendente até M3 verde.

Segurança:
  Patch `torch.load(weights_only=False)` ativo por compatibilidade com
  pyannote.audio 3.1.x/3.4.x + torch>=2.6. Usar APENAS checkpoints oficiais
  do Hugging Face com HF_TOKEN válido.

Milestones cobertos por este arquivo:
  M3 -> --check (exit 0 em sucesso, 1 em falha)
  M4 -> --audio (implementação futura; exit 4 enquanto não implementado)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# Patch de segurança: torch.load(weights_only=False)
# =============================================================================
# pyannote.audio 3.x chama torch.load() sem o argumento weights_only=.
# A partir do torch>=2.6, o default passou a ser weights_only=True, o que
# quebra o load de checkpoints do pyannote com "Unsupported global" ou
# erro de unpickling.
#
# Este monkey-patch restaura o default anterior (weights_only=False).
#
# >>> RISCO <<<
# weights_only=False desativa a proteção contra pickle malicioso. Um
# checkpoint adulterado pode executar código arbitrário no processo que
# carrega o arquivo.
#
# >>> MITIGAÇÃO OBRIGATÓRIA <<<
# 1. Usar apenas checkpoints oficiais do Hugging Face:
#    - pyannote/speaker-diarization-3.1
#    - pyannote/segmentation-3.0
# 2. HF_TOKEN configurado e com acesso aprovado no HF.
# 3. Nunca carregar checkpoints de origem desconhecida.
# 4. Validar origem do checkpoint antes de cada build de produção.
# =============================================================================
import torch as _torch

_orig_torch_load = _torch.load


def _patched_torch_load(*args, **kwargs):
    """Restaura o default weights_only=False para compatibilidade com
    pyannote.audio 3.4.0 em torch>=2.6."""
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
        return 0
    except Exception as exc:
        print(
            f"FALHA no check: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


def run_diarize(
    audio_path: str,
    hf_token: str,
    model_name: str = "pyannote/speaker-diarization-3.1",
    device: str = "cpu",
) -> int:
    """M4+: diarização com áudio real.

    Implementação pendente. Disponível apenas quando M3 estiver 100% verde.
    """
    audio = Path(audio_path)
    if not audio.is_file():
        print(f"ERRO: arquivo de áudio não encontrado: {audio_path}", file=sys.stderr)
        return 2
    if not hf_token:
        print("ERRO: HF_TOKEN não configurado no ambiente.", file=sys.stderr)
        return 3

    print(f"DIARIZAÇÃO — áudio: {audio_path}")
    print(f"Modelo: {model_name}")
    print(f"Device: {device}")
    print(
        "AVISO: implementação M4 pendente. Use --check para validar stack. "
        "Quando M3 estiver verde, habilitar pipeline completo aqui.",
        file=sys.stderr,
    )
    return 4


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diarização whisperx + pyannote.audio",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Valida a stack sem áudio (M3). Imprime versões + aviso de segurança.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Caminho do áudio para diarização (M4+).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Modelo HF para diarização (default: pyannote/speaker-diarization-3.1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device para inferência (default: cpu). M5 habilita cuda.",
    )

    args = parser.parse_args(argv)

    if args.check:
        return run_check()

    if args.audio:
        return run_diarize(
            args.audio,
            os.environ.get("HF_TOKEN", ""),
            args.model,
            args.device,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
