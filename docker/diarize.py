"""
diarize.py — sanity check + otimizado para diarização por pasta.

Modos:
  --check          Valida a stack sem áudio.
  --audio PATH     Diariza um arquivo ou uma pasta de áudios.
                   Aceita formatos que o ffmpeg consiga converter.

Segurança: patch torch.load(weights_only=False) ativo para compatibilidade com
pyannote.audio em torch >= 2.6. Use APENAS checkpoints oficiais do HF.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch as _torch

_orig_torch_load = _torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


_torch.load = _patched_torch_load

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"}


def _print_banner(title: str) -> None:
    bar = "=" * 72
    print(bar)
    print(title)
    print(bar)


def run_check() -> int:
    try:
        import ctranslate2
        import torch
        import torchaudio
        import pyannote.audio
        import whisperx

        _print_banner("CHECK DE STACK — diarize.py")
        print(f"python:          {sys.version.split()[0]}")
        print(f"torch:           {torch.__version__}")
        print(f"torch CUDA build: {torch.version.cuda or 'CPU-only'}")
        print(f"torchaudio:      {torchaudio.__version__}")
        print(f"pyannote.audio:  {pyannote.audio.__version__}")
        print(f"whisperx:        {getattr(whisperx, '__version__', 'unknown')}")
        print(f"ctranslate2:     {ctranslate2.__version__}")
        print(f"cuda available:  {torch.cuda.is_available()}")
        print()
        print("AVISO: torch.load(weights_only=False) está ativo para compatibilidade.")
        print("Use somente checkpoints confiáveis/oficiais do Hugging Face.")
        _print_banner("=== --check OK ===")
        return 0
    except Exception as exc:
        print(f"FALHA no check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg não encontrado no ambiente.")


def _normalize_audio(audio: Path, output_dir: Path) -> Path:
    if audio.suffix.lower() == ".wav":
        return audio
    _ensure_ffmpeg()
    converted = output_dir / f"{audio.stem}.wav"
    if converted.exists():
        return converted
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", "-vn", str(converted)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return converted


def _iter_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]


def _safe_json_dump(data: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _export_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["arquivo", "inicio", "fim", "falante", "texto"])
        writer.writeheader()
        writer.writerows(rows)


def _run_whisperx_diarization(
    audio_path: Path,
    output_dir: Path,
    hf_token: str,
    model_name: str,
    device: str,
    language: str,
    batch_size: int,
) -> dict:
    import torch
    import whisperx
    from pyannote.audio import Pipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA foi solicitado, mas não está disponível. Rode com '--gpus all' e configure "
            "o NVIDIA Container Toolkit e o driver NVIDIA do host."
        )

    normalized = _normalize_audio(audio_path, output_dir)
    asr_model = whisperx.load_model(
        "medium",
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        language=language,
    )
    audio = whisperx.load_audio(str(normalized))
    # WhisperX 3.1.1/Faster-Whisper accepts the decoded audio array; fp16 is
    # selected via compute_type at model load, not a transcribe() kwarg.
    result = asr_model.transcribe(audio, batch_size=batch_size, language=language)

    diar_model = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    diar_model.to(torch.device(device))
    diarization = diar_model(str(normalized))
    result = whisperx.assign_word_speakers(diarization, result)

    rows = []
    for segment in result.get("segments", []):
        if "speaker" not in segment:
            continue
        rows.append({
            "arquivo": audio_path.name,
            "inicio": round(float(segment.get("start", 0.0)), 3),
            "fim": round(float(segment.get("end", 0.0)), 3),
            "falante": segment.get("speaker", "SPEAKER_00"),
            "texto": (segment.get("text") or "").strip(),
        })

    json_path = output_dir / f"{audio_path.stem}.json"
    _safe_json_dump({"arquivo": audio_path.name, "segments": rows}, json_path)
    csv_path = output_dir / f"{audio_path.stem}.csv"
    _export_csv(rows, csv_path)
    return {"json": str(json_path), "csv": str(csv_path), "segments": len(rows)}


def run_diarize(
    audio_path: str,
    hf_token: str,
    model_name: str = "pyannote/speaker-diarization-3.1",
    device: str = "cuda",
    output_dir: Optional[str] = None,
    language: str = "pt",
    batch_size: int = 8,
) -> int:
    source = Path(audio_path)
    if not source.exists():
        print(f"ERRO: arquivo ou pasta não encontrado: {audio_path}", file=sys.stderr)
        return 2
    if not hf_token:
        print("ERRO: HF_TOKEN não configurado no ambiente.", file=sys.stderr)
        return 3

    output_root = Path(output_dir) if output_dir else Path("/saidas")
    output_root.mkdir(parents=True, exist_ok=True)
    files = _iter_audio_files(source)
    if not files:
        print(f"ERRO: nenhum áudio válido encontrado em: {source}", file=sys.stderr)
        return 2

    print(f"DIARIZAÇÃO — origem: {source}", file=sys.stderr)
    print(f"Modelo: {model_name}; device: {device}; arquivos: {len(files)}", file=sys.stderr)
    for audio in files:
        try:
            result = _run_whisperx_diarization(
                audio, output_root, hf_token, model_name, device, language, batch_size
            )
            print(f"OK: {audio.name} -> {result['segments']} segmentos | {result['json']}", file=sys.stderr)
        except Exception as exc:
            print(f"FALHA em {audio}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 4
    print(f"DIARIZAÇÃO concluída: {len(files)} arquivo(s).", file=sys.stderr)
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Diarização WhisperX + pyannote.audio")
    parser.add_argument("--check", action="store_true", help="Valida a stack sem áudio.")
    parser.add_argument("--audio", type=str, help="Arquivo ou pasta de áudios.")
    parser.add_argument("--model", type=str, default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="/saidas")
    parser.add_argument("--language", type=str, default="pt")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    if args.check:
        return run_check()
    if args.audio:
        return run_diarize(
            args.audio,
            os.environ.get("HF_TOKEN", ""),
            args.model,
            args.device,
            output_dir=args.output_dir,
            language=args.language,
            batch_size=args.batch_size,
        )
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
