# diarizacao

Diarização de falantes com `whisperx` + `pyannote.audio`.

## Estado

| Marco | O que prova | Estado |
|---|---|---|
| M1 | repositório existe e builda | 🟡 |
| M2 | dependências fecham (CI) | 🔴 |
| M3 | `--check` roda no host alvo | 🔴 |
| M4 | diariza áudio real | 🔴 |
| M5 | usa GPU | 🔴 |
| M6 | `--audio` em `.txt` existente | 🔴 |

## Build

Contexto = **`./docker`** (a pasta do Dockerfile). Todo `COPY` é sem prefixo.

```bash
docker build --progress=plain -f docker/Dockerfile docker/ -t diarizacao:cpu
```

GPU (M5):

```bash
docker build --progress=plain -f docker/Dockerfile docker/ \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
  --build-arg TORCH_VERSION=2.5.1+cu124 \
  -t diarizacao:cuda
```

> [!note] arm64
> `torch 2.5.1+cpu` só publica wheel `linux_x86_64`/`win_amd64` no índice CPU. Em arm64 o `pip` termina em `No matching distribution found` — não é bug do Dockerfile.

## Validação

```bash
# Gate 1 — dependências
docker run --rm --entrypoint pip diarizacao:cpu check
# esperado: No broken requirements found.

# Gate 2 — stack importa
docker run --rm diarizacao:cpu --check
# esperado: 6 linhas + aviso de segurança, exit 0

# Gate 3 — --audio não mente (M4 ainda não implementado)
docker run --rm diarizacao:cpu --audio x.wav
# esperado: exit 4
```

## Diarização real (M4+)

Requer `HF_TOKEN` e acesso aprovado aos modelos no Hugging Face:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

```bash
docker run --rm -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD/audio:/audio" -v "$PWD/saidas:/saidas" -v "$PWD/models:/models" \
  diarizacao:cpu --audio /audio/sessao.wav --device cpu
```

## Volumes

| Caminho | Uso |
|---|---|
| `/audio` | áudio de entrada |
| `/saidas` | `.txt` / `.json` de saída |
| `/models` | cache HF (`HF_HOME=/models`) — sem este volume, cada execução rebaixa os modelos |

## ⚠️ Segurança — `torch.load(weights_only=False)`

O `diarize.py` aplica um monkey-patch que restaura `weights_only=False` no
`torch.load`, porque o `pyannote.audio` 3.1.x chama `torch.load()` sem esse
argumento e, a partir do `torch>=2.6`, o default mudou para `weights_only=True`.
Com o pin atual (`2.5.1`) o patch é inócuo — mas fica, porque o M5 pode subir de
`torch`.

**Risco:** `weights_only=False` desativa a proteção contra pickles maliciosos.
Um checkpoint adulterado pode executar código arbitrário.

**Mitigação obrigatória:**

- Use **apenas** checkpoints oficiais do Hugging Face
  (`pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`).
- `HF_TOKEN` com acesso aprovado.
- Nunca carregue checkpoints de origem desconhecida.

## Estrutura

```text
.
├── .github/workflows/build-diarizacao.yml
├── docker/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── diarize.py
│   └── .dockerignore
├── .env.example
├── .gitignore
└── README.md
```
