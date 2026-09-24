# 🎨 FLUX API - Geração & Edição Universal de Imagens

Uma API REST de alta performance, leve e stateless para geração (**Text-to-Image**) e edição (**Image-to-Image / Inpainting**) de imagens, construída com FastAPI e acelerada pelo modelo **FLUX.1** com quantização **4-bit (NF4)**.

Projetada especificamente para integração com bots (Telegram, WhatsApp) e agentes autônomos, servindo como o motor de geração e edição visual primário para o ecossistema [Ravena AI](https://github.com/moothz/ravena-ai).

---

## ✨ Recursos

- **Geração Text-to-Image (T2I)**: Geração fotorrealista a partir de prompts de texto (`POST /v1/images/generations`).
- **Edição Image-to-Image (Img2Img)**: Transformação e modificação de imagens existentes guiada por texto e fator de força (`POST /v1/images/edits`).
- **Inpainting com Máscara**: Substituição ou adição pontual de elementos em áreas mascaradas (`POST /v1/images/inpainting`).
- **Quantização 4-bit (NF4)**: Ocupa apenas **~7.0 a 7.5 GB de VRAM**, permitindo coexistência perfeita com LLMs pesados na GPU.
- **Memória Compartilhada**: Text2Image, Img2Img e Inpainting compartilham os mesmos pesos em VRAM sem duplicação de memória.
- **Docker Compose & PM2 Ready**: Totalmente containerizado com suporte a GPU NVIDIA e scripts prontos para gerenciamento de processos.
- **OpenAI Compatible**: Suporta retornos em formato binário direto (`image/png`) ou JSON Base64 padrão OpenAI.

---

## ⚙️ Pré-requisitos

1. **Docker** & **Docker Compose** com suporte a **NVIDIA Container Toolkit** (`nvidia-smi` acessível dentro de containers).
2. **GPU NVIDIA** com 8 GB+ de VRAM (otimizado para RTX 30/40/50 series).
3. **PM2** (opcional, para gerenciamento como serviço no host).

---

## 🚀 Como Rodar

### 1. Clone o Repositório

```bash
git clone https://github.com/moothz/flux-api.git
cd flux-api
```

### 2. Configure as Variáveis de Ambiente

Copie o modelo de ambiente `.env.example` para `.env`:

```bash
cp .env.example .env
```

Ajuste as configurações no `.env` conforme necessário:

```env
HOST=0.0.0.0
PORT=13005
MODEL_ID=black-forest-labs/FLUX.1-schnell
QUANTIZATION=nf4
DEFAULT_STEPS=4
DEFAULT_WIDTH=1024
DEFAULT_HEIGHT=1024
```

### 3. Inicie com Docker Compose

```bash
# Build e execução em primeiro plano
docker compose up --build

# Ou em segundo plano (detached)
docker compose up -d
```

### 4. Ou gerencie via PM2

```bash
pm2 start ./start_flux.sh --name flux-api
pm2 save
```

---

## 📚 Endpoints Principais

### 1. Text-to-Image (`POST /v1/images/generations`)

```bash
curl -X POST "http://localhost:13005/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cinematic photograph of a majestic raven in a misty forest, 8k",
    "width": 1024,
    "height": 1024,
    "steps": 4
  }' \
  --output result.png
```

### 2. Image-to-Image (`POST /v1/images/edits`)

```bash
curl -X POST "http://localhost:13005/v1/images/edits" \
  -F "image=@input.png" \
  -F "prompt=Transform into a cyberpunk neon aesthetic with rainy reflection" \
  -F "strength=0.60" \
  -F "steps=4" \
  --output edited.png
```

### 3. Status & VRAM Telemetry (`GET /health`)

```bash
curl -s "http://localhost:13005/health"
```

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
