# Deploying breve web

The **huge win** is the browser: chat (optional) + live 3D. Deploy the web app; users open a URL.

## Local (dev)

```bash
cd breve
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[webai]"
export XAI_API_KEY=xai-...   # optional — users can paste a key in the UI
breve-web --host 0.0.0.0 --port 8765
```

Open http://127.0.0.1:8765  

- **No key:** curriculum demos auto-play  
- **With key:** natural-language scene building via Grok  

## Docker

```bash
docker build -t breve-web .
docker run --rm -p 8765:8765 -e XAI_API_KEY -e XAI_MODEL=grok-4.5 breve-web
```

## Fly.io (example)

```bash
fly launch --name breve-sandbox --region sjc
fly secrets set XAI_API_KEY=xai-...
fly deploy
```

`Dockerfile` uses port **8765**. Set `PORT` if your host injects it (see Dockerfile `CMD`).

## Railway / Render

1. Connect the repo  
2. Build: `pip install -e ".[webai]"`  
3. Start: `breve-web --host 0.0.0.0 --port ${PORT:-8765}`  
4. Secret: `XAI_API_KEY` (optional if clients paste keys)  

## Share links

- **Example:** `https://your.host/?example=example_gravity`  
- **Full scene:** `https://your.host/?s=<compressed-token>`  
  Created via the **Share** button (copies to clipboard).  

Tokens are zlib+base64url of scene JSON (no server storage). Very large scenes may exceed browser URL limits; the API reports `ok: false` when oversized.

## Security notes

- Prefer **server-side** `XAI_API_KEY` for demos you host; keys pasted in the browser are sent only to *your* backend for that request (stored in `localStorage` on the client).  
- Do not expose an unrestricted public proxy to xAI without rate limits — add auth or quotas before a public internet deploy.  
- The model only emits **declarative JSON**; the server never `exec`s model code.  

## Health checks

```bash
curl -s https://your.host/api/status
curl -s https://your.host/api/curriculum
```
