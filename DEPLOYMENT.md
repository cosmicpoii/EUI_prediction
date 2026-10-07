# Deployment Guide

这个项目可以部署成别人也能访问的网站。推荐从 **Render 单服务部署** 开始，因为当前 `eui_api.py` 已经会在 `/` 直接返回 `frontend/index.html`，所以一个 Render 服务就可以同时提供网页和 API。

## Option A: Render 单服务部署，最推荐

这个方式最简单：

```text
User browser -> Render URL -> FastAPI backend + frontend + ANN model
```

### 1. 上传代码到 GitHub

Render 通常从 GitHub repo 部署。确认 repo 里至少包含：

- `eui_api.py`
- `frontend/index.html`
- `models_torch_range2_physics_01/eui_ann_torch_model.pt`
- `models_torch_range2_physics_01/eui_ann_torch_scalers.joblib`
- `requirements.txt`
- `render.yaml`

不需要上传大型 EnergyPlus hourly outputs。

### 2. 在 Render 创建服务

推荐使用 Render 的 Blueprint：

1. 打开 Render Dashboard
2. New -> Blueprint
3. 选择这个 GitHub repo
4. Render 会读取 `render.yaml`
5. 创建 `apartment-eui-predictor` web service

如果手动创建 Web Service，配置如下：

```text
Environment: Python
Build command: pip install -r requirements.txt
Start command: uvicorn eui_api:app --host 0.0.0.0 --port $PORT
```

### 3. 设置环境变量

Render service 的 Environment Variables 里设置：

```text
EUI_MODEL_PATH=models_torch_range2_physics_01/eui_ann_torch_model.pt
EUI_SCALER_PATH=models_torch_range2_physics_01/eui_ann_torch_scalers.joblib
EUI_CORS_ORIGINS=*
ANTHROPIC_API_KEY=your_anthropic_key_here
```

`ANTHROPIC_API_KEY` 只影响 AI Agent 自然语言解析。如果不设置，手动参数预测和 Find Lower EUI 仍然可以工作，但 Claude agent 功能会报 key missing。

部署包暂时保留了旧的 `models_torch_range2_physics_1` 路径作为兼容副本；其中的模型文件已经替换为最终的 `lambda=0.1` 模型。建议 Render 环境变量仍更新为上面的 `models_torch_range2_physics_01` 路径。

### 4. 访问网站

部署完成后，Render 会给一个 URL，例如：

```text
https://apartment-eui-predictor.onrender.com
```

打开这个 URL 就是网站页面。API docs 在：

```text
https://apartment-eui-predictor.onrender.com/docs
```

健康检查在：

```text
https://apartment-eui-predictor.onrender.com/health
```

## Option B: 前端 Vercel / Netlify + 后端 Render

如果想要更正式地拆成静态前端和 API 后端，可以这样：

```text
User browser -> Vercel/Netlify frontend -> Render backend API
```

### 1. 先部署后端

按照 Option A 部署 Render 后端。记下后端 URL，例如：

```text
https://apartment-eui-predictor.onrender.com
```

### 2. 部署前端到 Netlify

Netlify 可以使用根目录的 `netlify.toml`：

```toml
[build]
  publish = "frontend"
```

部署后，访问前端时加上后端 API URL：

```text
https://your-netlify-site.netlify.app/?api=https://apartment-eui-predictor.onrender.com
```

前端会把这个 API 地址保存到 browser localStorage，之后不需要每次都加 query parameter。

### 3. 部署前端到 Vercel

Vercel 项目建议把 Project Root 设置为：

```text
frontend
```

`frontend/vercel.json` 已经包含静态页面 rewrite 设置。前端默认连接下面的 Render 后端：

```text
https://apartment-eui-predictor.onrender.com
```

因此部署后的 Vercel URL 可以直接打开。若后端域名发生变化，仍可用
`?api=https://your-new-backend.example.com` 覆盖并保存新的 API 地址。

### 4. 收紧 CORS

当前为了 demo 方便，`EUI_CORS_ORIGINS=*`。正式部署时可以改成你的前端域名：

```text
EUI_CORS_ORIGINS=https://your-vercel-site.vercel.app,https://your-netlify-site.netlify.app
```

## Option C: Docker / Railway / Fly.io

项目也包含 `Dockerfile`，可以用于 Railway、Fly.io 或其他 Docker-based 平台。

本地测试：

```bash
docker build -t apartment-eui-predictor .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY="your_key_here" \
  apartment-eui-predictor
```

然后打开：

```text
http://localhost:8000
```

## Security Notes

- 不要把 `ANTHROPIC_API_KEY` 写进 `frontend/index.html`。
- Claude key 只能放在后端环境变量里。
- 前端只调用你自己的 backend API。
- ANN 模型文件可以随后端一起部署；它不包含 API key。

## Troubleshooting

### 页面右上角显示 API/model unavailable

常见原因：

- 后端还没启动成功
- 前端 API URL 没设置对
- 模型文件路径错误
- Render free service 正在 cold start，需要等几十秒

检查：

```text
https://your-backend-url/health
```

### Agent 显示 ANTHROPIC_API_KEY is not set

说明后端服务环境变量没有设置 key，或者设置后没有重启服务。

### 手动 Predict 可以用，但 Agent 不行

这是正常的分离设计。手动预测只需要 ANN 模型；Agent 需要 Claude API key。
