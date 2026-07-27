# ─────────────────────────────────────────────────────────────
# BeeSearch — React Frontend Container
#
# Multi-stage build: compiles the Vite/React SPA, then serves the
# static output through nginx with /api/* reverse-proxied to the
# FastAPI backend container (see ../docker-compose.yml).
#
# Build:   docker build -t beesearch-frontend ./frontend
# Run:     docker run -p 5173:80 beesearch-frontend
# ─────────────────────────────────────────────────────────────

# ── Stage 1: build the static assets ──────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

# Copy manifests first so Docker layer cache is reused on code-only changes
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Stage 2: serve the build with nginx ────────────────────────
FROM nginx:1.27-alpine

LABEL maintainer="BeeSearch"
LABEL description="BeeSearch React frontend -- static build served by nginx"

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
