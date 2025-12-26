# =========================
# 1️⃣ Build stage
# =========================
FROM rust:1.85-slim AS builder


WORKDIR /app

# ---- system deps (minimal) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*


# ---- cache deps ----
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# ---- real source ----
COPY src ./src
COPY embeddings.bin .
COPY models ./models

# ---- build ----
ENV RUSTFLAGS="-C target-cpu=native"
RUN cargo build --release


# =========================
# 2️⃣ Runtime stage
# =========================
FROM debian:bookworm-slim

WORKDIR /app

# ---- runtime deps only ----
RUN apt-get update && apt-get install -y \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- copy binary + assets ----
COPY --from=builder /app/target/release/animood /app/animood
COPY --from=builder /app/embeddings.bin /app/embeddings.bin
COPY --from=builder /app/models /app/models

# ---- environment (IMPORTANT for 1-core VM) ----
ENV RUST_LOG=info
ENV RAYON_NUM_THREADS=1
ENV MODEL_DIR=/app/models/jina-embeddings-v2-small-en
ENV EMBEDDINGS_PATH=/app/embeddings.bin

# ---- expose axum port ----
EXPOSE 3000

# ---- run ----
CMD ["/app/animood"]
