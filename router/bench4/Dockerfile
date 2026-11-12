# bench-4: one image holding every harness arm, pinned, so the arms differ only in the harness.
# Built on the runner before each batch. Nothing in it holds a key: the model is reached through the
# host's meter, which holds the key and is the container's only network peer.
FROM mirror.gcr.io/library/node:22-bookworm

ARG PI_VERSION=1.5.2
ARG OPENCODE_VERSION=1.19.5
ARG GOOSE_TAG=v1.59.0
ARG MINI_VERSION=2.4.6
ARG OH_REF=v1.61.0

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends bzip2 ca-certificates curl git python3 \
    && rm -rf /var/lib/apt/lists/*

# TypeScript harnesses
RUN npm install -g --no-fund --no-audit "@earendil-works/pi-coding-agent@${PI_VERSION}" "opencode-ai@${OPENCODE_VERSION}"

# Goose (Rust binary, pinned release)
RUN curl -fsSL -o /tmp/goose.tbz "https://github.com/block/goose/releases/download/${GOOSE_TAG}/goose-x86_64-unknown-linux-gnu.tar.bz2" \
    && mkdir -p /tmp/g && tar -xjf /tmp/goose.tbz -C /tmp/g && install -m 755 "$(find /tmp/g -type f -name goose | head -1)" /usr/local/bin/goose && rm -rf /tmp/goose.tbz /tmp/g && goose --version

# Python harnesses, each in its own environment
RUN curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv venv -q -p 3.12 /opt/mini && uv pip install -q -p /opt/mini "mini-swe-agent==${MINI_VERSION}" \
    && uv venv -q -p 3.12 /opt/oh \
    && for p in openhands-sdk openhands-tools; do \
         uv pip install -q -p /opt/oh "$p @ git+https://github.com/OpenHands/software-agent-sdk@${OH_REF}#subdirectory=$p"; done

COPY run_in_container.sh run_openhands.py mock_upstream.py /opt/bench4/

# Warm every harness once against a local mock model while the network is still open, so that nothing
# a harness fetches on first run (OpenCode's provider package, model lists) is needed inside the seal.
RUN cd /tmp && mkdir -p /workspace && (python3 /opt/bench4/mock_upstream.py 8099 /tmp/mock.log &) && sleep 2 \
    && echo "Say done." > /tmp/warm.md \
    && for h in goose pi opencode mini openhands; do \
         BENCH4_BASE=http://127.0.0.1:8099 BENCH4_MODEL=warm/model timeout 300 sh /opt/bench4/run_in_container.sh "$h" /tmp/warm.md /tmp/warm-$h \
           || echo "warm $h exit $?"; done \
    && cat /tmp/mock.log | wc -l && rm -rf /tmp/warm-* /workspace/* /tmp/mock.log
