#!/bin/bash
# Pilot 18: Utility Engineering's own environment, with their pinned versions (requirements.txt at commit 5e5966d), in a
# venv next to the pod's Python, which our own code keeps using. vLLM 0.6.6.post1 brings torch 2.5.1 (CUDA 12.4).
#   bash ue_setup.sh            (UE = their clone, VENV = where the venv goes)
set -e
UE=${UE:-/workspace/emergent-values}
VENV=${VENV:-/workspace/uevenv}
python3 -m pip install -q --break-system-packages uv 2>/dev/null || python3 -m pip install -q uv
uv venv -q "$VENV" --python "$(command -v python3)"
VPY=$VENV/bin/python
PINNED=("vllm==0.6.6.post1" "torch==2.5.1" "transformers==4.46.2" "anthropic==0.45.2" "fireworks-ai==0.15.8"
        "litellm==1.61.5" "openai==1.63.0" "python-dotenv==1.0.1" "networkx==3.3" "scikit-learn==1.6.1" "pandas==2.2.3"
        "huggingface_hub==0.26.2" "PyYAML==6.0.2" "scipy==1.15.1" "tqdm==4.67.0" "accelerate==1.1.1" "Pillow==11.1.0"
        "protobuf==5.29.3" google-generativeai)
if ! uv pip install -q --python "$VPY" "${PINNED[@]}"; then
  echo "UE SETUP: the full pin set does not resolve; keeping vLLM, torch and transformers pinned, the rest loose"
  uv pip install -q --python "$VPY" "vllm==0.6.6.post1" "torch==2.5.1" "transformers==4.46.2" anthropic fireworks-ai litellm \
    openai python-dotenv networkx scikit-learn pandas huggingface_hub pyyaml scipy tqdm accelerate pillow protobuf google-generativeai
fi
# Their llm_agent.py imports every API client at the top. If one still fails to import, a stub stands in for it inside
# the venv only: the API clients are never called here, since the model runs locally through vLLM.
"$VPY" - <<'EOF'
import importlib, site, os
sp = site.getsitepackages()[0]
stub = "class _Any(type):\n    def __getattr__(cls, n):\n        return n\n\n\ndef __getattr__(name):\n    return _Any(name, (), {})\n"
for mod, files in {"google.generativeai": ["google/generativeai/__init__.py", "google/generativeai/types.py"],
                   "fireworks.client": ["fireworks/__init__.py", "fireworks/client.py"],
                   "litellm": ["litellm/__init__.py"], "anthropic": ["anthropic/__init__.py"],
                   "openai": ["openai/__init__.py"], "dotenv": ["dotenv/__init__.py"]}.items():
    try:
        importlib.import_module(mod)
    except Exception as e:
        print(f"UE SETUP: stub for {mod} ({e!r})")
        for f in files:
            p = os.path.join(sp, f)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(stub)
EOF
"$VPY" -c "import vllm, torch, transformers; print('UE env: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
cd "$UE/utility_analysis" && "$VPY" -c "import compute_utilities.utils, compute_utilities.compute_utilities; print('UE imports OK')"
