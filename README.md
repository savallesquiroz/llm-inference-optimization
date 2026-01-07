# LLM Inference Optimization (llama.cpp + Phi-2)

CPU-only benchmarking of llama.cpp on Microsoft Phi-2 to quantify the impact of quantization, threading, context length, and memory mapping. The suite automates runs, parses tokens/sec and timing metrics, and produces CSVs and a markdown summary.

## Repository Layout
- scripts/script.py — benchmark driver (runs all experiments sequentially)
- models/ — place GGUF models (e.g., phi-2.Q4_K_M.gguf)
- results/ — CSV outputs, logs/, debug/, and auto-generated summary_report.md
- llama.cpp/ — upstream inference runtime (build here to produce llama-cli.exe)
- report/ — LaTeX report and slides (optional deliverables)

## Prerequisites
- Windows with MSVC Build Tools and CMake (to build llama.cpp)
- Python 3.10+; optional `psutil` for peak RSS sampling
- GGUF models (phi-2.*.gguf) stored under models/

## Setup
1) Build llama.cpp (Release)
```powershell
Push-Location "llama.cpp"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
Pop-Location
```

2) (Optional) Python virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install psutil
```

3) Models
- Place your GGUF files in models/ (e.g., phi-2.Q4_K_M.gguf, phi-2.Q5_K_M.gguf, phi-2.Q8_0.gguf).
- The script searches for llama-cli.exe under llama.cpp/build/bin/Release by default.

## Running the Benchmark Suite
```powershell
python scripts\script.py
```
The suite runs, in order:
1) Quantization comparison (Q4_K_M vs Q5_K_M vs Q8_0, 64 tokens, 4 threads)
2) Context-length scaling (32–1024 tokens, 4 threads)
3) Thread scaling (1–16 threads, 3 runs each, 64 tokens)
4) Memory mapping on/off (64 tokens, 4 threads)
5) Model size comparison placeholder (Phi-2 2.7B)

Outputs (results/):
- quantization_comparison.csv
- context_length_scaling.csv
- thread_scaling_detailed.csv
- memory_configuration.csv
- model_size_comparison.csv
- summary_report.md (averages + next steps)
- logs/ per-run stdout; debug/ captures zero-metric runs with command context

## Customization
- Update the paths at the bottom of scripts/script.py to point to your llama.cpp build and models directory.
- Change prompts, token counts, thread lists, or timeouts by editing the corresponding experiment methods.
- `psutil` is optional; without it, memory_mb is reported as 0.
- To avoid concurrent runs, the script uses simple lock files under results/locks/.

## Notes
- All runs use the prompt: "The quick brown fox jumps over the lazy dog." with `--gpu-layers 0` to force CPU.
- Ensure enough disk space for logs; large models are .gitignored by default.
- If llama-cli.exe is not found, the script will warn and skip runs until the path is corrected.
