# Pilot 4: where does Qwen3-14B compute its trade-offs?

Exploratory. Qwen3-14B's choices in pilot 3 were close to one consistent valuation; pilot 4
records the model's insides on the same prompts and looks for the value code the way
Padoa-Schioppa did in monkeys: which directions and neurons track each offer's value, the
chosen value and the choice, whether they form one common currency, and where an assigned
objective enters. **Read [`DESIGN.md`](DESIGN.md) first.**

| File | What it does |
|---|---|
| `make_prompts.py` | Builds `prompts.jsonl` (run once, here): every strangers prompt of pilot 3 plus twins without the goal line, with value labels from pilot 3's fit (`values.json`) |
| `record.py` | Runs the model once over every prompt; saves the residual stream at 10 positions, every token of 500 prompts, and running sums for every MLP neuron. `--random`: the random-weights control |
| `probes.py` | Token map, value probes, one currency or several, the objective |
| `neurons.py` | Neurons classified the monkey way, with a shuffle control |
| `compare.py` | The real model against its random-weights control |
| `steer.py` | A first causal look: pushing the value direction while the model reads option A |
| `pod.ipynb` | One cell for the pod; it beeps when done |
| `tests/make_synthetic.py` | Activations with a planted value code, to test the battery |

## On the pod

1. In RunPod, create a **network volume** of 200 GB (Secure Cloud).
2. Deploy an **RTX A6000** pod in the same data centre with that volume attached (it mounts at
   `/workspace`), PyTorch template.
3. In JupyterLab: File → New → Terminal, then
   `cd /workspace && git clone https://github.com/JasperBlank/llm-value-trade-offs.git`
4. Open `llm-value-trade-offs/pilot4-value-code/pod.ipynb` and run the cell (about 75
   minutes). At the beeps, download `/workspace/pilot4_results.tar.gz` and send it to Claude.
5. Terminate the pod. Keep the volume until we are done with follow-up rounds, then delete it.
