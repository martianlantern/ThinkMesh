import json
import typer
from thinkmesh import think, ThinkConfig, ModelSpec, StrategySpec

app = typer.Typer()

@app.command("think")
def think_cmd(prompt: str, backend: str = typer.Option("transformers", "--backend","-b"),
              model: str = typer.Option("Qwen2.5-7B-Instruct", "--model","-m"),
              strategy: str = typer.Option("deepconf","--strategy","-s"),
              parallel: int = typer.Option(8,"--parallel","-p"),
              max_steps: int = typer.Option(2,"--max-steps"),
              max_tokens: int = typer.Option(256,"--max-tokens"),
              temperature: float = typer.Option(0.7,"--temperature","-t")):
    cfg = ThinkConfig(
        model=ModelSpec(backend=backend, model_name=model, max_tokens=max_tokens, temperature=temperature),
        strategy=StrategySpec(name=strategy, parallel=parallel, max_steps=max_steps),
        reducer={"name":"majority"},
        budgets={"wall_clock_s":30,"tokens":8000},
    )
    ans = think(prompt, cfg)
    print(json.dumps({"content": ans.content, "confidence": ans.confidence, "meta": ans.meta}, ensure_ascii=False))

def app():
    typer.run(think_cmd)
