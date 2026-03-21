import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime

from tqdm import tqdm

from benchmark.client import BenchmarkClient
from benchmark.config import load_config, validate_config
from benchmark.datasets import DATASET_REGISTRY
from benchmark.datasets.base import Sample
from benchmark.evaluators import EVALUATOR_REGISTRY

# Force dataset/evaluator registration by importing all modules
import benchmark.datasets.kmmlu
import benchmark.datasets.korquad
import benchmark.datasets.gsm8k
import benchmark.datasets.summarization
import benchmark.datasets.docvqa
import benchmark.evaluators.exact_match
import benchmark.evaluators.f1_em
import benchmark.evaluators.anls
import benchmark.evaluators.llm_judge


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


class BenchmarkRunner:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        validate_config(self.config)
        self.settings = self.config["settings"]
        os.makedirs(self.settings["results_dir"], exist_ok=True)

    def _filter_models(self, vision_only: bool) -> list[dict]:
        if vision_only:
            return [m for m in self.config["models"] if m.get("vision", False)]
        return list(self.config["models"])

    async def _run_model_on_samples(
        self, client: BenchmarkClient, model_name: str, samples: list[Sample],
        temperature: float,
    ) -> list[str]:
        tasks = []
        for sample in samples:
            if isinstance(sample.prompt, list):
                messages = [{"role": "user", "content": sample.prompt}]
            else:
                messages = [{"role": "user", "content": sample.prompt}]
            tasks.append(client.generate(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.settings["max_tokens"],
            ))

        predictions = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"  {model_name}", leave=False):
            try:
                result = await coro
                predictions.append(result)
            except Exception as e:
                predictions.append(f"ERROR: {e}")

        return predictions

    async def _evaluate(
        self, ds_name: str, ds_config: dict, evaluator_name: str,
        predictions: list[str], samples: list[Sample],
        judge_client: BenchmarkClient | None = None,
    ) -> dict:
        evaluator_cls = EVALUATOR_REGISTRY[evaluator_name]
        evaluator = evaluator_cls()

        references = [s.reference for s in samples]

        if evaluator_name == "llm_judge":
            sources = [s.metadata.get("source", "") for s in samples]
            return await evaluator.evaluate_async(
                client=judge_client,
                judge_model=self.config["judge"]["model_id"],
                predictions=predictions,
                references=references,
                sources=sources,
            )
        elif evaluator_name == "anls":
            threshold = ds_config.get("anls_judge_threshold", 0.5)
            result = evaluator.evaluate(predictions, references, threshold=threshold)

            if judge_client and threshold > 0:
                low_indices = [
                    i for i, d in enumerate(result["details"])
                    if d["anls"] < threshold
                ]
                if low_indices:
                    judge_eval = EVALUATOR_REGISTRY["llm_judge"]()
                    low_preds = [predictions[i] for i in low_indices]
                    low_refs = [references[i] for i in low_indices]
                    low_sources = ["(document image)" for _ in low_indices]
                    judge_result = await judge_eval.evaluate_async(
                        client=judge_client,
                        judge_model=self.config["judge"]["model_id"],
                        predictions=low_preds,
                        references=low_refs,
                        sources=low_sources,
                    )
                    result["judge_fallback"] = {
                        "count": len(low_indices),
                        "mean_score": judge_result["mean_score"],
                    }
            return result
        else:
            return evaluator.evaluate(predictions, references)

    async def run(
        self, dataset_filter: list[str] | None = None,
        model_filter: list[str] | None = None,
        sample_override: int | None = None,
    ):
        results = {}
        timestamp = datetime.now().isoformat(timespec="seconds")
        temperatures = self.settings["temperatures"]
        stochastic_runs = self.settings["stochastic_runs"]

        # --- Header ---
        enabled_ds = [k for k, v in self.config["datasets"].items()
                      if v.get("enabled") and (not dataset_filter or k in dataset_filter)]
        model_names = [m["name"] for m in self.config["models"]]
        if model_filter:
            model_names = [n for n in model_names if n in model_filter]

        print("=" * 60)
        print("  LLM Benchmark Runner")
        print("=" * 60)
        print(f"  Models:       {', '.join(model_names)}")
        print(f"  Datasets:     {', '.join(enabled_ds)}")
        print(f"  Temperatures: {temperatures}")
        print(f"  Samples:      {sample_override or 'per-dataset config'}")
        print(f"  Stochastic:   {stochastic_runs} runs (for t>0)")
        print(f"  Concurrency:  {self.settings['concurrent_requests']}")
        print("=" * 60)

        judge_client = None
        judge_cfg = self.config.get("judge", {})
        if judge_cfg.get("base_url"):
            judge_client = BenchmarkClient(
                base_url=judge_cfg["base_url"],
                api_key=judge_cfg.get("api_key", ""),
                timeout=self.settings["timeout"],
                max_concurrent=2,
            )
            print(f"  Judge:        {judge_cfg['model_id']} @ {judge_cfg['base_url']}")
        else:
            print("  Judge:        (not configured)")
        print("=" * 60)

        run_start = time.time()

        for ds_name, ds_config in self.config["datasets"].items():
            if not ds_config.get("enabled", False):
                continue
            if dataset_filter and ds_name not in dataset_filter:
                continue

            n_samples = sample_override or ds_config.get("samples", 100)
            evaluator_name = ds_config["evaluator"]
            vision_only = ds_config.get("vision_only", False)

            print(f"\n{'─' * 60}")
            print(f"  [{ds_name.upper()}]  samples={n_samples}  evaluator={evaluator_name}"
                  f"{'  (vision-only)' if vision_only else ''}")
            print(f"{'─' * 60}")

            ds_cls = DATASET_REGISTRY.get(ds_name)
            if not ds_cls:
                print(f"  ⚠ dataset '{ds_name}' not registered, skipping")
                continue

            dataset = ds_cls(ds_config)
            print(f"  Loading dataset...", end=" ", flush=True)
            ds_load_start = time.time()
            samples = dataset.load_samples(n=n_samples, seed=42)
            print(f"done ({_fmt_duration(time.time() - ds_load_start)}, {len(samples)} samples)")

            models = self._filter_models(vision_only)
            if model_filter:
                models = [m for m in models if m["name"] in model_filter]

            results[ds_name] = {}

            for mi, model in enumerate(models):
                model_name = model["name"]
                client = BenchmarkClient(
                    base_url=model["base_url"],
                    api_key="",
                    timeout=self.settings["timeout"],
                    max_concurrent=self.settings["concurrent_requests"],
                )

                results[ds_name][model_name] = {}

                for temp in temperatures:
                    runs_needed = 1 if temp == 0.0 else stochastic_runs

                    if runs_needed == 1:
                        print(f"\n  [{mi+1}/{len(models)}] {model_name}  t={temp}")
                        t0 = time.time()
                        predictions = await self._run_model_on_samples(
                            client, model_name, samples, temp,
                        )
                        infer_time = time.time() - t0
                        error_count = sum(1 for p in predictions if p.startswith("ERROR:"))
                        print(f"  Inference done: {_fmt_duration(infer_time)}"
                              f"  ({len(predictions) - error_count}/{len(predictions)} ok"
                              f"{f', {error_count} errors' if error_count else ''})")

                        print(f"  Evaluating ({evaluator_name})...", end=" ", flush=True)
                        eval_result = await self._evaluate(
                            ds_name, ds_config, ds_config["evaluator"],
                            predictions, samples, judge_client,
                        )
                        print("done")
                        results[ds_name][model_name][f"temperature_{temp}"] = eval_result
                        self._print_result(ds_name, model_name, temp, eval_result)
                    else:
                        print(f"\n  [{mi+1}/{len(models)}] {model_name}  t={temp}  ({runs_needed} runs)")
                        run_results = []
                        for run_i in range(runs_needed):
                            print(f"    Run {run_i+1}/{runs_needed}...", flush=True)
                            t0 = time.time()
                            predictions = await self._run_model_on_samples(
                                client, model_name, samples, temp,
                            )
                            infer_time = time.time() - t0
                            error_count = sum(1 for p in predictions if p.startswith("ERROR:"))
                            print(f"    Inference: {_fmt_duration(infer_time)}"
                                  f"  ({len(predictions) - error_count}/{len(predictions)} ok"
                                  f"{f', {error_count} errors' if error_count else ''})")

                            eval_result = await self._evaluate(
                                ds_name, ds_config, ds_config["evaluator"],
                                predictions, samples, judge_client,
                            )
                            run_results.append(eval_result)
                            # Show per-run score
                            sk = self._find_score_key(eval_result)
                            print(f"    Score: {eval_result[sk]:.1%}")

                        agg = self._aggregate_runs(run_results)
                        results[ds_name][model_name][f"temperature_{temp}"] = agg
                        self._print_result(ds_name, model_name, temp, agg)

        total_time = time.time() - run_start

        # --- Summary table ---
        print(f"\n{'=' * 60}")
        print("  RESULTS SUMMARY")
        print(f"{'=' * 60}")
        for ds_name, ds_results in results.items():
            print(f"\n  [{ds_name.upper()}]")
            for model_name, temp_results in ds_results.items():
                for temp_key, result in temp_results.items():
                    temp = temp_key.replace("temperature_", "t=")
                    self._print_result_summary(model_name, temp, result)
        print(f"\n{'─' * 60}")
        print(f"  Total time: {_fmt_duration(total_time)}")

        output = {
            "timestamp": timestamp,
            "config_snapshot": self.config,
            "results": results,
        }
        filename = f"{timestamp.replace(':', '-')}_benchmark.json"
        filepath = os.path.join(self.settings["results_dir"], filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  Results saved: {filepath}")
        print(f"{'=' * 60}")

    def _aggregate_runs(self, run_results: list[dict]) -> dict:
        score_key = self._find_score_key(run_results[0])
        scores = [r[score_key] for r in run_results]
        return {
            "mean_score": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "runs": scores,
            "total": run_results[0].get("total", 0),
            "errors": sum(r.get("errors", 0) for r in run_results),
        }

    def _find_score_key(self, result: dict) -> str:
        for key in ("score", "f1", "anls", "mean_score"):
            if key in result:
                return key
        return "score"

    def _format_score(self, result: dict) -> str:
        score_key = self._find_score_key(result)
        errors = result.get("errors", 0)
        total = result.get("total", 0)
        error_str = f", {errors} err" if errors else ""

        if "std" in result:
            score = result["mean_score"]
            std = result["std"]
            return f"{score:.1%} (±{std:.1%}{error_str})"
        elif score_key == "f1":
            f1 = result.get("f1", 0)
            em = result.get("em", 0)
            return f"F1={f1:.1%} EM={em:.1%}{error_str}"
        elif score_key == "mean_score":
            score = result["mean_score"]
            return f"{score:.1f}/5.0 (judge{error_str})"
        elif score_key == "anls":
            anls = result.get("anls", 0)
            return f"{anls:.1%} ANLS{error_str}"
        else:
            score = result.get("score", 0)
            correct = result.get("correct", 0)
            return f"{score:.1%} ({correct}/{total}{error_str})"

    def _print_result(self, ds_name: str, model_name: str, temp: float, result: dict):
        score_str = self._format_score(result)
        print(f"  >> {model_name:<25} t={temp}: {score_str}")

    def _print_result_summary(self, model_name: str, temp: str, result: dict):
        score_str = self._format_score(result)
        print(f"    {model_name:<25} {temp:<8} {score_str}")


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Runner")
    parser.add_argument(
        "--config", default="benchmark/config.yaml",
        help="Path to benchmark config (default: benchmark/config.yaml)",
    )
    parser.add_argument("--dataset", nargs="+", help="Run only these datasets")
    parser.add_argument("--model", nargs="+", help="Run only these models")
    parser.add_argument("--samples", type=int, help="Override sample count")
    args = parser.parse_args()

    runner = BenchmarkRunner(args.config)
    asyncio.run(runner.run(
        dataset_filter=args.dataset,
        model_filter=args.model,
        sample_override=args.samples,
    ))


if __name__ == "__main__":
    main()
