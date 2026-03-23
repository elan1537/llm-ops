"""RULER-style long context benchmark — Needle in a Haystack.

Generates synthetic long contexts with hidden "needles" (key facts) at various
positions. Tests whether the model can retrieve information from different
depths of the context window.

Supports multiple context lengths to map model's effective context range.
"""

import random
import string

from benchmark.datasets import register_dataset
from benchmark.datasets.base import BaseDataset, Sample

# Filler text paragraphs (boring, repetitive content to pad the context)
FILLER_PARAGRAPHS = [
    "The weather in the region has been mostly stable throughout the season. Temperatures remained within the expected range, with occasional fluctuations due to passing weather systems. Meteorological stations across the area reported consistent readings that aligned with historical averages for this time of year.",
    "Urban development projects continued to progress according to their planned schedules. Construction teams worked diligently to meet the established milestones. Regular inspections confirmed that building standards were being maintained across all active project sites.",
    "The local transportation network experienced typical usage patterns during the reporting period. Bus routes operated on their regular schedules, and railway services maintained their standard timetables. Traffic flow analysis showed no significant deviations from expected volumes.",
    "Agricultural activities in the surrounding areas followed seasonal patterns. Farmers reported standard crop growth conditions, with soil moisture levels remaining adequate for the current stage of the growing cycle. Supply chain logistics for agricultural products operated without notable disruptions.",
    "Community events and cultural activities took place as planned throughout the period. Local organizations hosted their scheduled gatherings, and public facilities maintained their regular operating hours. Attendance figures for various events were consistent with expectations.",
    "Financial markets showed typical trading patterns during the observed period. Market indices fluctuated within normal ranges, reflecting standard economic conditions. Banking institutions reported routine transaction volumes across their operations.",
    "Educational institutions operated according to their academic calendars. Student enrollment numbers remained stable, and academic programs proceeded as outlined in their curricula. Faculty and staff maintained their regular teaching and administrative duties.",
    "Healthcare facilities continued to provide services at their standard capacity. Patient volumes were within expected ranges, and medical staff maintained their regular schedules. Routine maintenance of medical equipment was carried out as planned.",
]

# Needles — the hidden facts to find
NEEDLE_TEMPLATES = [
    {
        "needle": "The secret password for Project Aurora is 'crystalline-butterfly-7429'.",
        "question": "What is the secret password for Project Aurora?",
        "answer": "crystalline-butterfly-7429",
    },
    {
        "needle": "Dr. Elena Vasquez discovered that the optimal temperature for the experiment is exactly 42.7 degrees Celsius.",
        "question": "What is the optimal temperature for the experiment discovered by Dr. Vasquez?",
        "answer": "42.7 degrees Celsius",
    },
    {
        "needle": "The annual budget allocated for the Mars exploration program is $3.2 billion USD.",
        "question": "What is the annual budget for the Mars exploration program?",
        "answer": "$3.2 billion USD",
    },
    {
        "needle": "According to internal memo #4872, the product launch date has been moved to September 15, 2025.",
        "question": "What is the new product launch date according to internal memo #4872?",
        "answer": "September 15, 2025",
    },
    {
        "needle": "The winning lottery numbers for the special drawing were 7, 14, 23, 38, 42, and bonus number 11.",
        "question": "What were the winning lottery numbers for the special drawing?",
        "answer": "7, 14, 23, 38, 42, and bonus number 11",
    },
]


def _generate_filler(target_chars: int, seed: int) -> list[str]:
    """Generate filler paragraphs to reach target character count."""
    rng = random.Random(seed)
    paragraphs = []
    total = 0
    while total < target_chars:
        p = rng.choice(FILLER_PARAGRAPHS)
        paragraphs.append(p)
        total += len(p) + 2  # +2 for newlines
    return paragraphs


def _build_context(needle: str, target_chars: int, position: float, seed: int) -> str:
    """Build a long context with a needle hidden at the specified position (0.0~1.0)."""
    paragraphs = _generate_filler(target_chars, seed)
    insert_idx = max(1, int(len(paragraphs) * position))
    paragraphs.insert(insert_idx, needle)
    return "\n\n".join(paragraphs)


@register_dataset("ruler")
class RulerDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "ruler"

    def load_samples(self, n: int, seed: int = 42) -> list[Sample]:
        # Context lengths to test (in characters, ~4 chars per token)
        context_lengths = self.config.get("context_lengths", [4000, 16000, 64000, 128000])
        # Needle positions: beginning (0.1), middle (0.5), end (0.9)
        positions = self.config.get("positions", [0.1, 0.5, 0.9])

        rng = random.Random(seed)
        samples = []

        for length in context_lengths:
            for pos in positions:
                needle_info = rng.choice(NEEDLE_TEMPLATES)

                context = _build_context(
                    needle=needle_info["needle"],
                    target_chars=length,
                    position=pos,
                    seed=rng.randint(0, 100000),
                )

                prompt = (
                    f"Read the following document carefully and answer the question at the end.\n\n"
                    f"{context}\n\n"
                    f"Question: {needle_info['question']}\n"
                    f"Answer concisely with only the answer, no explanation:"
                )

                pos_label = {0.1: "beginning", 0.5: "middle", 0.9: "end"}.get(pos, f"{pos:.0%}")

                samples.append(Sample(
                    id=f"ruler_{length}chars_{pos_label}",
                    prompt=prompt,
                    reference=needle_info["answer"],
                    metadata={
                        "context_chars": len(context),
                        "context_tokens_approx": len(context) // 4,
                        "needle_position": pos,
                        "position_label": pos_label,
                        "target_length": length,
                    },
                ))

                if len(samples) >= n:
                    return samples

        return samples[:n]
