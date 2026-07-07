from collections import defaultdict


class MutationStatistics:
    def __init__(self):
        self._stats = defaultdict(
            lambda: {
                "uses": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "total_delta": 0.0,
                "best_delta": float("-inf"),
            }
        )

    def update(self, strategy: str, delta: float):
        s = self._stats[strategy]

        s["uses"] += 1
        s["total_delta"] += delta

        if delta > 0:
            s["positive"] += 1
        elif delta < 0:
            s["negative"] += 1
        else:
            s["neutral"] += 1

        if delta > s["best_delta"]:
            s["best_delta"] = delta

    def average_delta(self, strategy: str):
        s = self._stats[strategy]
        if s["uses"] == 0:
            return 0.0
        return s["total_delta"] / s["uses"]

    def best_delta(self, strategy: str):
        s = self._stats[strategy]
        if s["uses"] == 0:
            return 0.0
        return s["best_delta"]

    def report(self):
        report = {}

        for strategy, s in self._stats.items():
            uses = s["uses"]
            positive = s["positive"]
            success_rate = positive / uses if uses > 0 else 0.0
            report[strategy] = {
                "uses": uses,
                "positive": positive,
                "negative": s["negative"],
                "neutral": s["neutral"],
                "success_rate": success_rate,
                "average_delta": self.average_delta(strategy),
                "best_delta": self.best_delta(strategy),
            }

        return report

    def print_report(self) -> None:
        report = self.report()
        if not report:
            return

        print("\n" + "=" * 50)
        print("Mutation Strategy Statistics")
        print("=" * 50)

        for strategy, stats in report.items():
            print()
            print(strategy)
            print(f"Uses: {stats['uses']}")
            print(f"Positive: {stats['positive']}")
            print(f"Negative: {stats['negative']}")
            print(f"Neutral: {stats['neutral']}")
            print(f"Success Rate: {stats['success_rate']*100:.1f}%")
            avg_delta = stats['average_delta']
            sign = "+" if avg_delta > 0 else ""
            print(f"Average Δ: {sign}{avg_delta:.3f}")
            best_delta = stats['best_delta']
            sign = "+" if best_delta > 0 else ""
            print(f"Best Δ: {sign}{best_delta:.3f}")
            print("-" * 34)

        print("=" * 50 + "\n")