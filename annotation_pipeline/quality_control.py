"""Quality control tools: Inter-annotator agreement and quality checks."""
from typing import Dict, Any, List, Tuple
import sqlite3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import os


# Inter-annotator agreement (IAA) utilities.
# References: Cohen (1960) for Kappa and Artstein & Poesio (2008) for IAA best practices.
class InterAnnotatorAgreement:
    """Compute pairwise agreement scores between annotators.

    The class expects a list of annotation rows with keys: task_id, user,
    dialogue_acts (JSON string).
    """

    def __init__(self, annotations: List[Dict[str, Any]]):
        self.annotations = annotations

    def _parse_acts(self, ann) -> List[str]:
        try:
            acts = json.loads(ann["dialogue_acts"]) if isinstance(ann["dialogue_acts"], str) else ann["dialogue_acts"]
        except Exception:
            return []
        # Normalize to a list of act labels
        if isinstance(acts, dict):
            # try common structure {"acts": [..]}
            acts = acts.get("acts", [])
        if not isinstance(acts, list):
            return []
        return [str(a) for a in acts]

    def pairwise_kappa(self) -> Dict[Tuple[str, str], float]:
        # build per-task per-user act lists
        per_task: Dict[int, Dict[str, List[str]]] = {}
        for ann in self.annotations:
            tid = int(ann["task_id"])
            per_task.setdefault(tid, {})[ann["user"]] = self._parse_acts(ann)

        # collect kappa per pair by flattening acts into single label per annotator per task
        users = sorted({ann["user"] for ann in self.annotations})
        pairs_kappa: Dict[Tuple[str, str], float] = {}
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u1 = users[i]
                u2 = users[j]
                labels1 = []
                labels2 = []
                for tid, mapping in per_task.items():
                    if u1 in mapping and u2 in mapping:
                        a1 = mapping[u1]
                        a2 = mapping[u2]
                        # collapse list of acts into a canonical string (sorted)
                        labels1.append(",".join(sorted(a1)) if a1 else "")
                        labels2.append(",".join(sorted(a2)) if a2 else "")
                if labels1 and labels2:
                    try:
                        kappa = cohen_kappa_score(labels1, labels2)
                    except Exception:
                        kappa = float("nan")
                else:
                    kappa = float("nan")
                pairs_kappa[(u1, u2)] = kappa
        return pairs_kappa


class AnnotationQualityController:
    """Controller to run QA procedures and produce reports."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_annotations(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT task_id, user, dialogue_acts FROM annotations")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def analyze(self) -> Dict[str, Any]:
        anns = self.load_annotations()
        if not anns:
            return {"num_annotations": 0, "per_user": {}, "pairwise_kappa": {}}
        # per-user counts
        per_user = {}
        for a in anns:
            per_user[a["user"]] = per_user.get(a["user"], 0) + 1
        iaa = InterAnnotatorAgreement(anns)
        pairwise = iaa.pairwise_kappa()
        return {"num_annotations": len(anns), "per_user": per_user, "pairwise_kappa": pairwise}

    def generate_iaa_report(self, output_path: str = "iaa_report.md") -> Dict[str, Any]:
        summary = self.analyze()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # write markdown summary
        lines = ["# IAA Report", ""]
        lines.append(f"Total annotations: {summary.get('num_annotations', 0)}")
        lines.append("## Per user counts")
        for u, c in summary.get("per_user", {}).items():
            lines.append(f"- {u}: {c}")
        lines.append("## Pairwise kappa")
        for (u1, u2), k in summary.get("pairwise_kappa", {}).items():
            lines.append(f"- {u1} / {u2}: {k}")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        # create visuals
        self.create_visualizations(summary)
        return summary

    def create_visualizations(self, summary: Dict[str, Any], out_dir: str = "plots"):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)

        per_user = summary.get("per_user", {})
        if per_user:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=list(per_user.keys()), y=list(per_user.values()), ax=ax)
            ax.set_title("Annotations per user")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fpath = p / "per_user_counts.png"
            fig.savefig(str(fpath))
            plt.close(fig)

        pairwise = summary.get("pairwise_kappa", {})
        if pairwise:
            # build matrix
            users = sorted({u for pair in pairwise.keys() for u in pair})
            matrix = np.full((len(users), len(users)), np.nan)
            user_index = {u: i for i, u in enumerate(users)}
            for (u1, u2), k in pairwise.items():
                i = user_index[u1]
                j = user_index[u2]
                matrix[i, j] = matrix[j, i] = k
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(matrix, annot=True, xticklabels=users, yticklabels=users, ax=ax, cmap="vlag", vmin=-1, vmax=1)
            ax.set_title("Pairwise Kappa")
            fig.tight_layout()
            fpath = p / "pairwise_kappa.png"
            fig.savefig(str(fpath))
            plt.close(fig)

    def create_sample_database(self, db_path: str = "sample_annotations.db") -> str:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS annotations (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, user TEXT, dialogue_acts TEXT)"
        )
        sample = [
            (1, "alice", json.dumps({"acts": ["refuse"]}, ensure_ascii=False)),
            (1, "bob", json.dumps({"acts": ["refuse"]}, ensure_ascii=False)),
            (2, "alice", json.dumps({"acts": ["offer"]}, ensure_ascii=False)),
            (2, "bob", json.dumps({"acts": ["inform"]}, ensure_ascii=False)),
            (3, "alice", json.dumps({"acts": ["offer", "inform"]}, ensure_ascii=False)),
        ]
        cur.executemany("INSERT INTO annotations (task_id, user, dialogue_acts) VALUES (?, ?, ?)", sample)
        conn.commit()
        conn.close()
        return db_path
