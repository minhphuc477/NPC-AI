"""Annotation tool: minimal Flask app and DB helpers for annotation tasks."""
from typing import Optional, Dict, Any, List
from flask import Flask, request, jsonify, session, render_template
import os
import sqlite3
import json
import csv
from pathlib import Path
from datetime import datetime


def create_app(db_path: str = ":memory:") -> Flask:
    """Create and configure the Flask app instance."""
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    secret = os.environ.get("ANNOTATION_SECRET")
    if secret is None:
        # fallback only acceptable in development mode
        if os.environ.get("FLASK_ENV") == "development":
            secret = "dev-annotation-secret"
        else:
            raise RuntimeError("ANNOTATION_SECRET not set in production environment")
    app.secret_key = secret
    app.config["DB_PATH"] = db_path

    db = AnnotationDatabase(db_path)
    db.init_db()

    task_gen = AnnotationTaskGenerator(db)

    @app.route("/")
    def index():
        # template will be created by create_html_template()
        return render_template("index.html")

    @app.route("/login", methods=["POST"])
    def login():
        data = request.json or {}
        user = data.get("user")
        pwd = data.get("pwd")
        # For the minimal app, accept any non-empty user/pwd combination
        if not user or not pwd:
            return jsonify({"error": "user and pwd required"}), 400
        session["user"] = user
        return jsonify({"status": "ok", "user": user})

    @app.route("/task", methods=["GET"])
    def task():
        user = session.get("user")
        if not user:
            return jsonify({"error": "not logged in"}), 403
        t = task_gen.get_next_task_for(user)
        if t is None:
            return jsonify({"task": None, "msg": "no tasks available"})
        return jsonify({"task": t})

    @app.route("/submit", methods=["POST"])
    def submit():
        user = session.get("user")
        if not user:
            return jsonify({"error": "not logged in"}), 403
        data = request.json or {}
        task_id = data.get("task_id")
        dialogue_acts = data.get("dialogue_acts")
        comment = data.get("comment")
        if task_id is None or dialogue_acts is None:
            return jsonify({"error": "task_id and dialogue_acts required"}), 400
        # ensure dialogue_acts is JSON-serializable
        dialogue_json = json.dumps(dialogue_acts, ensure_ascii=False)
        db.save_annotation(task_id=task_id, user=user, dialogue_acts=dialogue_json, comment=comment)
        return jsonify({"status": "saved"})

    @app.route("/stats", methods=["GET"])
    def stats():
        s = db.get_stats()
        return jsonify(s)

    # attach useful objects for tests
    app.annotation_db = db
    app.task_gen = task_gen
    return app


def create_html_template(output_dir: Optional[str] = None):
    """Write a minimal index.html template to templates directory.

    This is called by tests or dev setup to ensure a template exists.
    """
    base = Path(__file__).parent
    templates = base / "templates"
    templates.mkdir(exist_ok=True)
    html_path = templates / "index.html"
    content = """<!doctype html><html><head><meta charset='utf-8'><title>Annotation</title></head><body><h1>Annotation App</h1></body></html>"""
    html_path.write_text(content, encoding="utf-8")
    return html_path


class AnnotationDatabase:
    """Simple wrapper around SQLite storage for tasks and annotations."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        # For in-memory DB we create a persistent connection so that schema and data
        # persist across logical operations that open new connections.
        self._persistent_conn: Optional[sqlite3.Connection] = None

    def _connect(self):
        # If we have an explicitly kept persistent connection, return it
        if self._persistent_conn is not None:
            return self._persistent_conn

        # For an in-memory DB, create a named shared-in-memory DB and keep the
        # connection open via _persistent_conn for the object's lifetime.
        if self.db_path == ":memory:":
            uri = f"file:memdb_{id(self)}?mode=memory&cache=shared"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            self._persistent_conn = conn
            return conn

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        conn = self._connect()
        cur = conn.cursor()
        # create tables; avoid inline comments in SQL
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                assigned_to TEXT DEFAULT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                user TEXT NOT NULL,
                dialogue_acts TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        # If we created a persistent connection for in-memory DB, keep it open
        if self._persistent_conn is None:
            conn.close()

    def add_task(self, text: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("INSERT INTO tasks (text) VALUES (?)", (text,))
        conn.commit()
        if self._persistent_conn is None:
            conn.close()

    def task_exists(self, text: str) -> bool:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM tasks WHERE text = ? LIMIT 1", (text,))
        exists = cur.fetchone() is not None
        if self._persistent_conn is None:
            conn.close()
        return exists

    def get_unassigned_task(self) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id, text FROM tasks WHERE assigned_to IS NULL LIMIT 1")
        row = cur.fetchone()
        if row is None:
            if self._persistent_conn is None:
                conn.close()
            return None
        # mark assigned_to NULL now; actual assignment handled by app logic
        task = {"id": row["id"], "text": row["text"]}
        if self._persistent_conn is None:
            conn.close()
        return task

    def save_annotation(self, task_id: int, user: str, dialogue_acts: str, comment: Optional[str] = None):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO annotations (task_id, user, dialogue_acts, comment, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, user, dialogue_acts, comment, datetime.utcnow().isoformat()),
        )
        conn.commit()
        if self._persistent_conn is None:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as c FROM tasks")
        total_tasks = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM annotations")
        total_annotations = cur.fetchone()["c"]
        cur.execute("SELECT user, COUNT(*) as c FROM annotations GROUP BY user")
        per_user = {row["user"]: row["c"] for row in cur.fetchall()}
        conn.close()
        return {"total_tasks": total_tasks, "total_annotations": total_annotations, "per_user": per_user}


class AnnotationTaskGenerator:
    """Generate tasks for annotators from tasks table."""

    def __init__(self, db: AnnotationDatabase):
        self.db = db

    def get_next_task_for(self, user: str) -> Optional[Dict[str, Any]]:
        task = self.db.get_unassigned_task()
        if task is None:
            return None
        # In a real app we'd update assigned_to, for tests we just return the task
        return task

    def load_tasks_from_csv(self, csv_path: str, assign_state: bool = False) -> int:
        """Load tasks from a CSV file into the tasks table.

        Heuristics:
        - Accepts headers case-insensitively: instruction, input, output, optional npc_state
        - Skips rows missing instruction or output
        - Prevents duplicates both against the existing DB and within the CSV
        - If assign_state is True and an `npc_state` column exists, includes it in the task text
        Returns the number of tasks added.
        """
        added = 0
        seen = set()
        with open(csv_path, "r", encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                row_ci = {k.lower().strip(): (v.strip() if v is not None else "") for k, v in row.items()}
                instr = row_ci.get("instruction") or row_ci.get("prompt") or row_ci.get("context") or ""
                output = row_ci.get("output") or row_ci.get("response") or row_ci.get("utterance") or ""
                inp = row_ci.get("input") or ""
                npc_state = row_ci.get("npc_state") or None
                if not instr or not output:
                    continue
                base_text = instr
                if inp:
                    base_text = f"{base_text} {inp}".strip()
                if assign_state and npc_state:
                    task_text = f"[state: {npc_state}]\n{base_text}\n\nNPC: {output}"
                else:
                    task_text = f"{base_text}\n\nNPC: {output}"
                # normalize fingerprint to prevent duplicates
                fingerprint = "::".join([instr.strip().lower(), output.strip().lower()])
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                # check DB to prevent duplicates
                if self.db.task_exists(task_text):
                    continue
                self.db.add_task(task_text)
                added += 1
        return added


def create_sample_tasks(db: AnnotationDatabase, count: int = 5):
    """Populate the DB with a small set of sample tasks for quick start."""
    for i in range(count):
        db.add_task(f"Sample task {i+1}: Does the gatekeeper allow entry without a pass?")


if __name__ == "__main__":
    # Quick CLI to start dev server if executed as a module
    app = create_app()
    print("Starting dev server on http://127.0.0.1:5000 (use Flask run for production)")
    app.run(debug=(os.environ.get("FLASK_ENV") == "development"))
