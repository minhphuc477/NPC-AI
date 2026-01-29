# Annotation Pipeline (NPC AI)

Quick-start instructions:

- Install minimal dependencies: pip install -r requirements.txt

- Generate guidelines (creates `guidelines/`):

  python -m annotation_guidelines

- Create sample tasks and start dev Flask app:

  from annotation_pipeline.annotation_tool import create_app, create_sample_tasks
  app = create_app(db_path=":memory:")
  create_sample_tasks(app.annotation_db)
  # Use Flask test client or run with FLASK_ENV=development and ANNOTATION_SECRET set

- Convert CSV to JSONL:

  python convert_csv_to_jsonl.py data/gatekeeper_dataset.csv data/train.jsonl

  # write to stdout (helpful for streaming/pipeline):
  python convert_csv_to_jsonl.py data/gatekeeper_dataset.csv -

Notes on CSV format:
- Expected headers (case-insensitive): `instruction`, `input`, `output`. An optional `npc_state` column is recognized.
- If `npc_state` is present and you use the annotation loader with `assign_state=True`, the value will be included in the generated task text as a state override (e.g. `[state: angry]`), which can influence annotation context.

- Run quality control sample and report:

  from annotation_pipeline.quality_control import AnnotationQualityController
  qc = AnnotationQualityController()
  db = qc.create_sample_database(db_path="sample_annotations.db")
  qc.generate_iaa_report("iaa_report.md")

- Run tests:

  pytest -q

Notes:
- Set ANNOTATION_SECRET environment variable in production; the app refuses to start without it unless FLASK_ENV is "development".
- The simple Flask app is minimal and intended for local/offline use or as a reference implementation.
