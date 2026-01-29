import os, tempfile
from annotation_pipeline.annotation_guidelines import AnnotationGuidelines
from annotation_pipeline.annotation_tool import create_app, create_html_template, create_sample_tasks
from annotation_pipeline.quality_control import AnnotationQualityController

print('Running smoke tests...')
# Guidelines
ag = AnnotationGuidelines()
md = ag.create_markdown_guidelines()
print('guidelines length:', len(md))
out = ag.save_guidelines(output_dir=tempfile.mkdtemp())
print('saved to', out)

# Flask
os.environ['ANNOTATION_SECRET'] = 'test-secret'
app = create_app(db_path=':memory:')
create_html_template()
client = app.test_client()
print('login status:', client.post('/login', json={'user':'u','pwd':'p'}).status_code)
create_sample_tasks(app.annotation_db, count=2)
print('task get:', client.get('/task').get_json())
ann = {'task_id': client.get('/task').get_json()['task']['id'], 'dialogue_acts': {'acts':['refuse']}}
print('submit status:', client.post('/submit', json=ann).status_code)
print('stats:', client.get('/stats').get_json())

# Quality control
qc = AnnotationQualityController()
db_file = qc.create_sample_database(db_path=tempfile.mktemp(suffix='.db'))
qc.db_path = db_file
summary = qc.generate_iaa_report(output_path=tempfile.mktemp(suffix='.md'))
print('qc summary keys:', list(summary.keys()))
print('Smoke tests finished.')
