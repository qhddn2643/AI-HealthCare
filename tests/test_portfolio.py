"""Small safety/portability checks without model fitting or private data."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication', ROOT/'tools/check_publication.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def test_credentials_are_detected_without_echoing_value(self):
        value = ('ghp_' + 'A'*36).encode()
        findings = publication.inspect_payload('note.md', value)
        self.assertTrue(any('credential' in item for item in findings))
        self.assertNotIn(value.decode(), str(findings))

    def test_raw_data_is_rejected(self):
        self.assertIn('restricted file type/name', publication.inspect_payload('data/rows.csv', b'a,b'))

    def test_real_colab_path_detected_without_scanner_self_match(self):
        path = ('/content/' + 'drive/MyDrive/private').encode()
        self.assertIn('machine-specific absolute path', publication.inspect_payload('note.md', path))
        source = (ROOT/'tools/check_publication.py').read_bytes()
        self.assertFalse(publication.inspect_payload('tools/check_publication.py', source))

    def test_notebook_output_is_rejected(self):
        nb = {'nbformat':4, 'cells':[{'cell_type':'code','source':['x=1'],
              'outputs':[{'text':['private']}],'execution_count':1}]}
        self.assertTrue(any('output' in x for x in publication.inspect_payload('a.ipynb',json.dumps(nb).encode())))

    def test_notebooks_compile_and_training_defaults_off(self):
        for p in ROOT.glob('projects/*/notebooks/*.ipynb'):
            nb=json.loads(p.read_text(encoding='utf-8'))
            text='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
            ast.parse(text)
            self.assertIn('ALLOW_TRAINING = False', text)
            self.assertIn('if not ALLOW_TRAINING:', text)
            review=json.loads((ROOT/'docs/notebook-output-review.json').read_text(encoding='utf-8'))['notebooks']
            self.assertFalse(publication.inspect_payload(p.relative_to(ROOT).as_posix(),p.read_bytes(),output_review=review))

    def test_preserved_bmi_boundaries(self):
        p=ROOT/'projects/smoking-health-eda/notebooks/analysis.ipynb'
        nb=json.loads(p.read_text(encoding='utf-8'))
        text='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='categorize_bmi')
        scope={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated BMI function>','exec'),scope)
        self.assertEqual(scope['categorize_bmi'](18.5),'정상')
        self.assertEqual(scope['categorize_bmi'](25),'1단계 비만')
        self.assertEqual(scope['categorize_bmi'](35),'3단계 비만')

    def test_input_and_generated_data_are_rejected(self):
        self.assertIn('input data directory', publication.inspect_payload('projects/demo/data/records.json', b'{}'))
        self.assertIn('generated output directory', publication.inspect_payload('projects/demo/outputs/report.html', b'<html>'))
        self.assertFalse(publication.inspect_payload('projects/demo/data/README.md', b'# Data instructions'))

    def test_all_project_sources_are_published_without_data(self):
        manifest=json.loads((ROOT/'docs/source-manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(all(r['publication_status']=='included' for r in manifest))
        self.assertEqual(len(list(ROOT.glob('projects/*/notebooks/*.ipynb'))),3)
        self.assertEqual(len(list((ROOT/'projects/smartdur-medication-alert/src').glob('*.py'))),3)
        for row in manifest:
            self.assertTrue((ROOT/row['target']).is_file(),row['target'])

    def test_all_project_modules_compile(self):
        for p in ROOT.glob('projects/*/src/*.py'):
            self.assertFalse(publication.inspect_payload(p.relative_to(ROOT).as_posix(),p.read_bytes()))

    def test_smartdur_training_is_explicitly_gated(self):
        p=ROOT/'projects/smartdur-medication-alert/src/run_analysis.py'
        tree=ast.parse(p.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        import os
        from unittest.mock import patch
        scope={'os':os,'keras':None}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated training gate>','exec'),scope)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'SMARTDUR_ALLOW_TRAINING'):
                scope['run']()

    def test_readme_links_resolve(self):
        import re
        for p in ROOT.rglob('README.md'):
            if '.git' in p.parts:
                continue
            for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
                if not link.startswith(('https:','http:','#')):
                    self.assertTrue((p.parent/link.split('#')[0]).exists(),f'{p.name}: {link}')

    def test_label_audit_requires_evidence_for_confirmed_decisions(self):
        import pandas as pd
        spec=importlib.util.spec_from_file_location('audit',ROOT/'projects/smartdur-medication-alert/src/label_audit.py')
        audit=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        # Review-state fixture only: no medical measurements or real identifiers.
        row={'case_token':'test-case','decision_status':'confirmed','y_reviewed':1,
             'reviewed_by':'','reviewed_at':'','evidence_reference':'','review_note':''}
        with self.assertRaisesRegex(ValueError,'reviewed_by'):
            audit.validate_decisions(pd.DataFrame([row]),{'test-case'})
        row.update(decision_status='pending',y_reviewed=float('nan'))
        result=audit.validate_decisions(pd.DataFrame([row]),{'test-case'})
        self.assertTrue(result.y_reviewed.isna().all())

    def test_label_audit_rejects_unknown_case_tokens(self):
        import pandas as pd
        spec=importlib.util.spec_from_file_location('audit',ROOT/'projects/smartdur-medication-alert/src/label_audit.py')
        audit=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        row={'case_token':'unknown-test-case','decision_status':'pending','y_reviewed':float('nan'),
             'reviewed_by':'','reviewed_at':'','evidence_reference':'','review_note':''}
        with self.assertRaisesRegex(ValueError,'case_token'):
            audit.validate_decisions(pd.DataFrame([row]),{'test-case'})

    def test_reviewed_outputs_are_present_and_match_approved_hashes(self):
        review=json.loads((ROOT/'docs/notebook-output-review.json').read_text(encoding='utf-8'))['notebooks']
        self.assertEqual(len(review),3)
        for name,record in review.items():
            nb=json.loads((ROOT/name).read_text(encoding='utf-8'))
            populated={c['id']:c for c in nb['cells'] if c['cell_type']=='code' and c.get('outputs')}
            self.assertEqual(set(populated),set(record['cells']))
            self.assertTrue(any('image/png' in o.get('data',{}) for c in populated.values() for o in c['outputs']))
            for key,cell in populated.items():
                self.assertEqual(publication.output_digest(cell),record['cells'][key]['sha256'])
                self.assertEqual(cell['execution_count'],record['cells'][key]['execution_count'])

    def test_modified_output_cannot_reuse_review_approval(self):
        import copy
        review=json.loads((ROOT/'docs/notebook-output-review.json').read_text(encoding='utf-8'))['notebooks']
        name=next(iter(review))
        nb=json.loads((ROOT/name).read_text(encoding='utf-8'))
        changed=copy.deepcopy(nb)
        cell=next(c for c in changed['cells'] if c['cell_type']=='code' and c.get('outputs'))
        cell['outputs'].append({'output_type':'stream','name':'stdout','text':['unreviewed row']})
        issues=publication.inspect_payload(name,json.dumps(changed).encode(),output_review=review)
        self.assertTrue(any('unreviewed notebook output' in x for x in issues))

    def test_active_html_is_rejected_even_with_matching_output_hash(self):
        cell={'id':'fixture','cell_type':'code','source':['x=1'],'execution_count':1,
              'outputs':[{'output_type':'display_data','metadata':{},'data':{'text/html':['<script>alert(1)</script>']}}]}
        nb={'nbformat':4,'cells':[cell]}
        review={'a.ipynb':{'cells':{'fixture':{'sha256':publication.output_digest(cell)}}}}
        issues=publication.inspect_payload('a.ipynb',json.dumps(nb).encode(),output_review=review)
        self.assertTrue(any('active notebook HTML' in x for x in issues))

    def test_per_row_prediction_arrays_stay_excluded(self):
        nb=json.loads((ROOT/'projects/smoking-health-eda/notebooks/analysis.ipynb').read_text(encoding='utf-8'))
        cell=next(c for c in nb['cells'] if c.get('metadata',{}).get('source_cell_index')==44)
        text=''.join(''.join(o.get('text',[])) for o in cell['outputs'])
        self.assertIn('정확도(Accuracy): 0.699',text)
        self.assertNotIn('예측값 (앞 10개):',text)
        self.assertNotIn('흡연일 확률 (앞 10개):',text)


if __name__ == '__main__':
    unittest.main()
