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
            self.assertFalse(publication.inspect_payload(p.as_posix(),p.read_bytes()))

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

    def test_unapproved_team_sources_are_not_public(self):
        manifest=json.loads((ROOT/'docs/source-manifest.json').read_text(encoding='utf-8'))
        withheld=[r for r in manifest if r['publication_status'].startswith('withheld')]
        self.assertEqual(len(withheld),5)
        for name in ['infertility-prediction','smartdur-medication-alert']:
            self.assertEqual(list((ROOT/'projects'/name).rglob('*.ipynb')),[])
            self.assertEqual(list((ROOT/'projects'/name).rglob('*.py')),[])


if __name__ == '__main__':
    unittest.main()
