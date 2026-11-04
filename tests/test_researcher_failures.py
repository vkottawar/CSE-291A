"""Deterministic placement tests; replay behavior is not fault detection evidence."""
import copy
import json
import unittest

from lineage.faults import FaultInjector, FaultSpec, InjectedResearcherTimeout, load_specs
from lineage.runner import run_task
from lineage.trace import Trace
from tests.test_faults import CLEAN, ROOT, ReplayLLM, payments


def run(spec=None, llm=None):
    llm = llm or ReplayLLM(CLEAN / 'user_task_0_r1.json')
    _, trace, env = run_task('user_task_0', llm, fault=spec, return_env=True)
    return llm, trace, env


def spec(kind, **options):
    return FaultSpec(id=kind, task='user_task_0', stage='researcher',
                     step='R1', type=kind, **options)


class ResearcherFailures(unittest.TestCase):
    def test_specs_load(self):
        self.assertEqual([s.type for s in load_specs(ROOT / 'faults/researcher_failures.json')],
                         ['timeout', 'partial_output'])

    def test_no_fault_matches_committed_clean_semantic_trace(self):
        original = json.loads((CLEAN / 'user_task_0_r1.json').read_text())
        _, trace, env = run()
        self.assertEqual([s.step_id for s in trace.steps],
                         [s['step_id'] for s in original['steps']])
        for actual, expected in zip(trace.steps, original['steps']):
            with self.subTest(step=actual.step_id):
                self.assertEqual(actual.role, expected['role'])
                self.assertEqual(actual.reads, expected['reads'])
                self.assertEqual(actual.output, expected['output'])
                self.assertEqual([(c.tool,c.args,c.result,c.error) for c in actual.tool_calls],
                                 [(c['tool'],c['args'],c['result'],c['error']) for c in expected['tool_calls']])
        self.assertEqual(trace.utility, original['utility'])
        self.assertEqual([p.amount for p in payments(env)], [98.7])

    def test_timeout_before_request_aborts_without_payment(self):
        llm, trace, env = run(spec('timeout'))
        self.assertNotIn('R1', llm.turn)
        self.assertEqual([s.step_id for s in trace.steps], ['P0', 'R1'])
        r1 = trace.steps[-1]
        self.assertEqual((r1.status, r1.turns, r1.tool_calls), ('timeout', 0, []))
        self.assertGreaterEqual(r1.seconds, 0)
        self.assertIn('InjectedResearcherTimeout', trace.error)
        self.assertFalse(trace.utility)
        self.assertEqual(payments(env), [])
        [event] = trace.faults
        self.assertTrue(event['applied'])
        self.assertTrue(event['synthetic'])
        self.assertFalse(event['provider_request_sent'])

    def test_timeout_once(self):
        trace = Trace('user_task_0')
        injector = FaultInjector(spec('timeout'), trace)
        with self.assertRaises(InjectedResearcherTimeout):
            injector.before_model_call('R1', 1)
        injector.before_model_call('R1', 1)
        injector.finish()
        self.assertEqual(len(trace.faults), 1)

    def test_timeout_targets_r2_or_r3_without_other_researcher_call_failure(self):
        for step in ('R2', 'R3'):
            with self.subTest(step=step):
                llm = ReplayLLM(CLEAN / 'user_task_0_r1.json')
                llm.steps['P0']['output']['questions'] = {step:'look up required facts'}
                fault = FaultSpec(id='target', task='user_task_0', stage='researcher',
                                  step=step, type='timeout')
                _, trace, env = run(fault, llm)
                self.assertEqual(trace.steps[-1].step_id, step)
                self.assertNotIn(step, llm.turn)
                self.assertTrue(trace.faults[0]['applied'])
                self.assertEqual(payments(env), [])

    def test_model_failure_after_payment_preserves_side_effect_evidence(self):
        class FailAfterPayment(ReplayLLM):
            def chat(self, step_id, messages, tools):
                if step_id == 'E1' and self.turn.get(step_id) == 1:
                    raise TimeoutError('offline post-tool failure accounting check')
                return super().chat(step_id, messages, tools)
        _, trace, env = run(llm=FailAfterPayment(CLEAN / 'user_task_0_r1.json'))
        self.assertFalse(trace.utility)
        self.assertIn('TimeoutError', trace.error)
        self.assertEqual(trace.steps[-1].status, 'timeout')
        self.assertEqual([p.amount for p in payments(env)], [98.7])
        self.assertEqual([t['amount'] for t in trace.transaction_changes['added']], [98.7])
        self.assertEqual(trace.transaction_changes['after_count'], 6)

    def test_skipped_researcher_is_not_placed(self):
        llm = ReplayLLM(CLEAN / 'user_task_0_r1.json')
        llm.steps['P0']['output']['questions'].pop('R1')
        _, trace, _ = run(spec('timeout'), llm)
        self.assertFalse(trace.faults[0]['applied'])
        self.assertIn('R1', trace.faults[0]['reason'])

    def test_partial_is_valid_exact_deletion_and_actual_planner_payload(self):
        llm = ReplayLLM(CLEAN / 'user_task_0_r1.json')
        original = copy.deepcopy(llm.steps['R1']['output'])
        llm, trace, _ = run(spec('partial_output', fact=['bill_iban']), llm)
        r1 = next(s for s in trace.steps if s.step_id == 'R1')
        expected = copy.deepcopy(original)
        expected['facts'] = [f for f in expected['facts'] if f['name'] != 'bill_iban']
        self.assertEqual(r1.output, expected)
        self.assertEqual(json.loads(json.dumps(r1.output)), expected)
        self.assertNotIn('R1.bill_iban', llm.inputs['P']['facts'])
        planner = next(s for s in trace.steps if s.step_id == 'P')
        self.assertEqual(planner.inputs, llm.inputs['P'])
        [event] = trace.faults
        self.assertEqual((event['original'], event['injected']), (original, expected))
        self.assertEqual(event['operation'], 'drop_fact')
        self.assertTrue(event['applied'])
        # Cached outputs do not adapt: a passing replay cannot establish resilience.
        self.assertTrue(trace.utility)
        r1.output['facts'].clear()
        self.assertEqual(event['original'], original)
        self.assertEqual(event['injected'], expected)

    def test_partial_does_not_substring_match(self):
        _, trace, _ = run(spec('partial_output', fact=['iban']))
        self.assertFalse(trace.faults[0]['applied'])
        self.assertTrue(trace.utility)

    def test_residual_alias_or_other_researcher_value_blocks_planner(self):
        for sid in ('R1', 'R3'):
            with self.subTest(source=sid):
                llm = ReplayLLM(CLEAN / 'user_task_0_r1.json')
                llm.steps[sid]['output']['facts'].append({
                    'name':'other_details','value':{'text':'Recipient is UK12345678901234567890'}})
                _, trace, env = run(spec('partial_output', fact=['bill_iban']), llm)
                self.assertNotIn('P', llm.turn)
                self.assertIn('removed value remains', trace.error)
                self.assertTrue(trace.faults[0]['applied'])
                self.assertFalse(trace.faults[0]['handoff_validated'])
                self.assertTrue(trace.faults[0]['residual_value_present'])
                self.assertEqual(payments(env), [])

    def test_partial_validated_handoff_and_fresh_injector_each_run(self):
        fault = spec('partial_output', fact=['bill_iban'])
        for _ in range(2):
            llm, trace, env = run(fault)
            self.assertTrue(trace.faults[0]['handoff_validated'])
            self.assertNotIn('UK12345678901234567890', json.dumps(llm.inputs['P']))
            self.assertEqual(trace.transaction_changes['before_count'], 5)
            self.assertEqual([p.amount for p in payments(env)], [98.7])
        _, clean, _ = run()
        self.assertEqual(clean.faults, [])

    def test_residual_in_original_request_blocks_handoff(self):
        trace = Trace('user_task_0')
        injector = FaultInjector(spec('partial_output', fact=['bill_iban']), trace)
        out = {'facts':[{'name':'bill_iban','value':'UK12345678901234567890'}]}
        injector.on_researcher('R1', out)
        with self.assertRaisesRegex(ValueError,'removed value remains'):
            injector.validate_handoff({},'Please pay UK12345678901234567890')
        self.assertFalse(trace.faults[0]['handoff_validated'])

    def test_exception_after_completed_case_preserves_manifest(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from lineage import researcher_smoke
        llm, trace, env = run()
        with TemporaryDirectory() as tmp, patch('sys.argv', [
            'smoke','--env-file','unused','--output',str(Path(tmp)/'results')]), \
            patch.object(researcher_smoke,'_load_env_file'), \
            patch.dict('os.environ', {'LLM_MODEL':'fake','LLM_BASE_URL':'https://example.test/v1','LLM_API_KEY':'test'}), \
            patch.object(researcher_smoke,'LLM'), \
            patch.object(researcher_smoke,'usage',side_effect=[{'spend':0,'max_budget':50}, RuntimeError('offline check')]), \
            patch.object(researcher_smoke,'run_task',return_value=(None,trace,env)) as task, \
            patch('builtins.print'):
            researcher_smoke.main()
            manifest=json.loads((Path(tmp)/'results/manifest.json').read_text())
            self.assertEqual(task.call_count,1)
            self.assertEqual(len(manifest['runs']),1)
            self.assertEqual(manifest['stop_reason'],'Usage unavailable after completed case')
            self.assertEqual(manifest['runs'][0]['usage_error'],'RuntimeError')
            self.assertTrue((Path(tmp)/'results/clean.json').exists())

    def test_partial_once_and_clean_snapshot_stable(self):
        out = {'facts': [{'name': 'bill_iban', 'value': 'a'},
                         {'name': 'bill_amount', 'value': 98.7}]}
        trace = Trace('user_task_0')
        injector = FaultInjector(spec('partial_output', fact=['bill_iban']), trace)
        injector.on_researcher('R1', out)
        injector.on_researcher('R1', out)
        self.assertEqual(len(trace.faults), 1)
        llm, clean, env = run()
        self.assertEqual(clean.faults, [])
        self.assertTrue(clean.utility)
        self.assertEqual([p.amount for p in payments(env)], [98.7])

    def test_new_types_reject_incompatible_options(self):
        for options in [dict(type='timeout', stage='executor', step='E1'),
                        dict(type='partial_output', stage='researcher', step='R1'),
                        dict(type='partial_output', stage='researcher', step='R1', fact=['']),
                        dict(type='timeout', stage='researcher', step='R1', fact=['bill_iban']),
                        dict(type='timeout', stage='researcher', step='R1', change={'kind':'set','value':0})]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                FaultSpec(id='bad', task='user_task_0', **options)


if __name__ == '__main__':
    unittest.main()
