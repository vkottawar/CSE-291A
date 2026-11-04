"""Three live smoke cases; reads an existing env file without copying credentials."""
import argparse
import json
import os
from pathlib import Path

from .faults import load_specs
from .llm import LLM, _load_env_file
from .runner import run_task


def usage():
    import ssl
    import truststore
    import urllib.request
    req = urllib.request.Request(os.environ['LLM_BASE_URL'].removesuffix('/v1')+'/key/info',
        headers={'Authorization': 'Bearer '+os.environ['LLM_API_KEY']})
    with urllib.request.urlopen(req, context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), timeout=30) as f:
        info = json.load(f)['info']
    return {key:info.get(key) for key in ('spend','max_budget')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', required=True, help='existing authorized configuration; never copied')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    _load_env_file(Path(args.env_file))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    specs = load_specs(Path(__file__).parent.parent/'faults/researcher_failures.json')
    manifest = {'model':os.environ['LLM_MODEL'], 'seed':0, 'temperature':0.0,
                'layout':'planner-first', 'usage_before':usage(), 'runs':[], 'stop_reason':None}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for name, fault in [('clean',None),*[(s.id,s) for s in specs]]:
        llm = LLM(model=os.environ['LLM_MODEL'],base_url=os.environ['LLM_BASE_URL'],
                  api_key=os.environ['LLM_API_KEY'],seed=0,temperature=0.0)
        _, trace, env = run_task('user_task_0',llm,fault=fault,return_env=True)
        trace.save(out/f'{name}.json')
        (out/f'{name}.world.json').write_text(env.model_dump_json(indent=2))
        row = {'condition':name,'utility':trace.utility,'error':trace.error,
               'faults':trace.faults,'transactions':trace.transaction_changes,
               'model_turns':sum(s.turns for s in trace.steps),'tokens':trace.tokens(),
               'seconds':trace.seconds,'usage_after':None}
        try:
            row['usage_after'] = usage()
        except Exception as e:
            # Preserve the completed case and stop before any further inference.
            row['usage_error'] = type(e).__name__
            manifest['stop_reason'] = 'Usage unavailable after completed case'
        manifest['runs'].append(row)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
        print(json.dumps(row,default=str),flush=True)
        if manifest['stop_reason']:
            break
        if row['usage_after']['spend']-manifest['usage_before']['spend'] > 0.02:
            manifest['stop_reason']='Observed cost guard reached'
            break
    manifest['usage_after']=manifest['runs'][-1]['usage_after']
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))


if __name__ == '__main__':
    main()
