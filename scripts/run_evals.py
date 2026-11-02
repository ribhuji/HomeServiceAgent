from home_service_agent.evals import run_evals


results = run_evals()
failed = [result for result in results if not result.passed]
print(f"Ran {len(results)} evals: {len(results) - len(failed)} passed, {len(failed)} failed")
for result in failed:
    print(f"FAIL {result.scenario}: expected {result.expected}, got {result.actual}")
raise SystemExit(1 if failed else 0)

