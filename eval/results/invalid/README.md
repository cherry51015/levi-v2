Runs kept for transparency but excluded from reported results.

- `redteam_20260926_232745_quota_errors.json`: 18 of 58 red-team queries and 5 of 6 injection queries
  failed with LLMUnavailable (Groq free-tier daily token quota exhausted). The eval at that time counted
  errors as "not refused" / "attack failed", which would have overstated the results. Router-only metrics
  from this run (accuracy 82.8%, uncertain rate 31%) do not depend on the LLM and are still valid.
