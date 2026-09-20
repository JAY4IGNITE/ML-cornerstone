# Claude Code Execution Rules

1. Begin with repository audit.
2. Ask for clarification only when an essential decision cannot be inferred from the repository or dataset.
3. Prefer small, testable changes.
4. Preserve working code unless a change is justified.
5. Never fabricate files, metrics, dataset columns, or test outcomes.
6. Inspect actual schemas before writing feature code.
7. Use configuration instead of hardcoded paths and thresholds.
8. Keep secrets out of source control.
9. Keep training and inference pipelines consistent.
10. Report blockers explicitly.
11. Run relevant tests after modifications.
12. Do not claim completion when a phase is incomplete.
13. Do not silently discard data.
14. Do not treat model predictions as definitive lending decisions.
15. At the end of each phase, report:
   - Files changed
   - What was implemented
   - Commands executed
   - Test results
   - Remaining issues
