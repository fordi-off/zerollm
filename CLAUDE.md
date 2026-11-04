# zerollm

Read `docs/BRIEF.md` before writing code: goal, hard constraints, target hardware, design decisions, milestone plan.

Hard rules (details in the brief):
- Zero npm dependencies, no native addons, no shelling out to external binaries. Ask the user before breaking this.
- Node built-ins only; ES modules; JSDoc types, no build step; tests with `node --test`.
- Correctness first: the plain-JS reference path stays forever as the oracle for the fast path.
- Work milestone by milestone (brief section 20); commit after each; log choices in `docs/DECISIONS.md`, numbers in `docs/BENCHMARKS.md`.
- Never claim speeds without measuring.

Commands: `npm test` (= `node --test`), `node bin/zerollm.js doctor|info|...`.
