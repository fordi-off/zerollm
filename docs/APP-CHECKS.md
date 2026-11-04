# Manual checks for the app (laptop, real models)

Run `npm run app -- --data <folder containing models/>` (or `node app/main.js`). A window opens; the first check is that it opens at all on the managed PC.

## A1 - chat
- [ ] Window opens (Edge app window or default browser); closing it makes the program exit within ~20 s.
- [ ] Model picker lists the .gguf files; unsupported files are greyed with a reason.
- [ ] NB Llama 3.1 8B: first message shows "Laster modellen..." then "Leser teksten..." with progress, then streams text. Note the time to first token and tok/s (shown under the reply) in docs/BENCHMARKS.md.
- [ ] Norwegian question gets a Norwegian answer; markdown lists/code blocks render; the Copy buttons work.
- [ ] Stop button ends a long answer; "Fortsett" appears after a cut-off answer and continues it.
- [ ] "Lag nytt svar" replaces the last answer.
- [ ] Context meter moves while typing and after replies; a very long paste does not freeze the window.
- [ ] Settings: context window change reloads the model on next send; thread count; Norwegian/English; light/dark.
- [ ] Task Manager: memory stays below the estimate shown in the model list; after "unload" it drops.
- [ ] Starting the app twice opens the same window instead of a second server.
