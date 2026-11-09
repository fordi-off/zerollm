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

## A2 - chats
- [ ] Send a message, wait for the reply: the chat appears in the sidebar (named after your first message) and a file `chats/<id>.json` exists.
- [ ] Close and restart the app: the chat is still there and opens with all messages.
- [ ] Rename (pencil) and delete (cross, then confirm) work; the model of the chat is selected when it is opened.
- [ ] Edit a user message ("Rediger"): the later replies are removed and a new reply is generated.
- [ ] Clicking another chat or "Ny samtale" while a reply is being written stops it first, and the partial reply is saved.
- [ ] Shared folder with Skibeti37: copy a few of its `chats/*.json` into this data folder - they show up and open; a chat saved here still opens in Skibeti37.

## A3 - models and downloads
- [ ] In an empty data folder the first-run card appears; "Last ned en modell" opens Settings with the catalog.
- [ ] Download the smallest model (Llama 3.2 1B): progress, speed and ETA move; when finished it is selectable and chats. **Note the download speed** (and compare with a browser download of the same file) - if much slower, tell me (Node vs curl TLS throttling).
- [ ] Cancel in the middle, then "Fortsett nedlasting" continues from the same byte count.
- [ ] Unplug the network for a few seconds during a download: it retries and continues.
- [ ] All catalog URLs work (each shows a size close to the listed one); report any 404.
- [ ] KV f32/f16 selector changes the RAM estimate; delete model asks for confirmation and removes the file.

## A4 - skills
- [ ] Type `/` or press the `/` button: the menu lists /humanize, /shorten, /simple, /norwegian, /proofread, /bokmal, /nynorsk, /tutor; arrows + Tab/Enter pick one, Esc closes.
- [ ] `/shorten Forklar hva en kommune er` gives a clearly shorter answer than the same question without the command; the chip `/shorten` shows under the message.
- [ ] `/proofread <tekst med feil>` fixes the text; the next message without a command behaves normally again.
- [ ] Add a command in Settings > Kommandoer; drop a `.md` file into `skills/` while the app runs - it appears (after focusing the window or reopening the menu).
- [ ] Skibeti37 chats that contain `commands` keep their chips.

## A5 - long conversations
- [ ] Set the context window to 2048 or 4096 (Innstillinger > Ytelse) and have a long conversation (15-20 exchanges). At some point the status line shows "Oppsummerer eldre meldinger..." once, then answers continue; the reply shows "Eldre meldinger er oppsummert".
- [ ] After that, ask about a fact from the very beginning: the answer should still know it (from the summary). **Note how long the summary step and the following "Leser teksten" take** (8B with and without a small second model installed, e.g. Llama 3.2 1B).
- [ ] Choose a "Modell for oppsummering" in Settings and confirm app.log says `summarizing with <that file>`; with too little free RAM it should fall back to the chat model (log line) instead of failing.
- [ ] Editing an early message in a summarized chat does not use the stale summary.

## A6 - documents
- [ ] Attach a Norwegian .txt/.md (paperclip or drag it onto the window): a chip appears; click the name to switch it off/on, the x removes it.
- [ ] Ask something that is in the document: the answer uses it and "Kilder · n utdrag" shows the excerpts. Ask something that is NOT in it: the model should say so.
- [ ] **Note the wait** for the first token with 4 excerpts on the 8B model (the context meter shows the prompt size); try 2 excerpts in Innstillinger > Dokumenter. A follow-up question in the same chat should be quicker because the history is cached.
- [ ] A big file (1-2 MB) indexes within a second or two and the app stays responsive.
- [ ] Settings > Dokumenter: library lists the file; "Vis tekst" shows it; "Lag sammendrag" produces a summary (note how long it takes with and without a small second model); delete removes it from every chat.
- [ ] Skibeti37's `documents/*.json` copied into this data folder show up and answer questions.

## A7 - polish
- [ ] Starting `node app/main.js` from a different working directory still finds `models/` (data folder = the zerollm folder).
- [ ] The Windows shortcut from docs/APP-GUIDE.md starts the app without a console window problem (a console window stays open while it runs - that is expected) and opens the window.
- [ ] Settings > Diagnose: report is complete; "Kopier rapport" puts it on the clipboard.
- [ ] Esc stops a reply; Alt+N / Alt+B work; `/` outside the text box jumps to it.
- [ ] Tab through the page: every control reachable, focus ring visible; dark mode readable.

## Search model (embeddings) - do this one first
- [ ] Download `nomic-embed-text` from Settings > Models > Download models (or put `nomic-embed-text-v1.5.Q4_K_M.gguf` in `models/`).
- [ ] **Ground truth:** `node bin/zerollm.js embed models/nomic-embed-text-v1.5.Q4_K_M.gguf --compare documents/<a document Skibeti37 made>.json --n 5` - every similarity should be about 0.99 or higher (it prints one line with accents stripped and one with accents kept; tell me both). Lower values mean a mistake in my encoder.
- [ ] `node bin/zerollm.js embed models/nomic-embed-text-v1.5.Q4_K_M.gguf "Hva er en kommune?" "Kommunen har ansvar for skoler" "Pizza med ost"` - the first two should be clearly more similar than either is to the third.
- [ ] `node tools/embed-bench.mjs models/nomic-embed-text-v1.5.Q4_K_M.gguf` - note chunks/second.
- [ ] Add a document in the app: Settings > Dokumenter shows "Lager søkeindeks… n/m" then "søkeindeks m/m"; a question phrased with different words than the text still finds the right passage.

## Thinking mode (Qwen3 only)
- [ ] Select a Qwen3 model: a "Tenk først" pill appears under the message box; with a Llama/NB model it is hidden.
- [ ] Turn it on and ask the falling-stone question (constant g, then with Newtonian gravity). While it works you see an open "Tenker… m:ss" block with the reasoning; when the answer starts the block folds to "Tenkte i m:ss"; click it to read the reasoning. Note the total time.
- [ ] Reload the page / reopen the chat: the folded reasoning is still there.
- [ ] Ask a follow-up question: it must start quickly (previous reasoning is not re-read).
- [ ] `node tools/think-eval.mjs models/<qwen3 file>.gguf --only 3,4` (then without --only): send me the printed table and `think-eval-results.json`.
- [ ] Non-thinking Qwen3 chats: second and later answers should start faster than before (earlier replies are no longer re-read).
