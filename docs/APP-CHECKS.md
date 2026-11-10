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

## A7 - polish
- [ ] Starting `node app/main.js` from a different working directory still finds `models/` (data folder = the zerollm folder).
- [ ] The Windows shortcut from docs/APP-GUIDE.md starts the app without a console window problem (a console window stays open while it runs - that is expected) and opens the window.
- [ ] Settings > Diagnose: report is complete; "Kopier rapport" puts it on the clipboard.
- [ ] Esc stops a reply; Alt+N / Alt+B work; `/` outside the text box jumps to it.
- [ ] Tab through the page: every control reachable, focus ring visible; dark mode readable.

## Thinking mode (Qwen3 only)
- [ ] Select a Qwen3 model: a "Tenk først" pill appears under the message box; with a Llama/NB model it is hidden.
- [ ] Turn it on and ask the falling-stone question (constant g, then with Newtonian gravity). While it works you see an open "Tenker… m:ss" block with the reasoning; when the answer starts the block folds to "Tenkte i m:ss"; click it to read the reasoning. Note the total time.
- [ ] Reload the page / reopen the chat: the folded reasoning is still there.
- [ ] Ask a follow-up question: it must start quickly (previous reasoning is not re-read).
- [ ] `node tools/think-eval.mjs models/<qwen3 file>.gguf --only 3,4` (then without --only): send me the printed table and `think-eval-results.json`.
- [ ] Non-thinking Qwen3 chats: second and later answers should start faster than before (earlier replies are no longer re-read).

## Attached files (replaces the old document library and search)
- [ ] Paperclip (or drag a .txt/.md into the window): a chip with the file name appears above the message box.
- [ ] Ask a question about it: the answer comes from the text; the file name stays as a chip on your message. Note how long the FIRST answer takes (the whole text is read once).
- [ ] A second question in the same chat starts quickly: the document is already in the model's memory.
- [ ] A document that is too long for the context window is refused with a message that says how many tokens it has and the limit; a shorter excerpt works. Raising the context window (Ytelse) raises the limit.
- [ ] Reload the page and reopen the chat: the file chip is still on the message and follow-up questions still know the document.

## Load the last model at start-up (Ytelse, on by default)
- [ ] Start the app, wait ~30 s ("Laster modellen…" shows at the top), then send a message: it must not show "Laster modellen…" again and the first word should come after only a few seconds.
- [ ] Turn the option off in Ytelse, restart: nothing is loaded until you send a message.

## Automatic chat titles (Ytelse > "Automatiske samtaletitler", on by default)
Run once with only the chat model installed (the chat model writes the title) and once with a small second model (e.g. Llama 3.2 1B) chosen as "Modell for oppsummering" (or auto-picked); app.log says `titling with <file>` / `title run (chat model): N prompt tokens, M cached, K generated, T ms`.
- [ ] Norwegian: start a new chat in Norwegian (e.g. "Hva er forskjellen på kommune og fylkeskommune?"). A few seconds after the first reply the sidebar entry changes from the first 45 characters to a short Norwegian title (3-6 words, no quotes, no final punctuation, æøå intact). Write down 5 titles you get and whether they are in the right language and sensible.
- [ ] English: same with an English question; the title must be English, not Norwegian.
- [ ] A very short first message ("hei") and a long one with an attached file: the title is still sensible or the name simply stays as it was (never garbage or an error message).
- [ ] Nothing slows the next reply: send the second message right after the title appears, and once while the title is still being written. In app.log the second reply's `done:` line should show the same `cached` as the first reply plus its own tokens (no full re-read), and the reply should start about as quickly as before. Note the `title run (chat model)` time on the 8B model.
- [ ] Sending a message while the title is being written starts the reply promptly (the log has `title for <id> aborted`), and the chat is titled after the next save or keeps its first-message name - it must never block, and the answer must be unaffected.
- [ ] Renaming wins: rename the chat (pencil) before the title shows up, or just after; the name you typed stays, also after the next reply and after a reload. Rename it back to something else later: still yours.
- [ ] Reload / close and reopen the page right after the first reply: the title is there (written by the server regardless of the page).
- [ ] Turn "Automatiske samtaletitler" off: new chats keep the first-45-characters name. Turn it on again: the next new chat gets a title. The hint text in Settings explains the trade-off.
- [ ] Old chats (and chats from Skibeti37) never change name by themselves.
