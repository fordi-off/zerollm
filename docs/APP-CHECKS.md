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
