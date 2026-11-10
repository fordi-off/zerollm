# Zerochat – bruksanvisning / user guide

Zerochat er en lokal chat-app (som ChatGPT/Claude i nettleseren) som kjører språkmodeller (GGUF) helt på din egen PC, bygget på zerollm-motoren.
Ingen skytjeneste, ingen installasjon utover **Node.js 22 eller nyere**. / Zerochat is a local chat app that runs GGUF language models entirely on your own PC. It needs nothing but **Node.js 22+**.

## Norsk

### Start
1. Legg mappen `zerollm` hvor du vil (f.eks. `C:\Users\<deg>\zerollm`). Modeller (`.gguf`) ligger i `zerollm\models`.
2. Åpne en terminal i mappen og kjør:
   ```
   node app/main.js
   ```
   Et vindu åpnes (Edge i app-modus, ellers standard nettleser). Lukk vinduet for å avslutte; programmet stopper etter ca. 20 sekunder.
3. Valg: `--data <mappe>` (annen datamappe), `--port <n>`, `--no-open` (ikke åpne vindu), `--keep-alive` (ikke avslutt når vinduet lukkes), `--quiet`.

### Snarvei på Windows (uten .bat)
Høyreklikk på skrivebordet → **Ny → Snarvei**. Mål:
```
"C:\Program Files\nodejs\node.exe" app\main.js
```
Gi den et navn. Åpne snarveiens **Egenskaper** og sett **Start i** til `zerollm`-mappen. (Finn Node-stien med `where node`.)
Starter du appen to ganger, åpnes bare det eksisterende vinduet på nytt.

### Første gang
Har du ingen modell, vises «Last ned en modell». Anbefalt for norsk: **NB Llama 3.1 8B** (ca. 5 GB, trenger ca. 6–7 GB ledig minne). Mindre maskin: Llama 3.2 3B/1B eller Qwen3 4B. Du kan også legge en `.gguf`-fil i `models`-mappen selv. Motoren støtter arkitekturene llama (Llama 3.x), qwen2 og qwen3; andre filer vises som «støttes ikke».

### Bruk
- **Enter** sender, **Shift+Enter** ny linje, **Esc** stopper svaret, **Alt+N** ny samtale, **Alt+B** sidepanel, **/** åpner kommandoer.
- **Kommandoer** (`/humanize`, `/shorten`, `/simple`, `/norwegian`, `/proofread`, `/bokmal`, `/nynorsk`, `/tutor`) gjelder ett svar. Egne kommandoer legges til under Innstillinger → Kommandoer, eller som `.md`-filer i `skills`.
- **Vedlegg**: trykk binders-knappen eller dra en tekstfil (.txt, .md, .csv …) inn i vinduet. Hele teksten sendes med neste melding, og modellen husker den resten av samtalen. Teksten må få plass i kontekstvinduet (rundt 4 900 tokens ved 8192); er den for lang får du beskjed, og du kan bruke et kortere utdrag eller øke kontekstvinduet under Ytelse.
- Samtaler lagres automatisk i `chats` i samme format som Skibeti37.
- Lange samtaler oppsummeres automatisk når kontekstvinduet nærmer seg fullt.

### Hastighet og minne
Første svar etter oppstart tar tid (modellen lastes, og systemprompten leses én gang og lagres i `cache`). Å lese lange tekster er tregt på en 8B-modell (omtrent 7 tokens/sekund på en vanlig bærbar PC), så korte vedlegg gir raskere første svar (selve vedlegget leses bare én gang). Modellen lastes i bakgrunnen når appen starter (kan slås av under Ytelse). Kontekstmåleren under meldingsfeltet viser hvor stor prompten er. Mindre modeller er raskere. Øk ikke kontekstvinduet mer enn nødvendig: det bruker mer minne.

### Når noe ikke virker
Innstillinger → **Diagnose** viser system, funksjoner, modeller og siste loggfil-linjer. «Kopier rapport» gir deg alt du trenger å sende. Loggen ligger i `app.log` i datamappen.

## English

### Start
1. Put the `zerollm` folder anywhere; models (`.gguf`) go into `zerollm\models`.
2. Open a terminal in the folder and run `node app/main.js`. A window opens (Edge in app mode, otherwise your default browser). Closing it quits the program after ~20 s. Options: `--data <folder>`, `--port <n>`, `--no-open`, `--keep-alive`, `--quiet`.
3. Windows shortcut without a .bat: right-click desktop → New → Shortcut → target `"C:\Program Files\nodejs\node.exe" app\main.js`, then set *Start in* to the `zerollm` folder.

### First run
With no model you get a "Download a model" card. Recommended for Norwegian: NB Llama 3.1 8B (~5 GB, ~6–7 GB free RAM). Smaller PC: Llama 3.2 3B/1B or Qwen3 4B. Qwen3.5 4B (~2.6 GB, ~3.5 GB free RAM) is in the list too: a newer model with a "Tenk først" switch like Qwen3; it reads long prompts slowly (about 5 tokens/s on a laptop), so prefer short messages and files with it. Supported architectures: llama (3.x), qwen2, qwen3, qwen35 (Qwen3.5).

### Use
Enter sends, Shift+Enter newline, Esc stops, Alt+N new chat, Alt+B sidebar, `/` opens commands. Slash commands apply to one reply; an attached text file (paperclip or drag-and-drop) goes into the message whole and stays in the model's memory for the rest of the chat; chats are stored in Skibeti37's format; long chats are summarized automatically.

### Speed
Prompt reading is the slow part on large models (about 7 tokens/s for an 8B model on an ordinary laptop), so keep attachments short (the model reads an attachment only once). The first reply after a start also loads the model and caches the system prompt in `cache/`.

### Troubleshooting
Settings → Diagnostics shows system, features, models and the log tail; "Copy report" gives everything needed for a bug report. The log is `app.log` in the data folder.

## Mappe / folder layout
```
zerollm/
  app/            programmet / the program
  src/            motoren / the engine
  models/         .gguf-filer / model files
  chats/          samtaler / conversations (+ _context_summaries/)
  skills/         kommandoer / slash commands (.md)
  cache/          lagrede systemprompt-tilstander / cached system-prompt states (can be deleted)
  config.json     innstillinger / settings
  app.log         logg / log
```
For en bærbar kopi: kopier hele mappen (uten `models` hvis den er stor) til en minnepinne – ingen installasjon trengs bortsett fra Node.js. / For a portable copy, copy the whole folder; only Node.js is needed on the target PC.
