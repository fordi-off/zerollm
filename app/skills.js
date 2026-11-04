// Skills = plain .md files in <data>/skills (same format as Skibeti37). One file can define several slash commands:
//
//   # Skill name
//   ## /command-name  Short description shown in the / menu
//   The instruction the model follows for the one reply that starts with /command-name.
//
// The folder is re-read on every call, so files added or edited outside the app are picked up without a restart.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SEED_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'seed-skills');
const COMMAND_HEADING = /^##\s+\/([A-Za-z0-9_-]+)[ \t]*(.*)$/;
const FILE_RE = /^[A-Za-z0-9][A-Za-z0-9 _.-]{0,80}\.(md|txt)$/i;

/** @returns {{file: string, name: string, commands: {command: string, description: string, body: string}[]}} */
export function parseSkill(fname, text) {
  const stem = fname.replace(/\.(md|txt)$/i, '').replace(/[_-]+/g, ' ').trim();
  let name = stem ? stem[0].toUpperCase() + stem.slice(1) : fname;
  const commands = []; let current = null;
  for (const line of text.split(/\r?\n/)) {
    const m = COMMAND_HEADING.exec(line);
    if (m) {
      if (current) commands.push(current);
      current = { command: m[1].toLowerCase(), description: m[2].trim(), bodyLines: [] };
      continue;
    }
    if (current === null) {
      const s = line.trim();
      if (s.startsWith('#') && !s.startsWith('##')) name = s.replace(/^#+/, '').trim() || name;
      continue;
    }
    current.bodyLines.push(line);
  }
  if (current) commands.push(current);
  const cleaned = [];
  for (const c of commands) {
    const lines = c.bodyLines;
    while (lines.length && !lines[0].trim()) lines.shift();
    let description = c.description;
    if (!description && lines.length) { description = lines.shift().trim(); while (lines.length && !lines[0].trim()) lines.shift(); }
    const body = lines.join('\n').trim();
    if (c.command && body) cleaned.push({ command: c.command, description, body });
  }
  return { file: fname, name, commands: cleaned };
}

export function createSkills(paths) {
  const dir = paths.skillsDir;
  const readAll = () => {
    let names = []; try { names = fs.readdirSync(dir); } catch { return []; }
    return names.filter((f) => /\.(md|txt)$/i.test(f)).sort().map((f) => {
      try { return parseSkill(f, fs.readFileSync(path.join(dir, f), 'utf8')); } catch { return null; }
    }).filter(Boolean);
  };
  const safeFile = (id) => { if (typeof id !== 'string' || id !== path.basename(id) || !FILE_RE.test(id)) throw new Error('invalid skill file'); return path.join(dir, id); };

  return {
    /** Copy the bundled skills into an empty skills folder (never brings back skills the user deleted). */
    seed() {
      fs.mkdirSync(dir, { recursive: true });
      if (readAll().length > 0 || fs.readdirSync(dir).some((f) => /\.(md|txt)$/i.test(f))) return false;
      for (const f of fs.readdirSync(SEED_DIR)) fs.copyFileSync(path.join(SEED_DIR, f), path.join(dir, f));
      return true;
    },
    /** Flat list for the / menu (later files win on a name clash). */
    commands() {
      const seen = new Map();
      for (const s of readAll()) for (const c of s.commands) seen.set(c.command, { command: c.command, description: c.description, skill: s.name });
      return [...seen.values()].sort((a, b) => a.command.localeCompare(b.command));
    },
    /** Grouped by file, for Settings. */
    list() {
      return readAll().map((s) => ({ id: s.file, name: s.name, commands: s.commands.map(({ command, description }) => ({ command, description })) }));
    },
    /** Instruction text of the given commands joined in order; null if none is known. */
    bodies(names) {
      if (!Array.isArray(names) || names.length === 0) return null;
      const index = new Map();
      for (const s of readAll()) for (const c of s.commands) index.set(c.command, c.body);
      const parts = names.map((n) => index.get(String(n).replace(/^\//, '').toLowerCase())).filter(Boolean);
      return parts.length ? parts.join('\n\n') : null;
    },
    /** Validate and save a new skill file. Throws a Norwegian message when it defines nothing usable. */
    add(name, content) {
      const slug = String(name ?? '').replace(/[^A-Za-z0-9 _-]/g, '').trim().replace(/ +/g, '-').toLowerCase().slice(0, 60) || 'skill';
      const text = String(content ?? '').trim();
      if (!text.split(/\r?\n/).some((l) => COMMAND_HEADING.test(l))) throw new Error("Fant ingen kommando. Hver kommando starter med en linje som '## /navn'.");
      const header = text.startsWith('#') && !text.startsWith('##') ? '' : `# ${String(name ?? '').trim() || slug}\n\n`;
      if (parseSkill(`${slug}.md`, header + text).commands.length === 0) throw new Error("Kommandoen mangler instruksjonstekst under '## /navn'-linjen.");
      fs.mkdirSync(dir, { recursive: true });
      let fname = `${slug}.md`; let n = 1;
      while (fs.existsSync(path.join(dir, fname))) fname = `${slug}-${n++}.md`;
      fs.writeFileSync(path.join(dir, fname), `${header}${text}\n`, 'utf8');
      return fname;
    },
    delete(id) { try { fs.unlinkSync(safeFile(id)); } catch (e) { if (e.message === 'invalid skill file') throw e; } return true; },
  };
}
