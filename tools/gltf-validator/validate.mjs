// Validate glTF/GLB files with the official Khronos glTF Validator.
//
// Usage: node validate.mjs [--allow-warnings] FILE...
//
// Prints one JSON summary per file and exits with status 1 if any file has
// errors (or warnings, unless --allow-warnings is given). Infos and hints are
// reported but never fail the run.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const validator = require("gltf-validator");

const args = process.argv.slice(2);
const allowWarnings = args.includes("--allow-warnings");
const files = args.filter((a) => a !== "--allow-warnings");
if (files.length === 0) {
  console.error("usage: node validate.mjs [--allow-warnings] FILE...");
  process.exit(2);
}

let failed = false;
for (const file of files) {
  const fullPath = path.resolve(file);
  const baseDir = path.dirname(fullPath);
  let report;
  try {
    report = await validator.validateBytes(new Uint8Array(await readFile(fullPath)), {
      uri: path.basename(fullPath),
      maxIssues: 0,
      externalResourceFunction: async (uri) => {
        const resolved = path.resolve(baseDir, decodeURIComponent(uri));
        if (path.relative(baseDir, resolved).startsWith("..")) {
          throw new Error(`refusing to read outside ${baseDir}: ${uri}`);
        }
        return new Uint8Array(await readFile(resolved));
      },
    });
  } catch (error) {
    console.log(JSON.stringify({ file, ok: false, exception: String(error) }));
    failed = true;
    continue;
  }
  const { numErrors, numWarnings, numInfos, numHints, messages } = report.issues;
  const ok = numErrors === 0 && (allowWarnings || numWarnings === 0);
  failed ||= !ok;
  console.log(
    JSON.stringify({
      file,
      ok,
      validatorVersion: report.validatorVersion,
      errors: numErrors,
      warnings: numWarnings,
      infos: numInfos,
      hints: numHints,
      messages: messages.map((m) => `${["ERROR", "WARNING", "INFO", "HINT"][m.severity]} ${m.code} ${m.pointer ?? ""} ${m.message}`),
    }),
  );
}
process.exit(failed ? 1 : 0);
