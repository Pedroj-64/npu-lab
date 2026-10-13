// End-to-end test of the web agent panel in a headless Chromium, WITHOUT a model: /chat is answered by a script, everything else is the real server code
// (/agent/run, /rules, the page's own JavaScript, the "Do it" / "Stop the agent" buttons). Needs a running server and a playwright-core + Chromium that are already on disk.
// usage: node tests/e2e_agent_panel.mjs   with env: URL=http://127.0.0.1:PORT/#code=CODE  PROJECT=<folder inside $HOME with a.py (return 1) and old.py>  PY=<python>
//        PLAYWRIGHT_CORE=<path to playwright-core>  CHROMIUM=<path to the chromium binary>
import { createRequire } from "module";
import fs from "fs";
const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_CORE);
const { URL: PAGE, PROJECT, PY, CHROMIUM } = process.env;
const read = f => fs.readFileSync(`${PROJECT}/${f}`, "utf8");
const edit = (a, b) => `a.py\n<<<<<<< SEARCH\nreturn ${a}\n=======\nreturn ${b}\n>>>>>>> REPLACE`;
let fails = 0;
const ok = (c, m) => { console.log((c ? "PASS " : "FAIL ") + m); if (!c) fails++; };

const browser = await chromium.launch({ executablePath: CHROMIUM, headless: true });
const page = await (await browser.newContext()).newPage();
let queue = [];                                              // scripted model replies, one per /chat call
page.on("pageerror", e => { console.log("FAIL page error: " + e.message); fails++; });
await page.route("**/status", async route => { const r = await route.fetch(); const j = await r.json(); await route.fulfill({ json: { ...j, mode: "llm", model: "scripted" } }); });
await page.route("**/chat", async route => {
  const text = queue.shift() ?? "DONE (script ran out)";
  const lines = [{ token: text }, { done: true, stats: { decode_tokens: 3, decode_s: 1, prefill_tokens: 5, reused_tokens: 0 } }].map(x => JSON.stringify(x)).join("\n") + "\n";
  await route.fulfill({ status: 200, contentType: "application/x-ndjson", body: lines });
});
await page.goto(PAGE);
await page.waitForFunction(() => typeof TOKEN === "string" && TOKEN.length > 10);
await page.evaluate(([root, py]) => api("/rules", { project_root: root, agent_guard: "objective", agent_test_command: `${py} -c "import a; assert a.f() == 2"` }), [PROJECT, PY]);
await page.evaluate(() => refresh());
await page.waitForFunction(() => ST && ST.mode === "llm");
await page.evaluate(() => { show("chat"); document.getElementById("agentmode").checked = true; });
const log = () => page.evaluate(() => document.getElementById("log").innerText);
const send = async text => { await page.fill("#prompt", text); await page.click("#send"); };
const press = async name => { await page.getByRole("button", { name }).first().click({ timeout: 20000 }); };

// A: edit -> Do it, DONE refused (tests not run), RUN -> Do it, DONE accepted
queue = [edit(1, 2), "DONE it works", "RUN", "DONE finished"];
await send("make f return 2");
await page.getByRole("button", { name: "Do it" }).first().waitFor({ timeout: 20000 });
ok(read("a.py").includes("return 1"), "A: nothing written before the click");
await press("Do it");
await page.waitForFunction(() => document.getElementById("log").innerText.includes("run the project's tests"), null, { timeout: 30000 });
ok(read("a.py").includes("return 2"), "A: the click wrote the edit");
ok((await log()).includes("DONE refused"), "A: DONE refused until the tests are run");
await page.getByRole("button", { name: "Do it" }).first().waitFor({ timeout: 20000 });
ok(/run the tests/.test(await log()), "A: RUN waits for the click and shows the command");
await press("Do it");
await page.waitForFunction(() => document.getElementById("log").innerText.includes("Agent finished"), null, { timeout: 30000 });
ok((await log()).includes("exit code 0"), "A: test output is shown");
ok((await log()).includes("Agent finished"), "A: DONE accepted after passing tests");

// B: Stop the agent -> nothing written
await page.evaluate(() => document.getElementById("newchat").click());
queue = [edit(2, 9)];
await send("make f return 9");
await page.getByRole("button", { name: "Stop the agent" }).first().waitFor({ timeout: 20000 });
await press("Stop the agent");
await page.waitForFunction(() => document.getElementById("log").innerText.includes("nothing was changed"), null, { timeout: 20000 });
ok(read("a.py").includes("return 2") && !read("a.py").includes("return 9"), "B: Stop leaves the file untouched");

// C: DONE with no change is refused by the guard
await page.evaluate(() => document.getElementById("newchat").click());
queue = ["DONE nothing to do", "LIST", "DONE"];
await send("do nothing");
await page.waitForFunction(() => document.getElementById("log").innerText.includes("no file changed"), null, { timeout: 30000 });
ok(true, "C: DONE with no change refused (guard)");

await browser.close();
console.log(fails ? `${fails} FAILED` : "agent panel e2e ok");
process.exit(fails ? 1 : 0);
