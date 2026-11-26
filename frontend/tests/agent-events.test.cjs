// Exercise App's real event callback with minimal React/render/transport substitutes.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const ts = require("typescript");

function loadApp(events) {
  const state = [];
  let cursor = 0;
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in state)) state[index] = initial;
      return [state[index], (value) => {
        state[index] = typeof value === "function" ? value(state[index]) : value;
      }];
    },
    useMemo: (fn) => fn(),
    useEffect() {},
    useRef: () => ({ current: null }),
  };
  const component = (name) => Object.assign(() => null, { displayName: name });
  const components = Object.fromEntries(
    ["Composer", "EmptyState", "MessageBubble"].map((name) => [name, component(name)]),
  );
  function load(path) {
    const source = readFileSync(path, "utf8").replaceAll("import.meta.env", "({})");
    const code = ts.transpileModule(source, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
    }).outputText;
    const module = { exports: {} };
    vm.runInNewContext(code, {
      module, exports: module.exports, Date, Math, AbortController, DOMException,
      crypto: { randomUUID: () => String(Math.random()) },
      require(name) {
        if (name === "react") return react;
        if (name === "react/jsx-runtime") {
          const jsx = (type, props) => ({ type, props });
          return { jsx, jsxs: jsx };
        }
        if (name === "lucide-react") return new Proxy({}, { get: () => component("icon") });
        if (name.startsWith("./components/")) {
          const key = name.split("/").at(-1);
          return { [key]: components[key] };
        }
        if (name === "./lib/agentApi") return {
          streamQuery: async (_query, { onEvent }) => events.forEach(onEvent),
        };
        if (name === "./lib/format") return load(resolve(__dirname, "../src/lib/format.ts"));
        throw new Error(`Unexpected dependency: ${name}`);
      },
    });
    return module.exports;
  }
  const App = load(resolve(__dirname, "../src/App.tsx")).default;
  function find(node, name) {
    if (!node || typeof node !== "object") return undefined;
    if (Array.isArray(node)) return node.map((child) => find(child, name)).find(Boolean);
    if (node.type?.displayName === name) return node;
    return find(node.props?.children, name);
  }
  function render() { cursor = 0; return App(); }
  return async () => {
    find(render(), "EmptyState").props.onUseExample("统计销售总额");
    await find(render(), "Composer").props.onSubmit();
    return state[0].find((message) => message.role === "assistant");
  };
}

test("successful terminal event preserves the result instead of converting it to an error", async () => {
  const message = await loadApp([
    { type: "result", data: [{ total_sales: 42 }] },
    { type: "terminal", status: "success", reason_code: "query_executed" },
  ])();
  assert.equal(message.status, "done");
  assert.equal(message.content, "查询完成，共 1 行结果。");
  assert.equal(message.result[0].total_sales, 42);
  assert.equal(message.error, undefined);
});

test("recall decision allows continuation and preserves an explicit blocking explanation", async () => {
  const allowed = await loadApp([
    { type: "recall_decision", status: "degraded", reason_code: "optional_source_unavailable", user_message: "部分召回不可用" },
    { type: "result", data: [] },
    { type: "terminal", status: "success", reason_code: "query_executed" },
  ])();
  assert.equal(allowed.status, "done");
  assert.equal(allowed.content, "查询完成，结果为空。");
  assert.equal(allowed.error, undefined);

  const blocked = await loadApp([
    { type: "recall_decision", status: "blocked", reason_code: "required_filter_unverified", user_message: "请补充地区；本次未执行SQL。" },
  ])();
  assert.equal(blocked.status, "error");
  assert.equal(blocked.content, "请补充地区；本次未执行SQL。");
  assert.equal(blocked.error, "required_filter_unverified");
});

test("terminal denial and execution errors are not presented as successful completion", async () => {
  for (const status of ["blocked", "error"]) {
    const message = await loadApp([
      { type: "terminal", status, reason_code: "sql_policy_denied" },
    ])();
    assert.equal(message.status, "error");
    assert.equal(message.error, "sql_policy_denied");
    assert.equal(message.result, undefined);
  }
});

test("legacy global errors remain visible", async () => {
  const message = await loadApp([{ type: "error", message: "服务不可用" }])();
  assert.equal(message.status, "error");
  assert.equal(message.error, "服务不可用");
});
