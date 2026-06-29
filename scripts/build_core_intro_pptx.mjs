import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(process.env.PROJECT_ROOT ?? process.cwd());
const finalPptx = path.resolve(
  process.env.FINAL_PPTX ?? path.join(projectRoot, "outputs", "creditbond_ai_core_intro.pptx"),
);
const previewDir = path.join(__dirname, "preview");
const layoutDir = path.join(__dirname, "layout");
const qaDir = path.join(__dirname, "qa");

const C = {
  ink: "#17202E",
  muted: "#58677A",
  paper: "#F7F9FC",
  line: "#D9E1EA",
  blue: "#2F6FA5",
  teal: "#27896B",
  amber: "#B98222",
  red: "#C94346",
  lavender: "#6E63B6",
  white: "#FFFFFF",
};

const W = 1280;
const H = 720;
const frame = { left: 72, top: 58, width: 1136, height: 596 };
const typeface = "Microsoft YaHei";

async function exists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function readJson(rel) {
  const p = path.join(projectRoot, rel);
  if (!(await exists(p))) return {};
  return JSON.parse(await fs.readFile(p, "utf-8"));
}

async function readCsv(rel) {
  const p = path.join(projectRoot, rel);
  if (!(await exists(p))) return [];
  const text = await fs.readFile(p, "utf-8");
  const rows = [];
  let current = "";
  let row = [];
  let inQuote = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"' && next === '"') {
      current += '"';
      i++;
    } else if (ch === '"') {
      inQuote = !inQuote;
    } else if (ch === "," && !inQuote) {
      row.push(current);
      current = "";
    } else if ((ch === "\n" || ch === "\r") && !inQuote) {
      if (ch === "\r" && next === "\n") i++;
      row.push(current);
      if (row.some((v) => v.length > 0)) rows.push(row);
      row = [];
      current = "";
    } else {
      current += ch;
    }
  }
  if (current.length || row.length) {
    row.push(current);
    rows.push(row);
  }
  const [header, ...data] = rows;
  if (!header) return [];
  return data.map((r) => Object.fromEntries(header.map((h, i) => [h.replace(/^\uFEFF/, ""), r[i] ?? ""])));
}

function pct(v, digits = 1) {
  const n = Number(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "-";
}

function bp(v) {
  const n = Number(v);
  return Number.isFinite(n) ? `${n >= 0 ? "+" : ""}${n.toFixed(1)} bp` : "-";
}

function modelName(modelDir) {
  const s = String(modelDir || "").replaceAll("\\", "/").toLowerCase();
  if (s.endsWith("/gru")) return "GRU";
  if (s.endsWith("/tcn")) return "TCN";
  if (s.endsWith("/transformer")) return "Transformer";
  return "模型";
}

function text(slide, name, content, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = content;
  shape.text.style = {
    typeface,
    fontSize: opts.size ?? 24,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
  };
  return shape;
}

function box(slide, name, x, y, w, h, opts = {}) {
  const config = {
    geometry: opts.geometry ?? "roundRect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: opts.fill ?? C.white,
    line: { style: "solid", fill: opts.line ?? C.line, width: opts.lineWidth ?? 1 },
  };
  if ((opts.geometry ?? "roundRect") === "roundRect") {
    config.borderRadius = opts.radius ?? "rounded-xl";
  }
  if (opts.shadow) {
    config.shadow = opts.shadow;
  }
  return slide.shapes.add(config);
}

function title(slide, label, kicker) {
  text(slide, "kicker", kicker, frame.left, 42, 540, 30, {
    size: 18,
    bold: true,
    color: C.muted,
  });
  text(slide, "slide-title", label, frame.left, 82, 930, 62, {
    size: 50,
    bold: true,
    color: C.ink,
  });
}

function footer(slide, page) {
  text(slide, "footer-left", "CreditBond AI 研究与决策辅助框架", 72, 666, 460, 28, {
    size: 16,
    color: C.muted,
  });
  text(slide, "footer-page", String(page), 1156, 666, 52, 28, {
    size: 16,
    color: C.muted,
    align: "right",
  });
}

function coloredRule(slide, x, y, w) {
  box(slide, "rule-blue", x, y, w * 0.52, 6, { geometry: "rect", fill: C.blue, line: C.blue, radius: "none" });
  box(slide, "rule-teal", x + w * 0.52, y, w * 0.18, 6, {
    geometry: "rect",
    fill: C.teal,
    line: C.teal,
    radius: "none",
  });
  box(slide, "rule-amber", x + w * 0.70, y, w * 0.16, 6, {
    geometry: "rect",
    fill: C.amber,
    line: C.amber,
    radius: "none",
  });
  box(slide, "rule-red", x + w * 0.86, y, w * 0.14, 6, {
    geometry: "rect",
    fill: C.red,
    line: C.red,
    radius: "none",
  });
}

function addBullet(slide, name, lines, x, y, w, h, opts = {}) {
  const content = lines.map((line) => `• ${line}`).join("\n");
  return text(slide, name, content, x, y, w, h, {
    size: opts.size ?? 24,
    color: opts.color ?? C.ink,
  });
}

function findBestByTenor(rows) {
  const out = {};
  for (const row of rows) {
    const tenor = row.tenor;
    const value = Number(row.total_return);
    if (!Number.isFinite(value)) continue;
    if (!out[tenor] || value > Number(out[tenor].total_return)) out[tenor] = row;
  }
  return out;
}

async function collectState() {
  const report = await readJson("data/dm_daily_master_smoke/reports/daily_dm_update_report.json");
  const bestRows = await readCsv("data/experiments/no_data_strategy_sweep_full/strategy_sweep_best_by_entity.csv");
  const best = findBestByTenor(bestRows);
  const tenors = (report.market_snapshot?.tenors ?? []).map((item) => ({
    label: item.label,
    date: item.latest_date,
    yield: item.latest_yield,
    change20: item.change_20d_bp,
    signal: item.model_prediction?.prediction ?? "-",
    model: modelName(item.model_prediction?.model_dir),
  }));
  return {
    runDate: report.run_date ?? "2026-06-24",
    dataStart: report.data_start ?? "-",
    dataEnd: report.data_end ?? "-",
    featuresShape: report.features_shape ?? ["-", "-"],
    modelCount: (report.predictions ?? []).length,
    tenors,
    bestReturns: ["3", "5", "10", "20"].map((t) => Number(best[t]?.total_return ?? 0)),
    bestReturnsPct: ["3", "5", "10", "20"].map((t) => Number(best[t]?.total_return ?? 0) * 100),
  };
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addSlide1(presentation, state) {
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  box(slide, "blue-band", 0, 0, 1280, 366, { geometry: "rect", fill: C.blue, line: C.blue, radius: "none" });
  coloredRule(slide, 0, 366, 1280);
  text(slide, "deck-title", "信用债 AI\n研究与决策辅助框架", 86, 126, 930, 176, {
    size: 66,
    bold: true,
    color: C.white,
  });
  text(
    slide,
    "deck-subtitle",
    "从 DM 数据到本地神经网络，再到日报、回测和人工决策辅助",
    90,
    306,
    930,
    50,
    { size: 28, color: "#E9F1FA" },
  );
  text(slide, "date", `汇报日期：${state.runDate}`, 90, 456, 420, 34, {
    size: 24,
    color: C.white,
    bold: true,
  });
  text(slide, "position", "定位：不是自动下单系统，而是把研究流程结构化、可回测、可复盘的 AI 助手。", 90, 512, 900, 80, {
    size: 24,
    color: "#DDE8F3",
  });
}

function addSlide2(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "我们解决什么问题", "01 / 问题意识");
  text(slide, "big", "信用债判断不是看一个数字，而是同时看期限、利率、资金面、宏观和历史位置。", 72, 170, 980, 90, {
    size: 36,
    bold: true,
    color: C.ink,
  });
  const items = [
    ["噪声", "人容易被短期市场波动牵着走。", C.red],
    ["纪律", "同一套数据、同一套规则，每天自动复盘。", C.teal],
    ["证据", "把模型概率、阈值、回测和误判一起摆出来。", C.blue],
  ];
  items.forEach(([head, body, color], i) => {
    const x = 72 + i * 374;
    box(slide, `problem-${i}`, x, 315, 330, 200, { fill: C.white, line: C.line, shadow: "shadow-sm" });
    box(slide, `problem-rule-${i}`, x, 315, 330, 8, { geometry: "rect", fill: color, line: color, radius: "none" });
    text(slide, `problem-head-${i}`, head, x + 24, 348, 250, 38, { size: 32, bold: true, color });
    text(slide, `problem-body-${i}`, body, x + 24, 408, 270, 84, { size: 23, color: C.ink });
  });
  footer(slide, 2);
}

function addSlide3(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "全流程一张图", "02 / 从数据到报告");
  const steps = [
    ["DM API", "自动拉取 EDB 和收益率曲线", C.blue],
    ["清洗", "只向前填充，避免未来函数", C.teal],
    ["特征", "把数据变成模型可读的序列", C.amber],
    ["模型", "GRU / TCN / Transformer", C.lavender],
    ["信号", "三分类概率、集成、阈值", C.red],
    ["报告", "日报、回测、复盘", C.blue],
  ];
  steps.forEach(([head, body, color], i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 82 + col * 374;
    const y = 206 + row * 168;
    box(slide, `step-${i}`, x, y, 310, 132, { fill: C.white, line: C.line, shadow: "shadow-sm" });
    box(slide, `step-band-${i}`, x, y, 310, 8, { geometry: "rect", fill: color, line: color, radius: "none" });
    text(slide, `step-num-${i}`, String(i + 1), x + 22, y + 28, 36, 34, { size: 30, bold: true, color });
    text(slide, `step-head-${i}`, head, x + 70, y + 30, 170, 34, { size: 27, bold: true, color: C.ink });
    text(slide, `step-body-${i}`, body, x + 22, y + 80, 260, 34, { size: 19, color: C.muted });
  });
  text(slide, "bottom-callout", "核心原则：先把数据底座做稳，再谈神经网络；先做回测和复盘，再谈仓位。", 92, 560, 1000, 56, {
    size: 28,
    bold: true,
    color: C.ink,
  });
  footer(slide, 3);
}

function addSlide4(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "数据底座：比模型更重要", "03 / 防止 AI 学错");
  const items = [
    ["DM API 接管", "后续数据主线不再依赖 Wind 手工导出，按配置表自动抓取。", C.blue],
    ["数据字典", "每个指标记录代码、频率、发布时间、可获得时间和类别。", C.teal],
    ["只向前填充", "整理数据时只用历史值补今天，不能用未来值补过去。", C.red],
  ];
  items.forEach(([head, body, color], i) => {
    const x = 88 + i * 370;
    text(slide, `lane-num-${i}`, `0${i + 1}`, x, 190, 100, 48, { size: 38, bold: true, color });
    box(slide, `lane-line-${i}`, x, 248, 280, 5, { geometry: "rect", fill: color, line: color, radius: "none" });
    text(slide, `lane-head-${i}`, head, x, 285, 270, 40, { size: 31, bold: true, color: C.ink });
    text(slide, `lane-body-${i}`, body, x, 350, 292, 135, { size: 24, color: C.muted });
  });
  text(slide, "warning", "如果数据里混进未来信息，回测会很好看，但真实使用会失效。", 92, 560, 980, 42, {
    size: 30,
    bold: true,
    color: C.red,
  });
  footer(slide, 4);
}

function addSlide5(presentation, state) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "模型层：不是一个模型押注", "04 / 本地神经网络");
  text(slide, "stat", `${state.modelCount} 个模型`, 82, 190, 370, 70, { size: 54, bold: true, color: C.ink });
  text(slide, "stat-sub", "3 / 5 / 10 / 20 年 × GRU / TCN / Transformer", 86, 270, 470, 40, {
    size: 23,
    color: C.muted,
  });
  const models = [
    ["GRU", "像带记忆的读表器，逐日读取历史。", C.blue],
    ["TCN", "用不同长度的时间尺子看趋势。", C.teal],
    ["Transformer", "判断哪些历史片段更值得注意。", C.lavender],
  ];
  models.forEach(([head, body, color], i) => {
    const y = 338 + i * 92;
    box(slide, `model-dot-${i}`, 92, y + 8, 18, 18, { geometry: "ellipse", fill: color, line: color, radius: "none" });
    text(slide, `model-head-${i}`, head, 130, y, 260, 32, { size: 28, bold: true, color });
    text(slide, `model-body-${i}`, body, 130, y + 40, 530, 36, { size: 22, color: C.ink });
  });
  box(slide, "right-surface", 770, 180, 360, 330, { fill: C.white, line: C.line, shadow: "shadow-sm" });
  text(slide, "right-head", "每个期限单独训练", 810, 222, 300, 44, { size: 31, bold: true, color: C.ink });
  addBullet(
    slide,
    "right-bullets",
    [
      "3年模型只学3年",
      "5年、10年、20年同理",
      "最后按期限分别输出信号",
      "不用 10 年核心权重",
    ],
    810,
    290,
    290,
    150,
    { size: 23 },
  );
  footer(slide, 5);
}

function addSlide6(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "信号如何变成决策辅助", "05 / 概率、集成、阈值");
  const labels = [
    ["看空", "收益率上行概率高，价格承压", C.red],
    ["看多", "收益率下行概率高，价格受益", C.teal],
    ["震荡", "变化不明显或模型分歧", C.amber],
  ];
  labels.forEach(([head, body, color], i) => {
    const x = 84 + i * 358;
    box(slide, `signal-${i}`, x, 176, 300, 128, { fill: C.white, line: C.line, shadow: "shadow-sm" });
    text(slide, `signal-head-${i}`, head, x + 24, 200, 120, 36, { size: 32, bold: true, color });
    text(slide, `signal-body-${i}`, body, x + 24, 252, 240, 36, { size: 20, color: C.ink });
  });
  const flow = [
    ["单模型概率", "三个模型各自输出三类概率"],
    ["同期限集成", "GRU / TCN / Transformer 概率取平均"],
    ["阈值过滤", "方向优势不够时降级为震荡"],
    ["日报展示", "给研究员看概率、信号和解释"],
  ];
  flow.forEach(([head, body], i) => {
    const x = 95 + i * 280;
    text(slide, `flow-head-${i}`, head, x, 398, 210, 32, { size: 26, bold: true, color: C.ink });
    text(slide, `flow-body-${i}`, body, x, 445, 220, 58, { size: 20, color: C.muted });
    if (i < flow.length - 1) {
      text(slide, `flow-arrow-${i}`, "→", x + 222, 430, 36, 40, { size: 32, color: C.muted, bold: true });
    }
  });
  text(slide, "bottom", "目标不是每天都行动，而是把低置信度信号挡在决策门外。", 92, 570, 950, 40, {
    size: 30,
    bold: true,
    color: C.ink,
  });
  footer(slide, 6);
}

function addSlide7(presentation, state) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "当前实验结论", "06 / 先看稳健性，再看收益");
  box(slide, "chart-surface", 74, 176, 540, 360, { fill: C.white, line: C.line, shadow: "shadow-sm" });
  text(slide, "chart-title", "不新增数据实验：各期限最佳回测累计收益（%）", 104, 204, 460, 32, {
    size: 23,
    bold: true,
    color: C.ink,
  });
  slide.charts.add("bar", {
    position: { left: 104, top: 258, width: 460, height: 220 },
    categories: ["3年", "5年", "10年", "20年"],
    series: [{ name: "累计收益率", values: state.bestReturnsPct, fill: C.blue }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
    yAxis: {
      majorGridlines: { style: "solid", fill: C.line, width: 1 },
    },
  });
  text(slide, "chart-note", "注：这是研究 proxy 回测，不是收益承诺。", 104, 492, 440, 28, {
    size: 17,
    color: C.muted,
  });
  const findings = [
    ["10 年", "滚动验证目前相对最稳。", C.teal],
    ["20 年", "策略弹性强，但分类稳定性还要继续验证。", C.amber],
    ["3年和5年", "可以继续榨现有数据，但更需要提升历史长度和特征质量。", C.red],
  ];
  findings.forEach(([head, body, color], i) => {
    const y = 190 + i * 118;
    box(slide, `finding-bar-${i}`, 672, y, 8, 80, { geometry: "rect", fill: color, line: color, radius: "none" });
    text(slide, `finding-head-${i}`, head, 700, y - 4, 220, 40, { size: 31, bold: true, color });
    text(slide, `finding-body-${i}`, body, 700, y + 48, 390, 60, { size: 23, color: C.ink });
  });
  footer(slide, 7);
}

function addSlide8(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  text(slide, "title", "下一步路线", 82, 72, 620, 70, { size: 58, bold: true, color: C.white });
  text(slide, "subtitle", "目标不是神秘黑箱，而是每天可更新、可复盘、可解释的研究机器。", 86, 152, 900, 44, {
    size: 27,
    color: "#DDE8F3",
  });
  const phases = [
    ["1", "补历史", "优先提升 CFETS 历史长度和可获得时间质量", C.blue],
    ["2", "稳模型", "滚动验证、概率校准、标签阈值实验", C.teal],
    ["3", "接组合", "加入久期、carry、交易成本和仓位约束", C.amber],
    ["4", "做闭环", "日报、误判复盘、人工备注、再训练", C.red],
  ];
  phases.forEach(([num, head, body, color], i) => {
    const x = 82 + i * 292;
    box(slide, `phase-line-${i}`, x, 288, 220, 5, { geometry: "rect", fill: color, line: color, radius: "none" });
    text(slide, `phase-num-${i}`, num, x, 318, 44, 44, { size: 38, bold: true, color });
    text(slide, `phase-head-${i}`, head, x + 54, 323, 150, 36, { size: 30, bold: true, color: C.white });
    text(slide, `phase-body-${i}`, body, x, 390, 220, 98, { size: 22, color: "#E9F1FA" });
  });
  coloredRule(slide, 82, 576, 660);
  text(slide, "close", "AI 给结构化证据和纪律，人负责解释环境与最终判断。", 82, 606, 900, 40, {
    size: 28,
    bold: true,
    color: C.white,
  });
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });
  await fs.mkdir(path.dirname(finalPptx), { recursive: true });

  await fs.writeFile(
    path.join(__dirname, "source-notes.txt"),
    [
      "No bundled default template was available in this runtime, so the deck was created from scratch.",
      "Local sources used: daily_dm_update_report.json and no-data experiment CSV summaries.",
      "No internet research or external market claims were used.",
    ].join("\n"),
    "utf-8",
  );

  const state = await collectState();
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  addSlide1(presentation, state);
  addSlide2(presentation, state);
  addSlide3(presentation, state);
  addSlide4(presentation, state);
  addSlide5(presentation, state);
  addSlide6(presentation, state);
  addSlide7(presentation, state);
  addSlide8(presentation, state);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(previewDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(layoutDir, `${stem}.layout.json`), await layout.text(), "utf-8");
  }

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(previewDir, "deck-montage.webp"), montage);
  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,chart,table",
    maxChars: 30000,
  });
  await fs.writeFile(path.join(qaDir, "inspect.ndjson"), inspect.ndjson, "utf-8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  console.log(finalPptx);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
