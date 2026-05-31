// Term project presentation builder.
// Multilingual Toxic Comment Classification — SEDS 537, Spring 2026.
// Run: node presentation/build_deck.js   (requires global pptxgenjs)
//
// Design: "Midnight Executive" palette (navy / ice blue) with a teal accent.
// Motif: thick teal left-bar + numbered circle on every content slide.

const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.join(__dirname, "..", "figures");
const fig = (name) => path.join(FIG, name);

// ---- Palette ---------------------------------------------------------------
const NAVY   = "1E2761"; // primary dark
const NAVY2  = "151B45"; // deeper navy for gradients/footers
const ICE    = "CADCFC"; // light secondary
const TEAL   = "00A896"; // accent
const TEALD  = "028090"; // darker teal
const INK    = "243044"; // body text on light
const MUTE   = "6B7480"; // muted captions
const LIGHT  = "F4F7FB"; // light content background
const WHITE  = "FFFFFF";
const RED     = "C0392B"; // wrong / toxic
const GREEN   = "1E8449"; // right / clean

const HFONT = "Georgia";
const BFONT = "Calibri";

const pptx = new pptxgen();
pptx.defineLayout({ name: "W", width: 13.33, height: 7.5 });
pptx.layout = "W";
pptx.author = "SEDS 537 Term Project";
pptx.title = "Multilingual Toxic Comment Classification";

const W = 13.33, H = 7.5;

// ---- Helpers ---------------------------------------------------------------
let _slideNum = 0;   // content slides auto-number themselves (title slide excluded)
function contentSlide(title, kicker) {
  const num = ++_slideNum;
  const s = pptx.addSlide();
  s.background = { color: LIGHT };
  // left accent bar
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.22, h: H, fill: { color: TEAL } });
  // numbered circle
  s.addShape(pptx.ShapeType.ellipse, { x: 0.55, y: 0.5, w: 0.72, h: 0.72, fill: { color: NAVY } });
  s.addText(String(num).padStart(2, "0"), {
    x: 0.55, y: 0.5, w: 0.72, h: 0.72, align: "center", valign: "middle",
    fontFace: HFONT, fontSize: 22, bold: true, color: WHITE,
  });
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 1.45, y: 0.5, w: 11.2, h: 0.3, fontFace: BFONT, fontSize: 12,
      bold: true, color: TEALD, charSpacing: 2,
    });
  }
  s.addText(title, {
    x: 1.45, y: 0.74, w: 11.3, h: 0.7, fontFace: HFONT, fontSize: 30,
    bold: true, color: NAVY,
  });
  // footer
  s.addText("Multilingual Toxic Comment Classification  ·  SEDS 537", {
    x: 0.55, y: 7.05, w: 9, h: 0.3, fontFace: BFONT, fontSize: 9, color: MUTE,
  });
  s.addText(String(num), { x: 12.4, y: 7.05, w: 0.5, h: 0.3, fontFace: BFONT, fontSize: 9, color: MUTE, align: "right" });
  return s;
}

// bullet helper
function bullets(items, opts) {
  return items.map((it) => {
    if (typeof it === "string") return { text: it, options: { bullet: { code: "2022", indent: 16 }, paraSpaceAfter: 8, fontSize: opts?.fontSize || 16, color: INK } };
    return it;
  });
}

// small card
function card(s, x, y, w, h, fill) {
  s.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || WHITE }, line: { color: "E2E8F0", width: 1 }, shadow: { type: "outer", color: "AAB4C2", blur: 6, offset: 2, angle: 90, opacity: 0.35 } });
}

// =====================================================================
// SLIDE 1 — TITLE
// =====================================================================
{
  const s = pptx.addSlide();
  s.background = { color: NAVY };
  // decorative motif: teal rounded frames top-right
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.18, fill: { color: TEAL } });
  s.addShape(pptx.ShapeType.ellipse, { x: 10.4, y: -1.6, w: 4.2, h: 4.2, fill: { color: NAVY2 }, line: { color: TEAL, width: 1.5 } });
  s.addShape(pptx.ShapeType.ellipse, { x: 11.5, y: -0.7, w: 2.4, h: 2.4, fill: { color: TEAL }, line: { type: "none" } });

  s.addText("SEDS 537 · MACHINE LEARNING · TERM PROJECT · SPRING 2026", {
    x: 0.9, y: 1.5, w: 11, h: 0.4, fontFace: BFONT, fontSize: 14, bold: true, color: TEAL, charSpacing: 2,
  });
  s.addText("Multilingual Toxic\nComment Classification", {
    x: 0.85, y: 2.0, w: 11.5, h: 2.0, fontFace: HFONT, fontSize: 48, bold: true, color: WHITE, lineSpacingMultiple: 1.0,
  });
  s.addText("Closing the cross-lingual gap with parameter-efficient mBERT adapters", {
    x: 0.9, y: 4.05, w: 11, h: 0.6, fontFace: BFONT, fontSize: 20, italic: true, color: ICE,
  });
  // divider
  s.addShape(pptx.ShapeType.rect, { x: 0.92, y: 4.85, w: 2.2, h: 0.05, fill: { color: TEAL } });
  s.addText([
    { text: "TF-IDF baselines", options: {} },
    { text: "   ·   ", options: { color: TEAL } },
    { text: "BiLSTM", options: {} },
    { text: "   ·   ", options: { color: TEAL } },
    { text: "mBERT fine-tuning & adapters", options: {} },
  ], { x: 0.9, y: 5.05, w: 11, h: 0.4, fontFace: BFONT, fontSize: 15, color: ICE });

  s.addText("Presentation: 1 / 8 June 2026, 16:30  ·  12 min talk + Q&A", {
    x: 0.9, y: 6.55, w: 11.5, h: 0.4, fontFace: BFONT, fontSize: 12, color: "8FA0C8",
  });
}

// =====================================================================
// SLIDE 2 — PROBLEM DEFINITION
// =====================================================================
{
  const s = contentSlide("Problem Definition", "What & Why");
  s.addText([
    { text: "The task: ", options: { bold: true, color: NAVY } },
    { text: "given an online comment, decide whether it is ", options: {} },
    { text: "toxic", options: { bold: true, color: RED } },
    { text: " or ", options: {} },
    { text: "non-toxic", options: { bold: true, color: GREEN } },
    { text: " — a binary classification problem.", options: {} },
  ], { x: 1.45, y: 1.7, w: 6.7, h: 0.9, fontFace: BFONT, fontSize: 17, color: INK, lineSpacingMultiple: 1.15 });

  s.addText(bullets([
    "Platforms host comments in dozens of languages, but labelled toxicity data is overwhelmingly English.",
    "Hand-labelling every language is expensive and slow — moderation must still work everywhere.",
    "Goal: train on English only, then generalise to languages never seen during training (cross-lingual transfer).",
  ], { fontSize: 16 }), { x: 1.45, y: 2.7, w: 6.7, h: 2.6 });

  // right side stat cards
  const cx = 8.6, cw = 4.1;
  card(s, cx, 1.7, cw, 1.5, NAVY);
  s.addText("223K", { x: cx, y: 1.85, w: cw, h: 0.7, align: "center", fontFace: HFONT, fontSize: 40, bold: true, color: TEAL });
  s.addText("English training comments", { x: cx, y: 2.55, w: cw, h: 0.5, align: "center", fontFace: BFONT, fontSize: 13, color: ICE });

  card(s, cx, 3.35, cw, 1.5, WHITE);
  s.addText("3 langs", { x: cx, y: 3.5, w: cw, h: 0.7, align: "center", fontFace: HFONT, fontSize: 36, bold: true, color: NAVY });
  s.addText("evaluated unseen: Turkish · Spanish · Italian", { x: cx, y: 4.2, w: cw, h: 0.5, align: "center", fontFace: BFONT, fontSize: 13, color: MUTE });

  card(s, cx, 5.0, cw, 1.5, WHITE);
  s.addText("~10%", { x: cx, y: 5.15, w: cw, h: 0.7, align: "center", fontFace: HFONT, fontSize: 36, bold: true, color: RED });
  s.addText("toxic — a strongly imbalanced target", { x: cx, y: 5.85, w: cw, h: 0.5, align: "center", fontFace: BFONT, fontSize: 13, color: MUTE });
}

// =====================================================================
// SLIDE 3 — THE CROSS-LINGUAL CHALLENGE
// =====================================================================
{
  const s = contentSlide("The Cross-Lingual Challenge", "Why it is hard");
  s.addText("Why does an English-trained model struggle in Turkish or Spanish?", {
    x: 1.45, y: 1.65, w: 11, h: 0.5, fontFace: BFONT, fontSize: 18, italic: true, color: TEALD,
  });

  const items = [
    ["Vocabulary mismatch", "Word-level models learn an English vocabulary. Turkish / Spanish words are simply unknown tokens — the signal disappears."],
    ["No shared meaning", "TF-IDF and LSTM treat 'idiot' and 'aptal' as unrelated symbols. They cannot transfer what 'toxic' looks like across languages."],
    ["Different scripts & morphology", "Diacritics, agglutination and word order differ — surface features learned on English do not carry over."],
    ["Our hypothesis", "A multilingual pretrained encoder (mBERT) already maps all languages into one shared space, so toxicity learned in English transfers."],
  ];
  let y = 2.35;
  items.forEach(([h, b], i) => {
    const accent = i === 3 ? TEAL : NAVY;
    s.addShape(pptx.ShapeType.roundRect, { x: 1.45, y, w: 11.3, h: 1.0, rectRadius: 0.06, fill: { color: i === 3 ? "E6F7F4" : WHITE }, line: { color: i === 3 ? TEAL : "E2E8F0", width: i === 3 ? 1.5 : 1 } });
    s.addShape(pptx.ShapeType.rect, { x: 1.45, y, w: 0.1, h: 1.0, fill: { color: accent } });
    s.addText(h, { x: 1.75, y: y + 0.1, w: 3.4, h: 0.8, valign: "middle", fontFace: HFONT, fontSize: 16, bold: true, color: accent });
    s.addText(b, { x: 5.2, y: y + 0.08, w: 7.4, h: 0.85, valign: "middle", fontFace: BFONT, fontSize: 14, color: INK });
    y += 1.12;
  });
}

// =====================================================================
// SLIDE 4 — DATASET & EXAMPLES
// =====================================================================
{
  const s = contentSlide("Dataset & Examples", "Jigsaw Multilingual");
  s.addText(bullets([
    "Source: Jigsaw Multilingual Toxic Comment Classification (Kaggle).",
    "Train: ~223K English comments, binary label.",
    "Validation: ~8K labelled multilingual (tr 3.0K · es 2.5K · it 2.5K).",
    "Test: ~63K multilingual but unlabelled — so labelled cross-lingual evaluation uses the validation split.",
  ], { fontSize: 14 }), { x: 1.45, y: 1.7, w: 5.55, h: 2.9 });

  // example snippets on the right
  const ex = [
    [GREEN, "NON-TOXIC · en", "“You, sir, are my hero. Any chance you remember what page that’s on?”"],
    [RED, "TOXIC · en", "“COCKS∗∗∗ER, before you piss around on my work…”"],
    [RED, "TOXIC · it", "“Incazzato come sei, non sei pure tu un sockpuppet…”"],
    [GREEN, "NON-TOXIC · es", "“Muchas gracias, LlamaAl. Sí que parece útil…”"],
  ];
  let y = 1.7, x = 7.25, w = 5.45;
  ex.forEach(([c, tag, txt]) => {
    s.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 1.18, rectRadius: 0.06, fill: { color: WHITE }, line: { color: "E2E8F0", width: 1 } });
    s.addShape(pptx.ShapeType.rect, { x, y, w: 0.1, h: 1.18, fill: { color: c } });
    s.addText(tag, { x: x + 0.25, y: y + 0.1, w: w - 0.4, h: 0.3, fontFace: BFONT, fontSize: 11, bold: true, color: c, charSpacing: 1 });
    s.addText(txt, { x: x + 0.25, y: y + 0.38, w: w - 0.45, h: 0.75, fontFace: BFONT, fontSize: 12.5, italic: true, color: INK });
    y += 1.3;
  });
  s.addText("Examples shortened / lightly censored for the slide.", { x: 7.25, y: 6.95, w: 5.4, h: 0.3, fontFace: BFONT, fontSize: 9, italic: true, color: MUTE, align: "right" });
}

// =====================================================================
// SLIDE 5 — METHODOLOGY OVERVIEW  (what we did / why)
// =====================================================================
{
  const s = contentSlide("Our Approach", "What we did & why");
  s.addText("A ladder of models, from simple to powerful — each answers a question the previous one could not.", {
    x: 1.45, y: 1.65, w: 11.2, h: 0.5, fontFace: BFONT, fontSize: 16, italic: true, color: TEALD,
  });

  const steps = [
    ["1", "TF-IDF baselines", "LR · NB · SVM", "How far do classic word-count models get? Establishes the floor.", NAVY],
    ["2", "BiLSTM", "recurrent net", "Does modelling word order help? A deep-learning baseline.", NAVY],
    ["3", "mBERT (full)", "fine-tune all", "Can a multilingual transformer transfer? The strong baseline.", TEALD],
    ["4", "mBERT + Adapters", "proposed method", "Same power for ~1% of the cost? Our parameter-efficient method.", TEAL],
  ];
  const cw = 2.78, gap = 0.18; let x = 1.45;
  steps.forEach(([n, title, sub, why, c]) => {
    s.addShape(pptx.ShapeType.roundRect, { x, y: 2.4, w: cw, h: 3.4, rectRadius: 0.08, fill: { color: WHITE }, line: { color: c, width: 1.5 } });
    s.addShape(pptx.ShapeType.rect, { x, y: 2.4, w: cw, h: 0.72, rectRadius: 0, fill: { color: c } });
    s.addText(title, { x: x + 0.1, y: 2.45, w: cw - 0.2, h: 0.62, align: "center", valign: "middle", fontFace: HFONT, fontSize: 15.5, bold: true, color: WHITE });
    s.addShape(pptx.ShapeType.ellipse, { x: x + cw / 2 - 0.33, y: 3.0, w: 0.66, h: 0.66, fill: { color: c } });
    s.addText(n, { x: x + cw / 2 - 0.33, y: 3.0, w: 0.66, h: 0.66, align: "center", valign: "middle", fontFace: HFONT, fontSize: 22, bold: true, color: WHITE });
    s.addText(sub.toUpperCase(), { x: x + 0.1, y: 3.72, w: cw - 0.2, h: 0.3, align: "center", fontFace: BFONT, fontSize: 11, bold: true, color: MUTE, charSpacing: 1 });
    s.addText(why, { x: x + 0.22, y: 4.1, w: cw - 0.44, h: 1.6, align: "center", fontFace: BFONT, fontSize: 13, color: INK });
    x += cw + gap;
  });

  // arrow strip
  s.addText("simple  →  increasing capacity & cross-lingual ability  →  efficient", {
    x: 1.45, y: 6.0, w: 11.3, h: 0.4, align: "center", fontFace: BFONT, fontSize: 13, italic: true, color: MUTE,
  });
}

// =====================================================================
// SLIDE 6 — MODEL EXPLAINERS (TF-IDF / LSTM / mBERT)
// =====================================================================
{
  const s = contentSlide("How the Models Work", "Quick primer");
  const cols = [
    [NAVY, "TF-IDF + classifier", "Bag of words", [
      "Counts how often each word appears, weighted by rarity.",
      "Ignores word order and meaning.",
      "Vocabulary is English — non-English words vanish.",
      "Fast, strong on English, no transfer.",
    ]],
    [TEALD, "BiLSTM", "Recurrent net", [
      "Reads the sentence left→right and right→left.",
      "Captures word order & local context.",
      "Embeddings learned from English only.",
      "Better English, still no real transfer.",
    ]],
    [TEAL, "mBERT", "Multilingual transformer", [
      "Transformer pretrained on 104 languages.",
      "Self-attention links every word to every other.",
      "All languages share one embedding space.",
      "Toxicity learned in English transfers out.",
    ]],
  ];
  const cw = 3.78, gap = 0.18; let x = 1.45;
  cols.forEach(([c, title, sub, lines]) => {
    s.addShape(pptx.ShapeType.roundRect, { x, y: 1.75, w: cw, h: 4.95, rectRadius: 0.08, fill: { color: WHITE }, line: { color: "E2E8F0", width: 1 } });
    s.addShape(pptx.ShapeType.roundRect, { x, y: 1.75, w: cw, h: 0.95, rectRadius: 0.08, fill: { color: c } });
    s.addShape(pptx.ShapeType.rect, { x, y: 2.35, w: cw, h: 0.35, fill: { color: c } });
    s.addText(title, { x: x + 0.15, y: 1.82, w: cw - 0.3, h: 0.5, align: "center", fontFace: HFONT, fontSize: 18, bold: true, color: WHITE });
    s.addText(sub.toUpperCase(), { x: x + 0.15, y: 2.32, w: cw - 0.3, h: 0.35, align: "center", fontFace: BFONT, fontSize: 11, bold: true, color: "FFFFFF", charSpacing: 1 });
    s.addText(lines.map((t) => ({ text: t, options: { bullet: { code: "2022", indent: 14 }, paraSpaceAfter: 10, fontSize: 13.5, color: INK } })), { x: x + 0.28, y: 2.95, w: cw - 0.5, h: 3.6, valign: "top" });
    x += cw + gap;
  });
}

// =====================================================================
// SLIDE 6b — TF-IDF + LOGISTIC REGRESSION / SVM, EXPLAINED WITH SHAPES
// =====================================================================
{
  const s = contentSlide("A Simple Model, Step by Step", "TF-IDF + LogReg / SVM");
  s.addText("How does the simplest model decide? It just adds up a score for the words it knows.", {
    x: 1.45, y: 1.6, w: 11.2, h: 0.45, fontFace: BFONT, fontSize: 16, italic: true, color: TEALD,
  });

  // ----- horizontal pipeline of 4 stages -----
  const stageY = 2.3, stageH = 0.6, stageW = 2.55, gap = 0.43;
  const stages = [
    ["1 · The comment", NAVY],
    ["2 · Split into words", NAVY],
    ["3 · Weigh each word", TEALD],
    ["4 · Add up → decide", TEAL],
  ];
  let sx = 1.45;
  stages.forEach(([t, c], i) => {
    s.addShape(pptx.ShapeType.roundRect, { x: sx, y: stageY, w: stageW, h: stageH, rectRadius: 0.06, fill: { color: c }, line: { type: "none" } });
    s.addText(t, { x: sx, y: stageY, w: stageW, h: stageH, align: "center", valign: "middle", fontFace: BFONT, fontSize: 13, bold: true, color: WHITE });
    if (i < stages.length - 1) {
      s.addText("→", { x: sx + stageW, y: stageY, w: gap, h: stageH, align: "center", valign: "middle", fontFace: HFONT, fontSize: 22, bold: true, color: MUTE });
    }
    sx += stageW + gap;
  });

  // ----- worked example below each stage -----
  const exY = 3.25;
  // Stage 1: the raw comment
  card(s, 1.45, exY, stageW, 2.4, WHITE);
  s.addText("“You are so stupid, just shut up.”", { x: 1.6, y: exY + 0.2, w: stageW - 0.3, h: 2.0, valign: "middle", fontFace: BFONT, fontSize: 14, italic: true, color: INK });

  // Stage 2: split into words (chips)
  const x2 = 1.45 + (stageW + gap);
  card(s, x2, exY, stageW, 2.4, WHITE);
  const words = ["you", "are", "so", "stupid", "just", "shut", "up"];
  let wy = exY + 0.18;
  words.forEach((w) => {
    s.addShape(pptx.ShapeType.roundRect, { x: x2 + 0.55, y: wy, w: stageW - 1.1, h: 0.26, rectRadius: 0.04, fill: { color: "EEF3FA" }, line: { color: "D5DEEA", width: 0.5 } });
    s.addText(w, { x: x2 + 0.55, y: wy, w: stageW - 1.1, h: 0.26, align: "center", valign: "middle", fontFace: BFONT, fontSize: 11, color: INK });
    wy += 0.31;
  });

  // Stage 3: per-word weights table (rare + toxic words score high)
  const x3 = 1.45 + 2 * (stageW + gap);
  card(s, x3, exY, stageW, 2.4, WHITE);
  const weights = [
    ["stupid", "+2.4", RED],
    ["shut", "+1.1", RED],
    ["you", "+0.1", MUTE],
    ["so", "+0.0", MUTE],
    ["are", "−0.2", GREEN],
  ];
  let wy3 = exY + 0.22;
  s.addText("word → weight", { x: x3 + 0.2, y: exY + 0.02, w: stageW - 0.4, h: 0.22, align: "center", fontFace: BFONT, fontSize: 10, bold: true, color: MUTE });
  weights.forEach(([w, val, c]) => {
    s.addText(w, { x: x3 + 0.3, y: wy3, w: 1.3, h: 0.32, valign: "middle", fontFace: BFONT, fontSize: 12.5, color: INK });
    s.addText(val, { x: x3 + stageW - 1.3, y: wy3, w: 1.0, h: 0.32, align: "right", valign: "middle", fontFace: BFONT, fontSize: 12.5, bold: true, color: c });
    wy3 += 0.4;
  });

  // Stage 4: sum → sigmoid → decision
  const x4 = 1.45 + 3 * (stageW + gap);
  card(s, x4, exY, stageW, 2.4, NAVY);
  s.addText("sum of weights", { x: x4, y: exY + 0.2, w: stageW, h: 0.3, align: "center", fontFace: BFONT, fontSize: 11, color: ICE });
  s.addText("+3.4", { x: x4, y: exY + 0.45, w: stageW, h: 0.6, align: "center", fontFace: HFONT, fontSize: 30, bold: true, color: TEAL });
  s.addText("→  toxic probability", { x: x4, y: exY + 1.1, w: stageW, h: 0.3, align: "center", fontFace: BFONT, fontSize: 11, color: ICE });
  s.addText("0.94", { x: x4, y: exY + 1.35, w: stageW, h: 0.5, align: "center", fontFace: HFONT, fontSize: 26, bold: true, color: WHITE });
  s.addShape(pptx.ShapeType.roundRect, { x: x4 + 0.5, y: exY + 1.95, w: stageW - 1.0, h: 0.36, rectRadius: 0.05, fill: { color: RED }, line: { type: "none" } });
  s.addText("TOXIC", { x: x4 + 0.5, y: exY + 1.95, w: stageW - 1.0, h: 0.36, align: "center", valign: "middle", fontFace: BFONT, fontSize: 13, bold: true, color: WHITE });

  // ----- two takeaway notes -----
  s.addText([
    { text: "TF-IDF", options: { bold: true, color: TEALD } },
    { text: " = how often a word appears, down-weighted if it is common everywhere. ", options: {} },
    { text: "LogReg / SVM", options: { bold: true, color: TEALD } },
    { text: " just learn one +/− weight per word and add them up.", options: {} },
  ], { x: 1.45, y: 5.95, w: 11.3, h: 0.55, fontFace: BFONT, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.1 });
  s.addText("Catch: it knows no word order and only English words — so “aptal” or “stupido” score 0, and transfer collapses.", {
    x: 1.45, y: 6.5, w: 11.3, h: 0.4, fontFace: BFONT, fontSize: 12.5, italic: true, color: RED,
  });
}

// =====================================================================
// SLIDE 7 — PROPOSED METHOD: ADAPTERS
// =====================================================================
{
  const s = contentSlide("Proposed Method: Adapter Tuning", "The simple idea");
  s.addText([
    { text: "Think of mBERT as a huge brain already trained on 104 languages — about ", options: {} },
    { text: "178 million tiny “knobs.”", options: { bold: true, color: NAVY } },
  ], { x: 1.45, y: 1.65, w: 11.2, h: 0.5, fontFace: BFONT, fontSize: 17, italic: true, color: TEALD });

  s.addText(bullets([
    "The usual way (full fine-tuning) re-tunes ALL 178M knobs for our task — slow, and you end up storing a whole new copy of the model.",
    "Adapters: we FREEZE the big brain and snap in a few small new layers inside it.",
    "We then train ONLY those small layers — about 1 knob in every 75 (1.34%).",
    "The pretrained knowledge stays intact; the adapters just nudge it toward toxicity detection.",
  ], { fontSize: 15.5 }), { x: 1.45, y: 2.35, w: 6.65, h: 3.4 });

  // simple "what trains vs what is frozen" diagram on the right
  const dx = 8.45, dw = 4.3;
  card(s, dx, 2.2, dw, 4.2, WHITE);
  s.addText("What actually trains?", { x: dx, y: 2.35, w: dw, h: 0.4, align: "center", fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });

  // big frozen mBERT box
  s.addShape(pptx.ShapeType.roundRect, { x: dx + 0.45, y: 2.9, w: dw - 0.9, h: 2.2, rectRadius: 0.06, fill: { color: "E8EDF5" }, line: { color: "B9C4D6", width: 1 } });
  s.addText("mBERT encoder · 178M weights\n❄ FROZEN (98.66%)", { x: dx + 0.45, y: 2.98, w: dw - 0.9, h: 0.7, align: "center", fontFace: BFONT, fontSize: 12.5, bold: true, color: "55617A" });
  // adapter chips inside
  const chipY = 3.7, chipW = (dw - 1.4) / 3, chipGap = 0.1;
  for (let i = 0; i < 3; i++) {
    const cxp = dx + 0.6 + i * (chipW + chipGap);
    s.addShape(pptx.ShapeType.roundRect, { x: cxp, y: chipY, w: chipW, h: 1.2, rectRadius: 0.05, fill: { color: TEAL }, line: { type: "none" } });
    s.addText("adapter", { x: cxp, y: chipY, w: chipW, h: 1.2, align: "center", valign: "middle", fontFace: BFONT, fontSize: 11, bold: true, color: WHITE });
  }
  // classifier head
  s.addShape(pptx.ShapeType.roundRect, { x: dx + 0.45, y: 5.25, w: dw - 0.9, h: 0.5, rectRadius: 0.05, fill: { color: TEALD }, line: { type: "none" } });
  s.addText("classifier head (trained)", { x: dx + 0.45, y: 5.25, w: dw - 0.9, h: 0.5, align: "center", valign: "middle", fontFace: BFONT, fontSize: 12, bold: true, color: WHITE });
  s.addText("green = trained (1.34%)   ·   grey = frozen", { x: dx, y: 5.9, w: dw, h: 0.35, align: "center", fontFace: BFONT, fontSize: 11, italic: true, color: MUTE });
}

// =====================================================================
// SLIDE 8 — EXPERIMENTAL DESIGN
// =====================================================================
{
  const s = contentSlide("Experimental Design", "Implementation details");
  const blocks = [
    ["Splits", ["80/20 English split for in-language test", "Full multilingual validation for transfer", "Validation order preserved for error analysis"]],
    ["Imbalance", ["Toxic is rare — weight it ~9.5× in the loss", "Tune the decision cut-off, not just 0.5", "so we don't just predict 'clean' for everything"]],
    ["Early stopping", ["Keep the epoch with best multilingual score", "Word models overfit English and peak early", "avoids reporting a worse, later epoch"]],
    ["Metrics", ["AUC-ROC, F1, accuracy, precision, recall", "Per-language breakdown (tr / es / it)", "Trainable params + ms / sample"]],
    ["Cleaning", ["Aggressive ASCII clean → TF-IDF", "Unicode-preserving clean → LSTM / mBERT", "by design, to expose the gap"]],
    ["Hardware & time", ["Single NVIDIA RTX 3060 Ti GPU", "TF-IDF baselines: ~8–12 min · mBERT: ~2–3 h", "PyTorch + HuggingFace + scikit-learn"]],
  ];
  const cw = 3.78, ch = 2.2, gx = 0.18, gy = 0.25;
  let i = 0;
  for (let r = 0; r < 2; r++) {
    for (let cI = 0; cI < 3; cI++) {
      const x = 1.45 + cI * (cw + gx);
      const y = 1.8 + r * (ch + gy);
      const [h, ls] = blocks[i++];
      s.addShape(pptx.ShapeType.roundRect, { x, y, w: cw, h: ch, rectRadius: 0.07, fill: { color: WHITE }, line: { color: "E2E8F0", width: 1 } });
      s.addShape(pptx.ShapeType.rect, { x, y, w: cw, h: 0.5, fill: { color: NAVY } });
      s.addText(h, { x: x + 0.2, y, w: cw - 0.4, h: 0.5, valign: "middle", fontFace: HFONT, fontSize: 15, bold: true, color: WHITE });
      s.addText(ls.map((t) => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, paraSpaceAfter: 5, fontSize: 11.5, color: INK } })), { x: x + 0.28, y: y + 0.58, w: cw - 0.5, h: ch - 0.65, valign: "top" });
    }
  }
}

// =====================================================================
// SLIDE 9 — KEY FINDINGS (table + chart)
// =====================================================================
{
  const s = contentSlide("Key Findings", "Results");
  // results table
  const rows = [
    [
      { text: "Model", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "left" } },
      { text: "English AUC", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } },
      { text: "Multilingual AUC", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } },
      { text: "Trainable params", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } },
    ],
  ];
  const data = [
    ["TF-IDF + LogReg", "0.969", "0.627", "—"],
    ["TF-IDF + Naive Bayes", "0.943", "0.605", "—"],
    ["TF-IDF + SVM", "0.965", "0.582", "—"],
    ["BiLSTM", "0.936", "0.601", "~7M"],
    ["mBERT (full)", "0.961", "0.848", "100%"],
    ["mBERT + Adapter (ours)", "0.973", "0.826", "1.34%"],
  ];
  data.forEach((d, ri) => {
    const isStar = ri === data.length - 1;
    const bg = isStar ? "E6F7F4" : (ri % 2 ? "EEF3FA" : WHITE);
    rows.push(d.map((c, ci) => ({
      text: c,
      options: {
        align: ci === 0 ? "left" : "center",
        bold: isStar,
        color: isStar ? TEALD : INK,
        fill: { color: bg },
        fontSize: 14,
      },
    })));
  });
  s.addTable(rows, { x: 1.45, y: 1.85, w: 6.7, colW: [2.6, 1.35, 1.6, 1.15], rowH: 0.44, border: { type: "solid", color: "D5DEEA", pt: 1 }, fontFace: BFONT, valign: "middle" });

  s.addText([
    { text: "What to read here:  ", options: { bold: true, color: TEALD } },
    { text: "every model is strong on English (~0.94–0.97). But on ", options: {} },
    { text: "unseen languages", options: { bold: true } },
    { text: " TF-IDF & LSTM collapse to ~0.6 (little better than guessing), while mBERT + adapters holds ", options: {} },
    { text: "0.83", options: { bold: true, color: TEAL } },
    { text: " — proving cross-lingual transfer, at ~1% of the training cost.", options: {} },
  ], { x: 1.45, y: 4.95, w: 6.7, h: 1.5, fontFace: BFONT, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.12, valign: "top" });

  // native bar chart: multilingual AUC
  const chartData = [{
    name: "Multilingual AUC",
    labels: ["LogReg", "Naive Bayes", "SVM", "BiLSTM", "mBERT full", "mBERT+Adapter"],
    values: [0.627, 0.605, 0.582, 0.601, 0.848, 0.826],
  }];
  s.addChart(pptx.ChartType.bar, chartData, {
    x: 8.4, y: 1.85, w: 4.4, h: 4.7,
    barDir: "col",
    chartColors: [NAVY, NAVY, NAVY, NAVY, TEALD, TEAL],
    chartColorsOpacity: [70, 70, 70, 70, 100, 100],
    showValue: true, dataLabelColor: INK, dataLabelFontSize: 10, dataLabelFormatCode: "0.00",
    valAxisMinVal: 0.5, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.1,
    catAxisLabelColor: INK, catAxisLabelFontSize: 9,
    valAxisLabelColor: MUTE, valAxisLabelFontSize: 9,
    showLegend: false, showTitle: true, title: "Cross-lingual AUC by model", titleColor: NAVY, titleFontSize: 13, titleFontFace: HFONT,
  });
}

// =====================================================================
// SLIDE 9b — KEY FINDINGS (TURKISH VERSION)
// =====================================================================
{
  const s = contentSlide("Temel Bulgular", "Sonuçlar (Türkçe)");
  // sonuç tablosu
  const rows = [
    [
      { text: "Model", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "left" } },
      { text: "İngilizce AUC", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } },
      { text: "Çok dilli AUC", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } },
      { text: "Eğitilen parametre", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } },
    ],
  ];
  const data = [
    ["TF-IDF + Lojistik Reg.", "0.969", "0.627", "—"],
    ["TF-IDF + Naive Bayes", "0.943", "0.605", "—"],
    ["TF-IDF + SVM", "0.965", "0.582", "—"],
    ["BiLSTM", "0.936", "0.601", "~7M"],
    ["mBERT (tam eğitim)", "0.961", "0.848", "%100"],
    ["mBERT + Adapter (bizim)", "0.973", "0.826", "%1.34"],
  ];
  data.forEach((d, ri) => {
    const isStar = ri === data.length - 1;
    const bg = isStar ? "E6F7F4" : (ri % 2 ? "EEF3FA" : WHITE);
    rows.push(d.map((c, ci) => ({
      text: c,
      options: {
        align: ci === 0 ? "left" : "center",
        bold: isStar,
        color: isStar ? TEALD : INK,
        fill: { color: bg },
        fontSize: 14,
      },
    })));
  });
  s.addTable(rows, { x: 1.45, y: 1.85, w: 6.7, colW: [2.6, 1.35, 1.6, 1.15], rowH: 0.52, border: { type: "solid", color: "D5DEEA", pt: 1 }, fontFace: BFONT, valign: "middle" });

  s.addText([
    { text: "Buradan ne anlamalı:  ", options: { bold: true, color: TEALD } },
    { text: "her model İngilizcede güçlü (~0.94–0.97). Ama ", options: {} },
    { text: "hiç görmediği dillerde", options: { bold: true } },
    { text: " TF-IDF ve LSTM ~0.6’ya düşüyor (neredeyse rastgele), mBERT + adapter ise ", options: {} },
    { text: "0.83", options: { bold: true, color: TEAL } },
    { text: "’te kalıyor — diller arası transferi, eğitim maliyetinin ~%1’iyle kanıtlıyor.", options: {} },
  ], { x: 1.45, y: 4.95, w: 6.7, h: 1.5, fontFace: BFONT, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.12, valign: "top" });

  // çok dilli AUC grafiği
  const chartData = [{
    name: "Çok dilli AUC",
    labels: ["LojReg", "Naive Bayes", "SVM", "BiLSTM", "mBERT tam", "mBERT+Adapter"],
    values: [0.627, 0.605, 0.582, 0.601, 0.848, 0.826],
  }];
  s.addChart(pptx.ChartType.bar, chartData, {
    x: 8.4, y: 1.85, w: 4.4, h: 4.7,
    barDir: "col",
    chartColors: [NAVY, NAVY, NAVY, NAVY, TEALD, TEAL],
    chartColorsOpacity: [70, 70, 70, 70, 100, 100],
    showValue: true, dataLabelColor: INK, dataLabelFontSize: 10, dataLabelFormatCode: "0.00",
    valAxisMinVal: 0.5, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.1,
    catAxisLabelColor: INK, catAxisLabelFontSize: 9,
    valAxisLabelColor: MUTE, valAxisLabelFontSize: 9,
    showLegend: false, showTitle: true, title: "Modele göre çok dilli AUC", titleColor: NAVY, titleFontSize: 13, titleFontFace: HFONT,
  });
}

// =====================================================================
// SLIDE 10 — PER-LANGUAGE PERFORMANCE
// =====================================================================
{
  const s = contentSlide("Per-Language Transfer", "Where it works");
  // grouped bar: per language AUC for adapter vs a baseline (LogReg)
  const chartData = [
    { name: "TF-IDF LogReg", labels: ["Turkish", "Spanish", "Italian"], values: [0.652, 0.614, 0.600] },
    { name: "mBERT + Adapter", labels: ["Turkish", "Spanish", "Italian"], values: [0.890, 0.822, 0.772] },
  ];
  s.addChart(pptx.ChartType.bar, chartData, {
    x: 1.45, y: 1.9, w: 6.7, h: 4.5,
    barDir: "col", barGrouping: "clustered",
    chartColors: [NAVY, TEAL],
    showValue: true, dataLabelColor: INK, dataLabelFontSize: 10, dataLabelFormatCode: "0.00",
    valAxisMinVal: 0.5, valAxisMaxVal: 0.95, valAxisMajorUnit: 0.1,
    catAxisLabelColor: INK, catAxisLabelFontSize: 12,
    valAxisLabelColor: MUTE, valAxisLabelFontSize: 9,
    showLegend: true, legendPos: "b", legendColor: INK, legendFontSize: 11,
    showTitle: true, title: "AUC by language", titleColor: NAVY, titleFontSize: 13, titleFontFace: HFONT,
  });

  s.addText(bullets([
    "Adapters lift every language well above the classical baseline.",
    "Turkish gains the most (0.65 → 0.89) despite being typologically far from English.",
    "Italian is hardest (0.77) — likely fewer toxic cues survive transfer.",
    "Confirms mBERT's shared space carries toxicity signal across scripts and families.",
  ], { fontSize: 15 }), { x: 8.45, y: 2.2, w: 4.35, h: 4.0 });
}

// =====================================================================
// SLIDE 10b — WHY DOES TRANSFER DIFFER BY LANGUAGE?
// =====================================================================
{
  const s = contentSlide("Why Does Transfer Differ by Language?", "Reading the numbers");
  s.addText("Same model, three languages, three different gains. Here is how we read those numbers — and where we are guessing.", {
    x: 1.45, y: 1.6, w: 11.2, h: 0.45, fontFace: BFONT, fontSize: 15.5, italic: true, color: TEALD,
  });

  const cards = [
    ["Turkish", "0.65 → 0.89", "+0.24", TEAL,
      "Biggest jump. Not because Turkish is close to English — it isn't. mBERT simply saw a lot of Turkish while pretraining, and the classical baseline started very low, so there was the most room to gain."],
    ["Spanish", "0.61 → 0.82", "+0.21", TEALD,
      "Solid gain. Shares the Latin script and many roots with English, so toxic cues line up fairly well in mBERT's shared space."],
    ["Italian", "0.60 → 0.77", "+0.17", NAVY,
      "Smallest gain / hardest. Likely fewer toxic words overlap with what was seen in English training, so subtler insults slip through. (Partly our interpretation.)"],
  ];
  const cw = 3.78, gap = 0.18; let x = 1.45;
  cards.forEach(([lang, arrow, delta, c, why]) => {
    s.addShape(pptx.ShapeType.roundRect, { x, y: 2.3, w: cw, h: 3.75, rectRadius: 0.08, fill: { color: WHITE }, line: { color: c, width: 1.5 } });
    s.addShape(pptx.ShapeType.rect, { x, y: 2.3, w: cw, h: 0.62, fill: { color: c } });
    s.addText(lang, { x, y: 2.3, w: cw, h: 0.62, align: "center", valign: "middle", fontFace: HFONT, fontSize: 18, bold: true, color: WHITE });
    s.addText(arrow, { x, y: 3.05, w: cw, h: 0.4, align: "center", fontFace: BFONT, fontSize: 14, color: MUTE });
    s.addText(delta, { x, y: 3.4, w: cw, h: 0.6, align: "center", fontFace: HFONT, fontSize: 34, bold: true, color: c });
    s.addText("AUC gain", { x, y: 4.02, w: cw, h: 0.3, align: "center", fontFace: BFONT, fontSize: 11, color: MUTE, charSpacing: 1 });
    s.addText(why, { x: x + 0.28, y: 4.4, w: cw - 0.56, h: 1.55, fontFace: BFONT, fontSize: 12.5, color: INK, valign: "top" });
    x += cw + gap;
  });

  s.addText([
    { text: "Honest caveat:  ", options: { bold: true, color: RED } },
    { text: "these are interpretations of the scores, not controlled experiments. The clear, measured fact is the ranking — Turkish gains most, Italian least; the “why” is our best explanation.", options: {} },
  ], { x: 1.45, y: 6.2, w: 11.3, h: 0.7, fontFace: BFONT, fontSize: 13, color: INK, lineSpacingMultiple: 1.1 });
}

// =====================================================================
// SLIDE 11 — EFFICIENCY: ADAPTER vs FULL
// =====================================================================
{
  const s = contentSlide("Models 3 vs 4: Full vs Adapter", "What is different?");
  s.addText("Both start from the very same pre-trained mBERT. The only difference is how much of it we train.", {
    x: 1.45, y: 1.62, w: 11.2, h: 0.45, fontFace: BFONT, fontSize: 16, italic: true, color: TEALD,
  });

  // two comparison columns
  const rowsLabels = ["What gets trained", "Trainable params", "Stored per task", "Multilingual AUC", "Cost / speed"];
  const full = ["ALL of mBERT's weights", "100%  (~178M)", "a full ~700 MB copy", "~0.85  (slightly higher)", "heaviest — updates the whole network"];
  const adp  = ["only small adapters + head", "1.34%  (~2.4M)", "just a ~2 MB adapter", "~0.83  (on par)", "light — 16.9 ms per comment"];

  const colW = 5.5, colX1 = 1.45, colX2 = 1.45 + colW + 0.4, top = 2.25, headH = 0.7, rowH = 0.72;
  // headers
  s.addShape(pptx.ShapeType.roundRect, { x: colX1, y: top, w: colW, h: headH, rectRadius: 0.06, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("Model 3 · Full Fine-Tuning", { x: colX1, y: top, w: colW, h: headH, align: "center", valign: "middle", fontFace: HFONT, fontSize: 17, bold: true, color: WHITE });
  s.addShape(pptx.ShapeType.roundRect, { x: colX2, y: top, w: colW, h: headH, rectRadius: 0.06, fill: { color: TEAL }, line: { type: "none" } });
  s.addText("Model 4 · Adapter Tuning  (ours)", { x: colX2, y: top, w: colW, h: headH, align: "center", valign: "middle", fontFace: HFONT, fontSize: 17, bold: true, color: WHITE });

  // rows
  let y = top + headH + 0.12;
  for (let i = 0; i < rowsLabels.length; i++) {
    const bg = i % 2 ? "EEF3FA" : WHITE;
    // left
    s.addShape(pptx.ShapeType.rect, { x: colX1, y, w: colW, h: rowH, fill: { color: bg }, line: { color: "E2E8F0", width: 0.5 } });
    s.addText(rowsLabels[i], { x: colX1 + 0.2, y, w: 2.1, h: rowH, valign: "middle", fontFace: BFONT, fontSize: 11.5, bold: true, color: MUTE });
    s.addText(full[i], { x: colX1 + 2.25, y, w: colW - 2.4, h: rowH, valign: "middle", fontFace: BFONT, fontSize: 13.5, color: INK });
    // right
    s.addShape(pptx.ShapeType.rect, { x: colX2, y, w: colW, h: rowH, fill: { color: i % 2 ? "E6F7F4" : "F2FBF9" }, line: { color: "CDEDE6", width: 0.5 } });
    s.addText(rowsLabels[i], { x: colX2 + 0.2, y, w: 2.1, h: rowH, valign: "middle", fontFace: BFONT, fontSize: 11.5, bold: true, color: TEALD });
    s.addText(adp[i], { x: colX2 + 2.25, y, w: colW - 2.4, h: rowH, valign: "middle", fontFace: BFONT, fontSize: 13.5, bold: true, color: INK });
    y += rowH;
  }

  s.addText([
    { text: "In short:  ", options: { bold: true, color: TEALD } },
    { text: "the adapter reaches almost the same accuracy as full fine-tuning while training ~75× fewer weights — far cheaper to train, store and swap.", options: {} },
  ], { x: 1.45, y: y + 0.15, w: 11.3, h: 0.7, fontFace: BFONT, fontSize: 14.5, color: INK });
}

// =====================================================================
// SLIDE 12 — ERROR ANALYSIS
// =====================================================================
{
  const s = contentSlide("Error Analysis", "Right & wrong");
  s.addText("Real mBERT-adapter predictions on the multilingual validation set (threshold 0.05).", {
    x: 1.45, y: 1.65, w: 11.2, h: 0.4, fontFace: BFONT, fontSize: 14, italic: true, color: TEALD,
  });

  const quad = [
    [GREEN, "✓ True Positive", "it · p = 0.98", "“Incazzato come sei, non sei pure tu un sockpuppet…”", "Toxic correctly flagged — transfer works."],
    [RED, "✗ False Negative", "es · p = 0.00", "“Supongo que eso de que eres un adicto al porno es un vandalismo…”", "Missed toxic — subtle insult, low recall cost."],
    [RED, "✗ False Positive", "it · p = 0.99", "“Forse stupida; vedo che riesci a editare da mobile, come fai?”", "Clean over-flagged — 'stupida' triggers it."],
    [GREEN, "✓ True Negative", "es · p = 0.00", "“Muchas gracias, LlamaAl. Sí que parece útil…”", "Clean correctly passed."],
  ];
  const cw = 5.5, ch = 2.2, gx = 0.3, gy = 0.25;
  let i = 0;
  for (let r = 0; r < 2; r++) {
    for (let cI = 0; cI < 2; cI++) {
      const x = 1.45 + cI * (cw + gx);
      const y = 2.25 + r * (ch + gy);
      const [c, head, meta, txt, note] = quad[i++];
      s.addShape(pptx.ShapeType.roundRect, { x, y, w: cw, h: ch, rectRadius: 0.07, fill: { color: WHITE }, line: { color: c, width: 1.5 } });
      s.addShape(pptx.ShapeType.rect, { x, y, w: 0.12, h: ch, fill: { color: c } });
      s.addText(head, { x: x + 0.3, y: y + 0.12, w: cw - 2.0, h: 0.4, fontFace: HFONT, fontSize: 16, bold: true, color: c });
      s.addText(meta, { x: x + cw - 1.85, y: y + 0.14, w: 1.7, h: 0.35, align: "right", fontFace: BFONT, fontSize: 12, bold: true, color: MUTE });
      s.addText(txt, { x: x + 0.3, y: y + 0.6, w: cw - 0.55, h: 0.95, fontFace: BFONT, fontSize: 13, italic: true, color: INK });
      s.addText(note, { x: x + 0.3, y: y + ch - 0.55, w: cw - 0.55, h: 0.45, fontFace: BFONT, fontSize: 12, color: c });
      i; // noop
    }
  }
}

// =====================================================================
// SLIDE 12b — THE AUC PARADOX: HIGH AUC, LOW RECALL
// =====================================================================
{
  const s = contentSlide("High AUC, but Misses Real Insults?", "Reading the metric");
  s.addText("Turkish AUC is a strong 0.89 — yet the model lets many toxic comments through. Both are true. Here is why.", {
    x: 1.45, y: 1.6, w: 11.2, h: 0.45, fontFace: BFONT, fontSize: 15.5, italic: true, color: TEALD,
  });

  // ----- left: the thermometer analogy -----
  card(s, 1.45, 2.25, 5.5, 4.05, WHITE);
  s.addText("Think of a thermometer", { x: 1.7, y: 2.4, w: 5.0, h: 0.4, fontFace: HFONT, fontSize: 16, bold: true, color: NAVY });
  s.addText([
    { text: "AUC asks: does the model give toxic comments a ", options: {} },
    { text: "higher", options: { bold: true, color: TEAL } },
    { text: " score than clean ones? ", options: {} },
    { text: "Yes — its ranking is good (0.89).", options: { bold: true, color: TEAL } },
  ], { x: 1.7, y: 2.85, w: 5.0, h: 0.95, fontFace: BFONT, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.12 });
  s.addText([
    { text: "But the actual scores it gives Turkish toxicity are ", options: {} },
    { text: "tiny", options: { bold: true, color: RED } },
    { text: " — often 0.2%–0.5%. They are ranked correctly, just far below any sensible cut-off.", options: {} },
  ], { x: 1.7, y: 3.85, w: 5.0, h: 1.0, fontFace: BFONT, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.12 });
  s.addText([
    { text: "Recall asks a different question: ", options: {} },
    { text: "did we actually flag them?", options: { bold: true, color: RED } },
    { text: "  Since the scores sit so low, few cross the line — so recall stays low even when AUC is high.", options: {} },
  ], { x: 1.7, y: 4.95, w: 5.0, h: 1.2, fontFace: BFONT, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.12 });

  // ----- right: censored missed-toxic (false negatives) -----
  s.addText("Missed Turkish toxicity (false negatives)", { x: 7.2, y: 2.25, w: 5.6, h: 0.35, fontFace: HFONT, fontSize: 15, bold: true, color: RED });
  const missed = [
    ["S*** git, b** p**.", "0.2%"],
    ["O****** ç*****, defol.", "0.4%"],
    ["A**l herif, k*** kafalı.", "0.3%"],
  ];
  let my = 2.7;
  missed.forEach(([txt, p]) => {
    s.addShape(pptx.ShapeType.roundRect, { x: 7.2, y: my, w: 5.55, h: 0.78, rectRadius: 0.06, fill: { color: WHITE }, line: { color: RED, width: 1 } });
    s.addShape(pptx.ShapeType.rect, { x: 7.2, y: my, w: 0.1, h: 0.78, fill: { color: RED } });
    s.addText(txt, { x: 7.45, y: my, w: 3.7, h: 0.78, valign: "middle", fontFace: BFONT, fontSize: 14, italic: true, color: INK });
    s.addText(p, { x: 11.2, y: my, w: 1.45, h: 0.78, align: "right", valign: "middle", fontFace: HFONT, fontSize: 18, bold: true, color: RED });
    my += 0.9;
  });
  s.addText("Toxic probability the model assigned →  all far below the 5% threshold, so all were let through.", {
    x: 7.2, y: 5.5, w: 5.55, h: 0.6, fontFace: BFONT, fontSize: 11.5, italic: true, color: MUTE,
  });
  s.addText("Profanity heavily censored. The model never saw these explicit Turkish words in English-only training.", {
    x: 7.2, y: 6.1, w: 5.55, h: 0.6, fontFace: BFONT, fontSize: 11, italic: true, color: MUTE,
  });
}

// =====================================================================
// SLIDE 12c — RELATED WORK / COMPARISON
// =====================================================================
{
  const s = contentSlide("How We Compare to Other Work", "Related work");
  s.addText("We are not the first to tackle multilingual toxicity. Here is how our single-GPU result sits next to the literature.", {
    x: 1.45, y: 1.62, w: 11.2, h: 0.45, fontFace: BFONT, fontSize: 15.5, italic: true, color: TEALD,
  });

  const head = ["Study / system", "Method", "Multilingual AUC", "Setup"];
  const rowsData = [
    ["Jigsaw Kaggle winners (2020)", "XLM-R (large) + ensembles", "~0.95", "full test set · heavy GPUs"],
    ["Published mBERT cross-lingual", "full fine-tuning", "~0.80–0.85", "similar idea to ours"],
    ["Houlsby et al. (2019), adapters", "adapters on BERT", "≈ full  (−0.4%)", "shows adapters ≈ full tuning"],
    ["Ours: mBERT + adapter", "adapters, English-only train", "0.83", "single GPU · 1.34% params"],
  ];
  const table = [head.map((h) => ({ text: h, options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "left", fontSize: 13 } }))];
  rowsData.forEach((d, ri) => {
    const isStar = ri === rowsData.length - 1;
    const bg = isStar ? "E6F7F4" : (ri % 2 ? "EEF3FA" : WHITE);
    table.push(d.map((c) => ({ text: c, options: { align: "left", bold: isStar, color: isStar ? TEALD : INK, fill: { color: bg }, fontSize: 12.5 } })));
  });
  s.addTable(table, { x: 1.45, y: 2.2, w: 11.3, colW: [3.5, 3.0, 2.2, 2.6], rowH: 0.5, border: { type: "solid", color: "D5DEEA", pt: 1 }, fontFace: BFONT, valign: "middle" });

  s.addText(bullets([
    "The best public scores (~0.95) use far bigger models (XLM-R), many languages and ensembles — well beyond a single-GPU budget.",
    "Our 0.83 lands in the same range as published mBERT cross-lingual results, even though we trained on English only.",
    "Adapter studies (Houlsby 2019; Pfeiffer MAD-X 2020) report adapters matching full fine-tuning — exactly the pattern we see (0.83 vs 0.85).",
  ], { fontSize: 13.5 }), { x: 1.45, y: 5.0, w: 11.3, h: 1.6 });

  s.addText("Different test sets, model sizes and compute — shown for context, not a head-to-head ranking.", {
    x: 1.45, y: 6.6, w: 11.3, h: 0.35, fontFace: BFONT, fontSize: 11, italic: true, color: MUTE,
  });
}

// =====================================================================
// SLIDE 13 — CONCLUSIONS
// =====================================================================
{
  const s = contentSlide("Concluding Remarks", "What we learned");
  const points = [
    ["Cross-lingual transfer is real", "A multilingual encoder generalises toxicity from English to unseen languages; word-count and LSTM models cannot."],
    ["Adapters are the sweet spot", "~1.34% of parameters reach full-fine-tuning-level transfer (~0.83 AUC) — the project's main result."],
    ["Honest evaluation matters", "Per-language metrics, threshold tuning and best-checkpoint selection prevent over-optimistic numbers."],
    ["Limitations & next steps", "Recall is modest under imbalance; the official test set is unlabelled. Future: adapter-size ablation, more languages, focal loss."],
  ];
  let y = 1.85;
  points.forEach(([h, b], i) => {
    s.addShape(pptx.ShapeType.ellipse, { x: 1.5, y: y + 0.05, w: 0.55, h: 0.55, fill: { color: i === 1 ? TEAL : NAVY } });
    s.addText(String(i + 1), { x: 1.5, y: y + 0.05, w: 0.55, h: 0.55, align: "center", valign: "middle", fontFace: HFONT, fontSize: 18, bold: true, color: WHITE });
    s.addText(h, { x: 2.3, y: y, w: 10.3, h: 0.45, fontFace: HFONT, fontSize: 19, bold: true, color: i === 1 ? TEALD : NAVY });
    s.addText(b, { x: 2.3, y: y + 0.42, w: 10.3, h: 0.7, fontFace: BFONT, fontSize: 14.5, color: INK });
    y += 1.25;
  });
}

// =====================================================================
// SLIDE 14 — THANK YOU / Q&A
// =====================================================================
{
  const s = pptx.addSlide();
  s.background = { color: NAVY };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.18, fill: { color: TEAL } });
  s.addShape(pptx.ShapeType.ellipse, { x: -1.4, y: 5.2, w: 4.2, h: 4.2, fill: { color: NAVY2 }, line: { color: TEAL, width: 1.5 } });

  s.addText("Thank you", { x: 0.9, y: 2.1, w: 11.5, h: 1.1, fontFace: HFONT, fontSize: 54, bold: true, color: WHITE });
  s.addText("Questions & discussion", { x: 0.92, y: 3.25, w: 11, h: 0.6, fontFace: BFONT, fontSize: 22, italic: true, color: ICE });
  s.addShape(pptx.ShapeType.rect, { x: 0.95, y: 4.0, w: 2.2, h: 0.05, fill: { color: TEAL } });

  s.addText([
    { text: "Multilingual Toxic Comment Classification", options: { bold: true, color: WHITE } },
    { text: "\nSEDS 537 · Machine Learning · Term Project · Spring 2026", options: { color: ICE } },
    { text: "\nTF-IDF · BiLSTM · mBERT full fine-tuning · mBERT + adapters", options: { color: "8FA0C8" } },
  ], { x: 0.95, y: 4.3, w: 11, h: 1.3, fontFace: BFONT, fontSize: 16, lineSpacingMultiple: 1.3 });

  s.addText("Presentation: 1 / 8 June 2026, 16:30  ·  12-minute talk + 2–3 min Q&A", {
    x: 0.95, y: 6.6, w: 11.5, h: 0.4, fontFace: BFONT, fontSize: 12, color: "8FA0C8",
  });
}

// ---- save ------------------------------------------------------------------
const out = path.join(__dirname, "Multilingual_Toxic_Classification.pptx");
pptx.writeFile({ fileName: out }).then(() => console.log("Saved:", out));
