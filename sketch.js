let autoTokenizer, autoModel, matmul;
let loaded = false;

let lines = [];
let points = [];

let inputField, button;
let resultText = "Loading model...";
let hoverText = "";

// to store original high‑dim corpus embeddings
let corpusEmbeddings = [];
let topK = 10;          // 显示多少条最相似文本
let perPersonTopN = 5;  // 每个人用TopN条来算平均分（更稳定）
let allLoaded = 0;

function preload() {
  lines = [];

  loadStrings("Molly.txt", (arr) => addPersonLines("Molly", arr));
  loadStrings("Fu Xinchun.txt",  (arr) => addPersonLines("Fu Xinchun", arr));
  loadStrings("Quinn Zhou.txt",  (arr) => addPersonLines("Quinn Zhou", arr));
  loadStrings("610.txt",  (arr) => addPersonLines("610", arr));
}

function addPersonLines(name, arr) {
  // 去掉空行，并给每行加标签
  let cleaned = arr
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(s => `${name} | ${s}`);

  lines = lines.concat(cleaned);

  allLoaded++;
  if (allLoaded === 4) {
    console.log("✅ all people loaded, total lines:", lines.length);
  }
}

function setup() {
  createCanvas(600, 600);
  textSize(12);

  loadTransformers();

  inputField = createElement('textarea');
  inputField.style("width", width - 6 + "px");

  button = createButton("Search");
  button.style("display","block");
  button.attribute("disabled", true); // disable initially
  button.mousePressed(runSearch);
  
}

function draw() {
  background(20);

  if (!loaded) {
    fill(255);
    textAlign(LEFT);
    text(resultText, 20, 20);
  }else{
    fill(255);
    textAlign(LEFT);
    text(resultText, 20, 20);
    drawPoints();
    fill(255);
    textAlign(CENTER);
    text(hoverText, mouseX, mouseY - 10);
  }

}



//make the mapping of points
async function computeCorpusUMAP() {

  // add prefix to every line
  let prefixes = { line: "title: none | text: " };
  let formatted = [];
  for (let i = 0; i < lines.length; i++) {
    formatted.push(prefixes.line + lines[i]);
  }

  // tokenize + embed
  let inputs = await autoTokenizer(formatted, { padding: true, truncation: true });
  let output = await autoModel(inputs);
  corpusEmbeddings = output.sentence_embedding.tolist(); // array of [D]


  // run UMAP to take embeddings and map to 2 dimensions
  let umap = new UMAP.UMAP({ nComponents: 2, nNeighbors: 10, minDist: 0.1 });
  let embedding2D = umap.fit(corpusEmbeddings);

  points = [];
  for (let i = 0; i < lines.length; i++) {
    points.push({
      x: embedding2D[i][0],
      y: embedding2D[i][1],
      similarity: 0,
      text: lines[i]
    });
  }

  normalizePoints();
}

function getPersonName(lineText) {
  // lineText format: "Name | content..."
  let idx = lineText.indexOf("|");
  if (idx === -1) return "Unknown";
  return lineText.slice(0, idx).trim();
}

async function runSearch() {
  if (!autoModel) {
    console.log("Model Still loading");
    return;
  }

  let query = inputField.value().trim();
  if (!query) return;

  resultText = "Thinking…";

  let prefixes = {
    query: "task: search result | query: ",
    line: "title: none | text: "
  };

  let allTexts = [prefixes.query + query];
  for (let i = 0; i < lines.length; i++) {
    allTexts.push(prefixes.line + lines[i]);
  }

  let inputs = await autoTokenizer(allTexts, { padding: true, truncation: true });
  let output = await autoModel(inputs);

  let allEmbeddings = output.sentence_embedding;

  // similarity via matmul
  let scores = await matmul(allEmbeddings, allEmbeddings.transpose(1, 0));
  let similarities = scores.tolist()[0].slice(1);

  // update similarities for coloring
  for (let i = 0; i < points.length; i++) {
    points[i].similarity = similarities[i];
  }

  // ---- helpers ----
  function getPersonName(lineText) {
    let idx = lineText.indexOf("|");
    if (idx === -1) return "Unknown";
    return lineText.slice(0, idx).trim();
  }

  function getContentOnly(lineText) {
    // "Name | content" -> "content"
    let idx = lineText.indexOf("|");
    if (idx === -1) return lineText.trim();
    return lineText.slice(idx + 1).trim();
  }

  // ---- per-person max/min ----
  let stats = {};

  for (let i = 0; i < lines.length; i++) {
    let fullText = lines[i];          // e.g. "Molly | Today I ate ..."
    let name = getPersonName(fullText);
    let score = similarities[i];

    if (!stats[name]) {
      stats[name] = {
        maxScore: -Infinity,
        maxLine: "",
        minScore: Infinity,
        minLine: ""
      };
    }

    if (score > stats[name].maxScore) {
      stats[name].maxScore = score;
      stats[name].maxLine = fullText;
    }

    if (score < stats[name].minScore) {
      stats[name].minScore = score;
      stats[name].minLine = fullText;
    }
  }

  // ---- sort by max desc ----
  let arr = Object.entries(stats).map(([name, s]) => ({
    name,
    maxScore: s.maxScore,
    maxLine: s.maxLine,
    minScore: s.minScore,
    minLine: s.minLine
  }));

  arr.sort((a, b) => b.maxScore - a.maxScore);

  // ---- output (content only) ----
  let out = "People ranking (by MAX similarity):\n\n";

  for (let p of arr) {
    out += `${p.name}\n`;
    out += `  max: ${p.maxScore.toFixed(3)}  |  ${getContentOnly(p.maxLine)}\n`;
    out += `  min: ${p.minScore.toFixed(3)}  |  ${getContentOnly(p.minLine)}\n\n`;
  }

  resultText = out;
}


//fit points to canvas
function normalizePoints() {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity; 
  let maxY = -Infinity;
  
  for (let p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  for (let p of points) {
    p.x = map(p.x, minX, maxX, 50, width - 50);
    p.y = map(p.y, minY, maxY, 50, height - 50);
  }
}

function drawPoints() {

  noStroke();

  hoverText="";
  for (let p of points) {
    let col = color(
      map(p.similarity, 0, 1, 50, 255), 
      100,
      map(p.similarity, 0, 1, 255, 50)
    );
    fill(col);
    circle(p.x, p.y, 10);

    if (dist(mouseX, mouseY, p.x, p.y) < 6) {
      hoverText = p.text;
    }
  }
}


async function loadTransformers() {

  const transformers = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers"
  );

  autoTokenizer = await transformers.AutoTokenizer.from_pretrained(
    "onnx-community/embeddinggemma-300m-ONNX"
  );

  autoModel = await transformers.AutoModel.from_pretrained(
    "onnx-community/embeddinggemma-300m-ONNX",
    {
      device: "webgpu",
      dtype: "fp32"
    }
  );

  matmul = transformers.matmul;

  resultText = "Model loaded. Computing corpus UMAP…";
  await computeCorpusUMAP();

  loaded = true;
  resultText = "Enter a query and click Search!";
  button.removeAttribute("disabled"); // enable when ready
}