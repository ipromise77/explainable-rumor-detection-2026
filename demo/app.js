const samples = [
  "anonymous sources obtained documents showing a massive cover-up by officials",
  "family and friends gathered today for a peaceful community event",
  "This smear campaign has devolved into the worst crisis actor hoax online",
];

const tweet = document.querySelector("#tweet");
const detector = document.querySelector("#detector");
const runBtn = document.querySelector("#runBtn");
const copyBtn = document.querySelector("#copyBtn");
const sampleTabs = document.querySelectorAll(".sample-tab");
const verdict = document.querySelector("#verdict");
const prob = document.querySelector("#prob");
const source = document.querySelector("#source");
const detectorName = document.querySelector("#detectorName");
const localLabel = document.querySelector("#localLabel");
const analysisMode = document.querySelector("#analysisMode");
const explanation = document.querySelector("#explanation");
const meterFill = document.querySelector("#meterFill");

const params = new URLSearchParams(window.location.search);
const requestedDetector = params.get("detector");
const requestedSample = Number(params.get("sample"));
const requestedText = params.get("text");

if (requestedDetector === "main" || requestedDetector === "final") {
  detector.value = requestedDetector;
}

if (requestedText) {
  tweet.value = requestedText;
} else if (Number.isInteger(requestedSample) && samples[requestedSample]) {
  tweet.value = samples[requestedSample];
}

if (Number.isInteger(requestedSample) && samples[requestedSample]) {
  sampleTabs.forEach((tab) => {
    tab.classList.toggle("active", Number(tab.dataset.sample) === requestedSample);
  });
}

function setLoading() {
  verdict.textContent = "检测中";
  verdict.className = "verdict";
  prob.textContent = "-";
  source.textContent = "-";
  detectorName.textContent = "-";
  localLabel.textContent = "-";
  analysisMode.textContent = "Local";
  meterFill.style.width = "0%";
  explanation.textContent = "模型正在生成判断依据...";
  copyBtn.disabled = true;
}

function setResult(data) {
  const rumor = data.label === 1;
  verdict.textContent = `${data.label} / ${data.label_name}`;
  verdict.className = `verdict ${rumor ? "rumor" : "non-rumor"}`;
  const probability = Number(data.prob_rumor);
  prob.textContent = probability.toFixed(4);
  meterFill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
  source.textContent = data.source || "local";
  detectorName.textContent = data.detector === "final" ? "FinalRumourDetectClass" : "RumourDetectClass";
  localLabel.textContent = data.local_label === null || data.local_label === undefined ? String(data.label) : String(data.local_label);
  analysisMode.textContent = data.source && data.source.startsWith("rule:") ? "Rule Signal" : "Local Evidence";
  explanation.textContent = data.explanation || "未返回解释。";
  copyBtn.disabled = false;
}

async function runPrediction() {
  const text = tweet.value.trim();
  if (!text) {
    explanation.textContent = "请输入待检测文本。";
    return;
  }
  setLoading();
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, detector: detector.value }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "请求失败");
    }
    setResult(data);
  } catch (err) {
    verdict.textContent = "错误";
    verdict.className = "verdict rumor";
    explanation.textContent = err.message;
  }
}

document.querySelector("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});
sampleTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    sampleTabs.forEach((item) => item.classList.remove("active"));
    tab.classList.add("active");
    const index = Number(tab.dataset.sample);
    tweet.value = samples[index] || samples[0];
    detector.value = tab.dataset.detector || "main";
    runPrediction();
  });
});
copyBtn.addEventListener("click", async () => {
  const text = explanation.textContent.trim();
  if (!text || text === "模型正在生成判断依据...") {
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = "已复制";
  } catch {
    copyBtn.textContent = "复制失败";
  }
  window.setTimeout(() => {
    copyBtn.textContent = "复制解释";
  }, 1200);
});

runPrediction();
