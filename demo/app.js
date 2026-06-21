const samples = [
  "anonymous sources obtained documents showing a massive cover-up by officials",
  "BREAKING: police confirm the earlier viral claim is false after investigation #news",
  "This smear campaign has devolved into the worst crisis actor hoax online",
];

const tweet = document.querySelector("#tweet");
const detector = document.querySelector("#detector");
const runBtn = document.querySelector("#runBtn");
const sampleBtn = document.querySelector("#sampleBtn");
const verdict = document.querySelector("#verdict");
const prob = document.querySelector("#prob");
const source = document.querySelector("#source");
const detectorName = document.querySelector("#detectorName");
const localLabel = document.querySelector("#localLabel");
const analysisMode = document.querySelector("#analysisMode");
const explanation = document.querySelector("#explanation");
const meterFill = document.querySelector("#meterFill");

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
sampleBtn.addEventListener("click", () => {
  const current = samples.indexOf(tweet.value.trim());
  tweet.value = samples[(current + 1) % samples.length];
});

runPrediction();
