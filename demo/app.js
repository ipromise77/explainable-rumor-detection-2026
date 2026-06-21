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
const explanation = document.querySelector("#explanation");

function setLoading() {
  verdict.textContent = "检测中";
  verdict.className = "verdict";
  prob.textContent = "-";
  source.textContent = "-";
  detectorName.textContent = "-";
  explanation.textContent = "模型正在生成判断依据...";
}

function setResult(data) {
  const rumor = data.label === 1;
  verdict.textContent = `${data.label} / ${data.label_name}`;
  verdict.className = `verdict ${rumor ? "rumor" : "non-rumor"}`;
  prob.textContent = Number(data.prob_rumor).toFixed(4);
  source.textContent = data.source || "local";
  detectorName.textContent = data.detector === "final" ? "FinalRumourDetectClass" : "RumourDetectClass";
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

runBtn.addEventListener("click", runPrediction);
sampleBtn.addEventListener("click", () => {
  const current = samples.indexOf(tweet.value.trim());
  tweet.value = samples[(current + 1) % samples.length];
});

runPrediction();
