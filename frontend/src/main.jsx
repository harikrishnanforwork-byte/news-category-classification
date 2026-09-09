import { StrictMode, useEffect, useState } from "react";
import { Activity, ArrowUpRight, BarChart3, Check, ChevronRight, FileText, Gauge, Globe2, Layers3, Newspaper, Sparkles, UploadCloud } from "lucide-react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = "http://127.0.0.1:8000";
const categories = [
  { name: "World", color: "#4f83ff", icon: Globe2 },
  { name: "Sports", color: "#13a87b", icon: Activity },
  { name: "Business", color: "#e28a35", icon: BarChart3 },
  { name: "Sci/Tech", color: "#b95c9a", icon: Sparkles },
];
const examples = [
  "Scientists develop a battery that charges electric cars in five minutes",
  "The national team wins the football championship final",
  "Company profits increase as investors buy more shares",
];

function App() {
  const [headline, setHeadline] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState({ status: "checking", articles: 32, features: 958 });

  useEffect(() => {
    fetch(`${API_URL}/api/health`).then((response) => response.json()).then(setHealth).catch(() => setHealth((current) => ({ ...current, status: "offline" })));
  }, []);

  async function classify(event) {
    event.preventDefault();
    if (!headline.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: headline }),
      });
      setResult(await response.json());
    } catch {
      setResult({ error: "The prediction service is offline. Start the FastAPI server and try again." });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="logo"><span>N</span><strong>NewsLens</strong></div>
        <p className="sidebar-copy">Editorial intelligence for fast, consistent news tagging.</p>
        <div className="sidebar-section">
          <div className="sidebar-label">Workspace</div>
          <div className="nav-item active"><Newspaper size={17} /> Classifier <ChevronRight size={15} /></div>
          <div className="nav-item"><Layers3 size={17} /> Data pipeline</div>
        </div>
        <div className="sidebar-section category-list">
          <div className="sidebar-label">Label set</div>
          {categories.map(({ name, color, icon: Icon }) => <div className="category-row" key={name}><Icon size={16} color={color} /><span>{name}</span><i style={{ background: color }} /></div>)}
        </div>
        <div className="sidebar-bottom"><div className="mini-status"><span /> {health.status === "online" ? "Model online" : "API offline"}</div><small>Neural classification workspace<br />v1.0.0</small></div>
      </aside>

      <main className="content">
        <header className="topbar"><div><span className="crumb">WORKSPACE</span><span className="slash">/</span><strong>CLASSIFIER</strong></div><div className="top-actions"><span className="sync"><span /> {health.status === "online" ? "Synced" : "Local mode"}</span><button className="avatar">NL</button></div></header>
        <section className="hero"><div><div className="eyebrow">Four-category neural classifier</div><h1>Make every headline<br /><em>find its lane.</em></h1><p>Turn raw news into a clear editorial signal. Paste a headline or article excerpt and let the model rank the story.</p></div><div className="hero-orbit"><div className="orbit-ring ring-one" /><div className="orbit-ring ring-two" /><div className="orbit-core"><Gauge size={28} /><small>AI<br />READY</small></div></div></section>

        <section className="dashboard-grid">
          <form className="classifier-card" onSubmit={classify}>
            <div className="card-heading"><div><span className="card-kicker">LIVE CLASSIFICATION</span><h2>What is this story about?</h2></div><span className="live-dot">● LIVE</span></div>
            <label htmlFor="headline">Headline or article excerpt</label>
            <textarea id="headline" value={headline} onChange={(event) => setHeadline(event.target.value)} placeholder="Paste a headline, opening paragraph, or news brief..." />
            <div className="form-footer"><span>{headline.length} characters</span><button className="primary-button" disabled={loading}>{loading ? "Reading story..." : "Classify story"}<ArrowUpRight size={17} /></button></div>
            <div className="examples"><span>Try one</span>{examples.map((example) => <button type="button" key={example} onClick={() => setHeadline(example)}>{example}</button>)}</div>
          </form>

          <div className="result-card">
            <div className="card-heading"><div><span className="card-kicker">PREDICTION STAGE</span><h2>Model readout</h2></div><FileText size={19} /></div>
            {result?.category ? <><div className="prediction-label">PREDICTED CATEGORY</div><div className="prediction-category">{result.category}</div><div className="confidence-row"><span>Model confidence</span><strong>{Math.round(result.confidence * 100)}%</strong></div><div className="confidence-track"><span style={{ width: `${result.confidence * 100}%` }} /></div><div className="signals-title">Other signals</div>{result.signals.map((signal) => <div className="signal" key={signal.category}><span>{signal.category}</span><b>{Math.round(signal.confidence * 100)}%</b><div><i style={{ width: `${signal.confidence * 100}%` }} /></div></div>)}</> : <div className="empty-result"><div className="empty-icon"><Sparkles size={23} /></div><h3>{result?.error || "Awaiting a headline"}</h3><p>{result?.error ? "Check that the API is running on port 8000." : "Your classification will appear here after the model reads your story."}</p></div>}
          </div>
        </section>

        <section className="insights"><div className="insight-title"><span className="card-kicker">SYSTEM OVERVIEW</span><h2>Model health</h2></div><div className="stat"><span>Training articles</span><strong>{health.articles.toLocaleString()}</strong><small><Check size={13} /> Dataset loaded</small></div><div className="stat"><span>TF-IDF features</span><strong>{health.features.toLocaleString()}</strong><small><Check size={13} /> Vectorizer ready</small></div><div className="stat"><span>Categories</span><strong>04</strong><small><Check size={13} /> Label set active</small></div></section>
        <footer><span>NewsLens intelligence platform</span><span>Automatic preprocessing enabled · <a href="http://localhost:8502">Open Streamlit app</a></span></footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
