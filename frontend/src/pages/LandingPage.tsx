import { useEffect, useState, CSSProperties } from "react";

const INDEX_SNAPSHOT = [
  { term: "search", docs: ["doc_001", "doc_004", "doc_005"] },
  { term: "retriev", docs: ["doc_004", "doc_001"] },
  { term: "algorithm", docs: ["doc_003", "doc_007", "doc_009"] },
  { term: "index", docs: ["doc_001", "doc_002", "doc_004"] },
  { term: "rank", docs: ["doc_005", "doc_010"] },
  { term: "term", docs: ["doc_002", "doc_004"] },
  { term: "fuzzi", docs: ["doc_013"] },
  { term: "trie", docs: ["doc_015"] },
];

function IndexViz() {
  const [visible, setVisible] = useState(0);
  const [flash, setFlash] = useState(-1);

  useEffect(() => {
    if (visible >= INDEX_SNAPSHOT.length) return;
    const t = setTimeout(() => {
      setFlash(visible);
      setTimeout(() => setFlash(-1), 400);
      setVisible((v) => v + 1);
    }, 320);
    return () => clearTimeout(t);
  }, [visible]);

  return (
    <div
      style={{
        fontFamily: "var(--mono)",
        fontSize: "0.8rem",
        lineHeight: 1.9,
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        padding: "24px 28px",
        minHeight: "280px",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          color: "var(--muted)",
          fontSize: "0.7rem",
          marginBottom: "14px",
          letterSpacing: "0.1em",
          textTransform: "uppercase",
        }}
      >
        inverted_index
      </div>

      {INDEX_SNAPSHOT.slice(0, visible).map((row, i) => (
        <div
          key={row.term}
          style={{
            display: "flex",
            alignItems: "baseline",
            opacity: i < visible ? 1 : 0,
            transition: "opacity 0.3s",
            background: flash === i ? "rgba(99,102,241,0.08)" : "transparent",
            borderRadius: "3px",
            padding: "0 4px",
            margin: "0 -4px",
          }}
        >
          <span style={{ color: "#a5b4fc", minWidth: "96px" }}>{row.term}</span>
          <span style={{ color: "var(--muted)", margin: "0 8px" }}>→</span>
          <span style={{ color: "var(--text-2)" }}>
            {"["}
            {row.docs.map((d, j) => (
              <span key={d}>
                <span style={{ color: "var(--green)" }}>{d}</span>
                {j < row.docs.length - 1 && (
                  <span style={{ color: "var(--muted)" }}>, </span>
                )}
              </span>
            ))}
            {"]"}
          </span>
        </div>
      ))}

      {visible < INDEX_SNAPSHOT.length && (
        <span
          style={{
            display: "inline-block",
            width: "8px",
            height: "14px",
            background: "var(--accent)",
            verticalAlign: "middle",
            animation: "blink 1s step-end infinite",
          }}
        />
      )}
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
    </div>
  );
}

const STATS = [
  { value: "1,137", label: "index terms" },
  { value: "33", label: "documents" },
  { value: "<1ms", label: "query latency" },
  { value: "453", label: "tests passing" },
];

const FEATURES = [
  {
    tag: "TF-IDF",
    desc: "Relevance scoring from first principles — no libraries, pure math.",
  },
  {
    tag: "Fuzzy search",
    desc: "Levenshtein edit distance catches typos within a configurable threshold.",
  },
  {
    tag: "Phrase match",
    desc: "Positional posting lists enforce exact token adjacency.",
  },
  {
    tag: "Autocomplete",
    desc: "Trie prefix tree built over all 1,137 index terms.",
  },
  {
    tag: "Filters",
    desc: "Category, date range, word count — applied pre-rank for efficiency.",
  },
  {
    tag: "Live re-index",
    desc: "Add, update, remove documents without full rebuilds.",
  },
];

interface Props {
  onEnter: () => void;
}

export function LandingPage({ onEnter }: Props) {
  return (
    <div
      style={{ maxWidth: "860px", margin: "0 auto", padding: "64px 24px 80px" }}
    >
      {/* Hero */}
      <div style={{ marginBottom: "56px" }}>
        <h1
          style={{
            fontFamily: "var(--mono)",
            fontSize: "clamp(2rem, 5vw, 3.2rem)",
            fontWeight: 500,
            color: "var(--text)",
            lineHeight: 1.15,
            letterSpacing: "-0.03em",
            marginBottom: "20px",
          }}
        >
          QueryCore
        </h1>

        <p
          style={{
            color: "var(--text-2)",
            fontSize: "1.05rem",
            lineHeight: 1.7,
            maxWidth: "520px",
            marginBottom: "32px",
          }}
        >
          A text search engine built from scratch in Python — inverted index,
          TF-IDF ranking, fuzzy matching, autocomplete, phrase search, and a
          REST API. No Elasticsearch. No shortcuts.
        </p>

        <button
          onClick={onEnter}
          style={{
            background: "var(--accent)",
            border: "none",
            borderRadius: "var(--radius)",
            color: "#fff",
            cursor: "pointer",
            fontFamily: "var(--sans)",
            fontSize: "0.95rem",
            fontWeight: 600,
            padding: "14px 28px",
            transition: "background 0.15s, transform 0.1s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#4f46e5")}
          onMouseLeave={(e) =>
            (e.currentTarget.style.background = "var(--accent)")
          }
          onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.98)")}
          onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
        >
          Open search engine →
        </button>
      </div>

      {/* Two-column */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "24px",
          marginBottom: "32px",
          alignItems: "start",
        }}
      >
        <IndexViz />

        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {FEATURES.map(({ tag, desc }) => (
            <div
              key={tag}
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius)",
                padding: "14px 18px",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
              }}
            >
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: "0.78rem",
                  color: "var(--accent)",
                  fontWeight: 500,
                }}
              >
                {tag}
              </span>
              <span
                style={{
                  color: "var(--text-2)",
                  fontSize: "0.82rem",
                  lineHeight: 1.5,
                }}
              >
                {desc}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "1px",
          background: "var(--border)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
        }}
      >
        {STATS.map(({ value, label }) => (
          <div
            key={label}
            style={{
              background: "var(--surface)",
              padding: "20px 24px",
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: "1.4rem",
                fontWeight: 500,
                color: "var(--accent)",
                marginBottom: "4px",
              }}
            >
              {value}
            </div>
            <div style={{ color: "var(--text-2)", fontSize: "0.75rem" }}>
              {label}
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div
        style={{
          marginTop: "48px",
          paddingTop: "24px",
          borderTop: "1px solid var(--border)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          color: "var(--muted)",
          fontSize: "0.78rem",
          fontFamily: "var(--mono)",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <span>Python · FastAPI · React · TypeScript · Vite</span>
        <span>inverted index · tf-idf · levenshtein · trie</span>
      </div>
    </div>
  );
}
