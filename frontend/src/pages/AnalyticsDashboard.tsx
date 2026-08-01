import { useAnalytics } from "../hooks/useAnalytics";
import type { TopQuery, VolumeEntry } from "../types";

type BarRow = { [key: string]: string | number };

function BarChart({
  data,
  valueKey,
  labelKey,
}: {
  data: BarRow[];
  valueKey: string;
  labelKey: string;
}) {
  if (!data.length) return <Empty />;
  const max = Math.max(...data.map((d) => d[valueKey] as number));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      {data.map((row, i) => {
        const val = row[valueKey] as number;
        const label = row[labelKey] as string;
        const pct = max > 0 ? (val / max) * 100 : 0;

        return (
          <div
            key={i}
            style={{ display: "flex", alignItems: "center", gap: "12px" }}
          >
            <span
              style={{
                fontFamily: "var(--mono)",
                fontSize: "0.78rem",
                color: "var(--text-2)",
                minWidth: "140px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {label}
            </span>
            <div
              style={{
                flex: 1,
                background: "var(--border)",
                borderRadius: "3px",
                height: "8px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${pct}%`,
                  height: "100%",
                  background: "var(--accent)",
                  borderRadius: "3px",
                  transition: "width 0.6s ease",
                }}
              />
            </div>
            <span
              style={{
                fontFamily: "var(--mono)",
                fontSize: "0.75rem",
                color: "var(--accent)",
                minWidth: "28px",
                textAlign: "right",
              }}
            >
              {val}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Sparkline({ data }: { data: VolumeEntry[] }) {
  if (!data.length) return <Empty />;
  const max = Math.max(...data.map((d) => d.count), 1);
  const W = 400;
  const H = 60;
  const pad = 4;
  const step = data.length > 1 ? (W - pad * 2) / (data.length - 1) : W;

  const points = data
    .map((d, i) => {
      const x = pad + i * step;
      const y = H - pad - (d.count / max) * (H - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "60px" }}>
        <polyline
          points={points}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {data.map((d, i) => (
          <circle
            key={i}
            cx={pad + i * step}
            cy={H - pad - (d.count / max) * (H - pad * 2)}
            r="3"
            fill="var(--accent)"
          >
            <title>
              {d.date}: {d.count} searches
            </title>
          </circle>
        ))}
      </svg>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "var(--mono)",
          fontSize: "0.65rem",
          color: "var(--muted)",
          marginTop: "4px",
        }}
      >
        <span>{data[0]?.date}</span>
        <span>{data[data.length - 1]?.date}</span>
      </div>
    </div>
  );
}

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        padding: "24px",
      }}
    >
      <h2
        style={{
          color: "var(--text-2)",
          fontSize: "0.75rem",
          fontWeight: 600,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          marginBottom: "20px",
          fontFamily: "var(--sans)",
        }}
      >
        {title}
      </h2>
      {children}
    </div>
  );
}

function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: "1.8rem",
          fontWeight: 500,
          color: "var(--accent)",
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      <div
        style={{
          color: "var(--text-2)",
          fontSize: "0.75rem",
          marginTop: "6px",
        }}
      >
        {label}
      </div>
    </div>
  );
}

function Empty() {
  return (
    <p
      style={{
        color: "var(--muted)",
        fontSize: "0.8rem",
        fontFamily: "var(--mono)",
      }}
    >
      No data yet — run some searches first.
    </p>
  );
}

export function AnalyticsDashboard() {
  const { data, loading, error, reload } = useAnalytics();

  if (loading)
    return (
      <div
        style={{
          textAlign: "center",
          padding: "60px 0",
          color: "var(--text-2)",
          fontSize: "0.875rem",
        }}
      >
        Loading analytics…
      </div>
    );

  if (error)
    return (
      <div
        style={{
          background: "rgba(248,113,113,0.1)",
          border: "1px solid rgba(248,113,113,0.3)",
          borderRadius: "var(--radius)",
          color: "var(--red)",
          fontSize: "0.875rem",
          padding: "12px 16px",
          margin: "40px 20px",
        }}
      >
        {error} — is the backend running?
      </div>
    );

  if (!data) return null;

  const { summary, top, zero, volume } = data;
  const zeroRate = (summary.zero_result_rate * 100).toFixed(1);

  return (
    <div style={{ maxWidth: "860px", margin: "0 auto", padding: "40px 20px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "32px",
        }}
      >
        <div>
          <h1
            style={{
              fontFamily: "var(--mono)",
              fontSize: "1.4rem",
              fontWeight: 500,
              color: "var(--text)",
              marginBottom: "4px",
            }}
          >
            Analytics
          </h1>
          <p style={{ color: "var(--text-2)", fontSize: "0.8rem" }}>
            Search activity since last restart
          </p>
        </div>
        <button
          onClick={reload}
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            color: "var(--text-2)",
            cursor: "pointer",
            fontFamily: "var(--mono)",
            fontSize: "0.8rem",
            padding: "8px 14px",
            transition: "border-color 0.15s",
          }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.borderColor = "var(--accent)")
          }
          onMouseLeave={(e) =>
            (e.currentTarget.style.borderColor = "var(--border)")
          }
        >
          ↻ Refresh
        </button>
      </div>

      {/* Summary stats */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "1px",
          background: "var(--border)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          marginBottom: "24px",
        }}
      >
        {[
          { value: summary.total_searches, label: "total searches" },
          { value: summary.unique_queries, label: "unique queries" },
          { value: `${zeroRate}%`, label: "zero-result rate" },
          {
            value: `${summary.avg_latency_ms.toFixed(1)}ms`,
            label: "avg latency",
          },
        ].map(({ value, label }) => (
          <div
            key={label}
            style={{ background: "var(--surface)", padding: "20px 24px" }}
          >
            <Stat value={value} label={label} />
          </div>
        ))}
      </div>

      {/* Two-column: top queries + zero-result */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "16px",
          marginBottom: "16px",
        }}
      >
        <Card title="Top queries">
          <BarChart data={top} valueKey="count" labelKey="query" />
        </Card>

        <Card title="Zero-result queries">
          {zero.length === 0 ? (
            <p
              style={{
                color: "var(--green)",
                fontSize: "0.8rem",
                fontFamily: "var(--mono)",
              }}
            >
              All queries returned results ✓
            </p>
          ) : (
            <BarChart data={zero} valueKey="count" labelKey="query" />
          )}
        </Card>
      </div>

      {/* Volume over time */}
      <Card title="Search volume over time">
        <Sparkline data={volume} />
      </Card>
    </div>
  );
}
