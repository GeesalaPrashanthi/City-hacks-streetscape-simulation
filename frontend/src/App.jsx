import { useEffect, useMemo, useState } from "react"
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet"
import axios from "axios"
import "leaflet/dist/leaflet.css"

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8001"

function severityColor(severity) {
  if (severity === "compliant") return "#22c55e"
  if (severity === "critical") return "#ef4444"
  if (severity === "high") return "#f59e0b"
  return "#a855f7"
}

function qualityColor(label) {
  if (label === "Good") return "#22c55e"
  if (label === "Fair") return "#f59e0b"
  if (label === "Poor") return "#ef4444"
  return "#94a3b8"
}

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`
}

function useIsMobile(breakpoint = 960) {
  const [isMobile, setIsMobile] = useState(window.innerWidth <= breakpoint)

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= breakpoint)
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [breakpoint])

  return isMobile
}

function StatCard({ label, value, color }) {
  return (
    <div style={{ background: "#1e293b", borderRadius: 8, padding: "10px 14px", borderLeft: `3px solid ${color}` }}>
      <div style={{ fontSize: 22, fontWeight: "bold", color }}>{value}</div>
      <div style={{ fontSize: 12, color: "#94a3b8" }}>{label}</div>
    </div>
  )
}

function LoadingState() {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "#0f172a", flexDirection: "column", gap: 12 }}>
      <div style={{ fontSize: 36 }}>♿</div>
      <div style={{ color: "#3b82f6", fontSize: 18, fontWeight: "bold" }}>SidewalkScan</div>
      <div style={{ color: "#94a3b8" }}>Analyzing 2,754 Brookline sidewalks...</div>
    </div>
  )
}

function MapAuditTab({ data, filter, setFilter, selected, setSelected }) {
  const isMobile = useIsMobile()
  const filtered = data?.sidewalks.filter((s) => {
    if (filter === "compliant") return s.ada_compliant
    if (filter === "violations") return !s.ada_compliant
    if (filter === "critical") return s.severity === "critical"
    if (filter === "high") return s.severity === "high"
    if (filter === "obstructed") return s.obstacle_count > 0
    return true
  }) || []

  const summary = data?.summary

  return (
    <div style={{ display: "flex", flexDirection: isMobile ? "column" : "row", height: "100%", background: "#0f172a", fontFamily: "Arial, sans-serif", color: "white" }}>
      <div style={{ width: isMobile ? "100%" : 300, maxHeight: isMobile ? "45%" : "none", padding: 16, display: "flex", flexDirection: "column", gap: 12, overflowY: "auto", borderRight: isMobile ? "none" : "1px solid #1e293b", borderBottom: isMobile ? "1px solid #1e293b" : "none" }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: "bold", color: "#3b82f6" }}>♿ SidewalkScan</div>
          <div style={{ fontSize: 11, color: "#64748b" }}>Brookline, MA - ADA Compliance Audit</div>
          <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>Based on real Cyvl field inspection data</div>
        </div>

        {summary && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <StatCard label="ADA Compliance Rate" value={`${summary.compliance_rate}%`} color="#22c55e" />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              <StatCard label="Compliant" value={summary.compliant} color="#22c55e" />
              <StatCard label="Violations" value={summary.non_compliant} color="#ef4444" />
              <StatCard label="Missing Sidewalk" value={summary.missing_sidewalk} color="#ef4444" />
              <StatCard label="Poor Condition" value={summary.poor_condition} color="#f59e0b" />
              <StatCard label="Obstructed" value={summary.obstructed} color="#a855f7" />
              <StatCard label="Total Audited" value={summary.total} color="#3b82f6" />
            </div>
          </div>
        )}

        <div>
          <div style={{ fontSize: 11, color: "#64748b", fontWeight: "bold", textTransform: "uppercase", marginBottom: 8 }}>Filter</div>
          {[
            ["all", "🗺 All Sidewalks", "#3b82f6", data?.sidewalks.length],
            ["violations", "🚨 All Violations", "#ef4444", summary?.non_compliant],
            ["critical", "🔴 Missing Sidewalk", "#ef4444", summary?.missing_sidewalk],
            ["high", "🟠 Poor Condition", "#f59e0b", summary?.poor_condition],
            ["compliant", "✅ Compliant", "#22c55e", summary?.compliant],
            ["obstructed", "🚧 Obstructed", "#a855f7", summary?.obstructed],
          ].map(([key, label, color, count]) => (
            <div key={key} onClick={() => setFilter(key)}
              style={{
                background: filter === key ? "#1e3a5f" : "#1e293b",
                border: `1px solid ${filter === key ? color : "#334155"}`,
                borderRadius: 8,
                padding: "8px 12px",
                marginBottom: 6,
                cursor: "pointer",
                fontSize: 13,
                color: filter === key ? color : "#94a3b8",
              }}>
              {label}
              <span style={{ float: "right", fontWeight: "bold", color }}>{count}</span>
            </div>
          ))}
        </div>

        {selected && (
          <div style={{ background: "#1e293b", borderRadius: 8, padding: 12, borderLeft: `3px solid ${severityColor(selected.severity)}` }}>
            <div style={{ fontSize: 12, fontWeight: "bold", marginBottom: 8, color: severityColor(selected.severity) }}>
              📍 Selected Sidewalk
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.9 }}>
              <div>Type: <span style={{ color: "white" }}>{selected.sidewalk_type}</span></div>
              <div>Condition: <span style={{ color: selected.condition === "Poor" ? "#ef4444" : selected.condition === "Fair" ? "#f59e0b" : "#22c55e", fontWeight: "bold" }}>{selected.condition}</span></div>
              <div>Material: <span style={{ color: "white" }}>{selected.material}</span></div>
              <div>ADA Status: <span style={{ color: severityColor(selected.severity), fontWeight: "bold" }}>
                {selected.ada_compliant ? "✅ Compliant" : selected.severity === "critical" ? "🔴 Critical" : selected.severity === "high" ? "🟠 High Priority" : "🟡 Medium Priority"}
              </span></div>
            </div>
            {selected.violations?.length > 0 && (
              <div style={{ marginTop: 8, background: "#0f172a", borderRadius: 6, padding: 8 }}>
                <div style={{ fontSize: 11, color: "#ef4444", fontWeight: "bold", marginBottom: 4 }}>⚠ Issues Found:</div>
                {selected.violations.map((v, i) => (
                  <div key={i} style={{ fontSize: 11, color: "#fca5a5", marginBottom: 3 }}>- {v}</div>
                ))}
              </div>
            )}
            {selected.obstacles?.length > 0 && (
              <div style={{ marginTop: 6 }}>
                <div style={{ fontSize: 11, color: "#a855f7", fontWeight: "bold" }}>Obstacles Nearby:</div>
                {selected.obstacles.map((o, i) => (
                  <div key={i} style={{ fontSize: 11, color: "#94a3b8" }}>- {o.type} - {o.distance_m}m away</div>
                ))}
              </div>
            )}
            {selected.image_url && selected.image_url !== "None" && (
              <img src={selected.image_url} alt="Sidewalk" style={{ width: "100%", borderRadius: 6, marginTop: 8 }} />
            )}
          </div>
        )}

        <div style={{ fontSize: 10, color: "#334155", marginTop: "auto" }}>
          Data © Cyvl Inc., used under ODbL v1.0<br />
          ADA Standards: 28 CFR Part 36<br />
          Violations based on real field inspection data
        </div>
      </div>

      <div style={{ flex: 1, position: "relative", minHeight: isMobile ? 320 : 0 }}>
        <MapContainer center={[42.3318, -71.1212]} zoom={14} style={{ height: "100%", width: "100%" }}>
          <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution="© OpenStreetMap © CARTO" />
          {filtered.map((s, i) => {
            const coords = s.geometry.coordinates
            const isLine = s.geometry.type === "LineString"
            const midCoord = isLine ? coords[Math.floor(coords.length / 2)] : coords
            const lat = midCoord[1]
            const lng = midCoord[0]
            const color = severityColor(s.severity)

            return (
              <CircleMarker key={i}
                center={[lat, lng]}
                radius={selected?.feature_id === s.feature_id ? 10 : 6}
                pathOptions={{ color, fillColor: color, fillOpacity: 0.85, weight: selected?.feature_id === s.feature_id ? 3 : 1 }}
                eventHandlers={{ click: () => setSelected(s) }}>
                <Popup>
                  <div style={{ fontSize: 12 }}>
                    <b>{s.ada_compliant ? "✅ Compliant" : "❌ Violation"}</b><br />
                    Type: {s.sidewalk_type}<br />
                    Condition: {s.condition}<br />
                    Material: {s.material}<br />
                    {s.violations?.length > 0 && <span style={{ color: "#dc2626" }}>{s.violations[0]}</span>}
                  </div>
                </Popup>
              </CircleMarker>
            )
          })}
        </MapContainer>

        <div style={{ position: "absolute", bottom: 16, right: 16, zIndex: 1000, background: "rgba(15,23,42,0.95)", padding: "12px 16px", borderRadius: 8, fontSize: 12, border: "1px solid #1e293b" }}>
          <div style={{ fontWeight: "bold", marginBottom: 8, color: "white" }}>Legend</div>
          {[
            ["#22c55e", "✅ Compliant"],
            ["#ef4444", "🔴 Missing Sidewalk (Critical)"],
            ["#f59e0b", "🟠 Poor Condition (High)"],
            ["#a855f7", "🟡 Material/Obstacle (Medium)"],
          ].map(([c, l]) => (
            <div key={l} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4, color: "#94a3b8" }}>
              <div style={{ width: 10, height: 10, borderRadius: "50%", background: c, flexShrink: 0 }} />{l}
            </div>
          ))}
          <div style={{ marginTop: 8, color: "#64748b", fontSize: 10, borderTop: "1px solid #1e293b", paddingTop: 6 }}>
            Showing {filtered.length} of {data?.sidewalks.length} sidewalks
          </div>
        </div>

        <div style={{ position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)", zIndex: 1000, background: "#0f172a", padding: "6px 16px", borderRadius: 20, fontSize: 13, fontWeight: "bold", border: "1px solid #334155", color: "#94a3b8", whiteSpace: "nowrap" }}>
          ♿ Brookline ADA Sidewalk Audit - {filtered.length} sidewalks shown
        </div>
      </div>
    </div>
  )
}

function ImageAdvisorTab() {
  const isMobile = useIsMobile()
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState("")
  const [guidancePrompt, setGuidancePrompt] = useState("")
  const [includeGemini, setIncludeGemini] = useState(true)
  const [result, setResult] = useState(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const sortedProbs = useMemo(() => {
    if (!result?.probabilities) return []
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
  }, [result])

  function handleFileChange(event) {
    const file = event.target.files?.[0]
    setResult(null)
    setError("")
    if (!file) {
      setImageFile(null)
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
        setPreviewUrl("")
      }
      return
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setImageFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  async function handleAnalyze() {
    if (!imageFile) {
      setError("Please upload an image first.")
      return
    }

    const formData = new FormData()
    formData.append("image", imageFile)
    formData.append("include_gemini", String(includeGemini))
    formData.append("guidance_prompt", guidancePrompt)
    formData.append("enforce_sidewalk_check", "true")

    setLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await axios.post(`${API}/predict-sidewalk`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setResult(response.data)
    } catch (requestError) {
      const detail = requestError?.response?.data?.detail
      setError(detail || requestError.message || "Prediction request failed.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "380px 1fr", gap: 16, height: "100%", padding: 16, background: "#0f172a", color: "white", fontFamily: "Arial, sans-serif", overflowY: "auto" }}>
      <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 10, padding: 16, overflowY: "auto" }}>
        <div style={{ fontSize: 18, fontWeight: "bold", color: "#38bdf8", marginBottom: 6 }}>Image AI Advisor</div>
        <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 14 }}>
          Upload a sidewalk image and get model prediction plus optional Gemini recommendations.
        </div>

        <label style={{ display: "block", fontSize: 12, color: "#cbd5e1", marginBottom: 6 }}>Upload Sidewalk Photo</label>
        <input type="file" accept="image/*" onChange={handleFileChange} style={{ width: "100%", marginBottom: 12 }} />

        {previewUrl && (
          <img src={previewUrl} alt="Preview" style={{ width: "100%", borderRadius: 8, border: "1px solid #334155", marginBottom: 12 }} />
        )}

        <label style={{ display: "block", fontSize: 12, color: "#cbd5e1", marginBottom: 6 }}>Custom Guidance Prompt (optional)</label>
        <textarea
          value={guidancePrompt}
          onChange={(event) => setGuidancePrompt(event.target.value)}
          placeholder="Example: focus on wheelchair accessibility and low-cost fixes."
          rows={4}
          style={{ width: "100%", background: "#0b1220", color: "white", border: "1px solid #334155", borderRadius: 8, padding: 10, resize: "vertical", marginBottom: 10 }}
        />

        <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13, color: "#cbd5e1", marginBottom: 12 }}>
          <input type="checkbox" checked={includeGemini} onChange={(event) => setIncludeGemini(event.target.checked)} />
          Include Gemini recommendations
        </label>

        <button
          onClick={handleAnalyze}
          disabled={loading}
          style={{
            width: "100%",
            background: loading ? "#334155" : "#0284c7",
            color: "white",
            border: "none",
            borderRadius: 8,
            padding: "10px 12px",
            fontWeight: "bold",
            cursor: loading ? "default" : "pointer",
          }}
        >
          {loading ? "Analyzing..." : "Analyze Sidewalk"}
        </button>

        {error && (
          <div style={{ marginTop: 12, fontSize: 12, color: "#fca5a5", background: "#450a0a", border: "1px solid #991b1b", borderRadius: 8, padding: 10 }}>
            {error}
          </div>
        )}
      </div>

      <div style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 10, padding: 16, overflowY: "auto" }}>
        {!result && (
          <div style={{ color: "#64748b", fontSize: 14 }}>
            Run an analysis to view model confidence and AI guidance.
          </div>
        )}

        {result && (
          <>
            {result.classification_skipped && (
              <div style={{ marginBottom: 18, background: "#1f1720", border: "1px solid #7f1d1d", borderRadius: 10, padding: 14 }}>
                <div style={{ fontSize: 12, color: "#fca5a5", marginBottom: 6 }}>Sidewalk Presence Check</div>
                <div style={{ fontSize: 26, fontWeight: "bold", color: "#ef4444", marginBottom: 6 }}>
                  No Sidewalk Detected
                </div>
                <div style={{ fontSize: 13, color: "#fecaca", lineHeight: 1.5 }}>
                  {result.sidewalk_check?.reason || "This image does not appear to contain a real sidewalk."}
                </div>
                {typeof result.sidewalk_check?.confidence === "number" && (
                  <div style={{ fontSize: 12, color: "#fca5a5", marginTop: 8 }}>
                    Detection confidence: {formatPct(result.sidewalk_check.confidence)}
                  </div>
                )}
              </div>
            )}

            {!result.classification_skipped && (
              <>
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 12, color: "#94a3b8" }}>Model Prediction</div>
                  <div style={{ fontSize: 30, fontWeight: "bold", color: qualityColor(result.predicted_class) }}>
                    {result.predicted_class}
                  </div>
                  <div style={{ fontSize: 13, color: "#cbd5e1" }}>Confidence: {formatPct(result.confidence || 0)}</div>
                </div>

                <div style={{ marginBottom: 18 }}>
                  <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>Class Probabilities</div>
                  {sortedProbs.map(([label, score]) => (
                    <div key={label} style={{ marginBottom: 10 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                        <span style={{ color: qualityColor(label), fontWeight: "bold" }}>{label}</span>
                        <span style={{ color: "#cbd5e1" }}>{formatPct(score)}</span>
                      </div>
                      <div style={{ height: 8, borderRadius: 999, background: "#1f2937", overflow: "hidden" }}>
                        <div style={{ width: `${Math.max(score * 100, 2)}%`, height: "100%", background: qualityColor(label) }} />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {result.sidewalk_check?.error && (
              <div style={{ marginBottom: 12, fontSize: 12, color: "#fbbf24", background: "#2b1b08", border: "1px solid #92400e", borderRadius: 8, padding: 10 }}>
                Sidewalk check unavailable: {result.sidewalk_check.error}. Classifier result shown anyway.
              </div>
            )}

            <div style={{ background: "#0b1220", border: "1px solid #1f2937", borderRadius: 10, padding: 14 }}>
              <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>Gemini Summary</div>
              {result.gemini_model && (
                <div style={{ fontSize: 11, color: "#64748b", marginBottom: 8 }}>
                  Model: {result.gemini_model}
                </div>
              )}
              {result.gemini_summary && (
                <div style={{ fontSize: 14, lineHeight: 1.6, color: "#e2e8f0", whiteSpace: "pre-wrap" }}>
                  {result.gemini_summary}
                </div>
              )}
              {!result.gemini_summary && result.gemini_error && (
                <div style={{ fontSize: 13, color: "#fbbf24" }}>
                  {result.gemini_error}
                </div>
              )}
              {!result.gemini_summary && !result.gemini_error && (
                <div style={{ fontSize: 13, color: "#64748b" }}>
                  Gemini was skipped for this analysis.
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState("map")
  const [data, setData] = useState(null)
  const [filter, setFilter] = useState("all")
  const [selected, setSelected] = useState(null)
  const [loadingMap, setLoadingMap] = useState(true)

  useEffect(() => {
    axios.get(`${API}/sidewalks`)
      .then((res) => setData(res.data))
      .finally(() => setLoadingMap(false))
  }, [])

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#0f172a", color: "white" }}>
      <div style={{ height: 56, borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", padding: "0 16px", gap: 10 }}>
        <button
          onClick={() => setActiveTab("map")}
          style={{
            border: "none",
            background: activeTab === "map" ? "#1d4ed8" : "#1e293b",
            color: "white",
            fontWeight: "bold",
            borderRadius: 8,
            padding: "8px 12px",
            cursor: "pointer",
          }}
        >
          Map Audit
        </button>
        <button
          onClick={() => setActiveTab("advisor")}
          style={{
            border: "none",
            background: activeTab === "advisor" ? "#1d4ed8" : "#1e293b",
            color: "white",
            fontWeight: "bold",
            borderRadius: 8,
            padding: "8px 12px",
            cursor: "pointer",
          }}
        >
          Image AI Advisor
        </button>
      </div>

      <div style={{ flex: 1, minHeight: 0 }}>
        {activeTab === "map" && (loadingMap ? <LoadingState /> : <MapAuditTab data={data} filter={filter} setFilter={setFilter} selected={selected} setSelected={setSelected} />)}
        {activeTab === "advisor" && <ImageAdvisorTab />}
      </div>
    </div>
  )
}
