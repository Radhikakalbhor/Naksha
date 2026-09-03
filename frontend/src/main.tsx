import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import {
  MapContainer,
  TileLayer,
  GeoJSON,
} from "react-leaflet";

import type { GeoJsonObject } from "geojson";
import type { LatLngBoundsExpression } from "leaflet";

import "leaflet/dist/leaflet.css";


// ============================================================
// API
// ============================================================

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// Types
// ============================================================

type PostGISLayer = {
  id: number;
  layer_name: string;
  feature_type: string;
  version: number;
  feature_count: number;
};


// ============================================================
// GeoJSON bounds helper
// ============================================================

function calculateGeoJSONBounds(
  geojson: any
): LatLngBoundsExpression | null {

  const coordinates: number[][] = [];

  function collectCoordinates(value: any): void {
    if (!Array.isArray(value)) {
      return;
    }

    if (
      value.length >= 2 &&
      typeof value[0] === "number" &&
      typeof value[1] === "number"
    ) {
      coordinates.push([value[0], value[1]]);
      return;
    }

    for (const item of value) {
      collectCoordinates(item);
    }
  }

  function collectGeometry(geometry: any): void {
    if (!geometry) {
      return;
    }

    if (
      geometry.type === "GeometryCollection" &&
      Array.isArray(geometry.geometries)
    ) {
      for (const item of geometry.geometries) {
        collectGeometry(item);
      }
      return;
    }

    if (geometry.coordinates) {
      collectCoordinates(geometry.coordinates);
    }
  }

  if (
    geojson?.type === "FeatureCollection" &&
    Array.isArray(geojson.features)
  ) {
    for (const feature of geojson.features) {
      collectGeometry(feature?.geometry);
    }
  } else if (geojson?.type === "Feature") {
    collectGeometry(geojson.geometry);
  } else {
    collectGeometry(geojson);
  }

  if (coordinates.length === 0) {
    return null;
  }

  let minLon = Infinity;
  let maxLon = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;

  for (const coordinate of coordinates) {
    const lon = coordinate[0];
    const lat = coordinate[1];

    minLon = Math.min(minLon, lon);
    maxLon = Math.max(maxLon, lon);
    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
  }

  return [
    [minLat, minLon],
    [maxLat, maxLon],
  ];
}


// ============================================================
// APP
// ============================================================

function App() {

  // File & Upload state
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  // AI Detection state
  const [selectedFeatureType, setSelectedFeatureType] = useState<string>("buildings");
  const [loading, setLoading] = useState(false);
  const [activeOperation, setActiveOperation] = useState("");
  const [message, setMessage] = useState("Select an orthophoto to begin.");

  // Map Bounds
  const [mapBounds, setMapBounds] = useState<LatLngBoundsExpression | null>(null);

  // PostGIS layer state
  const [layers, setLayers] = useState<PostGISLayer[]>([]);
  const [selectedLayer, setSelectedLayer] = useState("");
  const [layerGeoJSON, setLayerGeoJSON] = useState<any | null>(null);
  const [layerLoading, setLayerLoading] = useState(false);

  // Selected Feature for QC & Confidence
  const [selectedFeature, setSelectedFeature] = useState<any | null>(null);
  const [qcActionLoading, setQcActionLoading] = useState(false);

  // GeoJSON Export State
  const [exporting, setExporting] = useState(false);


  // ============================================================
  // 1. Load available PostGIS layers on startup
  // ============================================================

  async function loadLayers() {
    try {
      const response = await fetch(`${API_URL}/layers`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || "Could not load PostGIS layers.");
      }
      setLayers(data.layers || []);
    } catch (error) {
      console.error("Could not load layers:", error);
    }
  }

  useEffect(() => {
    loadLayers();
  }, []);


  // ============================================================
  // 2. Load selected PostGIS layer
  // ============================================================

  async function loadPostGISLayer(layerName: string) {
    if (!layerName) {
      setLayerGeoJSON(null);
      setSelectedFeature(null);
      return;
    }

    setLayerLoading(true);
    setSelectedFeature(null);

    try {
      const response = await fetch(`${API_URL}/layers/${layerName}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Could not load selected layer.");
      }

      setLayerGeoJSON(data);

      const bounds = calculateGeoJSONBounds(data);
      if (bounds) {
        setMapBounds(bounds);
      }

      setMessage(`Layer ${layerName} loaded successfully.`);
    } catch (error) {
      console.error(error);
      setMessage(
        error instanceof Error ? error.message : "Could not load PostGIS layer."
      );
    } finally {
      setLayerLoading(false);
    }
  }


  // ============================================================
  // 3. File selection & backend /ingest call
  // ============================================================

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setFile(selectedFile);
    setSelectedFeature(null);

    if (selectedFile) {
      setMessage(`Selected file: ${selectedFile.name}. Uploading image to backend...`);
      setUploading(true);

      try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_URL}/ingest`, {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (response.ok) {
          setMessage(`File ${selectedFile.name} uploaded successfully. Select target feature type and click Run Detection.`);
        } else {
          setMessage(`File selected: ${selectedFile.name}. Ready for AI analysis.`);
        }
      } catch (error) {
        console.warn("Ingest upload warning:", error);
        setMessage(`File selected: ${selectedFile.name}. Ready for AI analysis.`);
      } finally {
        setUploading(false);
      }
    } else {
      setMessage("Select an orthophoto to begin.");
    }
  }


  // ============================================================
  // 4. Run AI Detection
  // ============================================================

  async function handleRunDetection() {
    if (!file) {
      setMessage("Please select an orthophoto file first.");
      return;
    }

    setLoading(true);
    setActiveOperation(`Running ${selectedFeatureType} detection`);
    setMessage(`Uploading orthophoto and running ${selectedFeatureType} AI detection...`);
    setSelectedFeature(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      let endpoint = `${API_URL}/jobs?feature_type=${selectedFeatureType}`;
      if (selectedFeatureType === "fields") {
        endpoint = `${API_URL}/inference/fields`;
      } else if (selectedFeatureType === "buildings") {
        endpoint = `${API_URL}/inference/buildings`;
      } else if (selectedFeatureType === "roads") {
        endpoint = `${API_URL}/inference/roads`;
      } else if (selectedFeatureType === "trees") {
        endpoint = `${API_URL}/inference/trees`;
      } else if (selectedFeatureType === "water") {
        endpoint = `${API_URL}/inference/water`;
      } else if (selectedFeatureType === "lulc") {
        endpoint = `${API_URL}/inference/lulc`;
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail?.message || data?.detail || "Detection failed."
        );
      }

      // Reload PostGIS layer list
      await loadLayers();

      const layerTypeMap: Record<string, string> = {
        buildings: "uploaded_buildings",
        roads: "uploaded_roads",
        trees: "uploaded_trees",
        water: "uploaded_water",
        lulc: "uploaded_lulc",
        fields: "uploaded_farms",
      };

      const uploadedLayerName = layerTypeMap[selectedFeatureType] || `uploaded_${selectedFeatureType}`;
      setSelectedLayer(uploadedLayerName);
      await loadPostGISLayer(uploadedLayerName);

      setMessage(`${selectedFeatureType} detection completed successfully.`);
    } catch (error) {
      console.error(error);
      setMessage(
        error instanceof Error ? error.message : "Detection failed."
      );
    } finally {
      setLoading(false);
      setActiveOperation("");
    }
  }


  // ============================================================
  // 5. Human QC Accept / Reject
  // ============================================================

  async function handleQCAction(action: "accept" | "reject") {
    if (!selectedFeature) return;

    const featureId = selectedFeature.properties?.feature_id || selectedFeature.id;
    if (!featureId) return;

    setQcActionLoading(true);

    try {
      const response = await fetch(`${API_URL}/qc/${featureId}/${action}`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || `Failed to ${action} feature.`);
      }

      const newStatus = action === "accept" ? "accepted" : "rejected";

      setSelectedFeature((prev: any) => ({
        ...prev,
        properties: {
          ...prev.properties,
          qc_status: newStatus,
        },
      }));

      // Update layerGeoJSON feature list so map instantly reflects QC color
      if (layerGeoJSON && Array.isArray(layerGeoJSON.features)) {
        const updatedFeatures = layerGeoJSON.features.map((f: any) => {
          if ((f.properties?.feature_id || f.id) === featureId) {
            return {
              ...f,
              properties: {
                ...f.properties,
                qc_status: newStatus,
              },
            };
          }
          return f;
        });

        setLayerGeoJSON({
          ...layerGeoJSON,
          features: updatedFeatures,
        });
      }

      setMessage(`Feature #${featureId} marked as ${newStatus.toUpperCase()}.`);
    } catch (error) {
      console.error(error);
      setMessage(
        error instanceof Error ? error.message : `Failed to ${action} feature.`
      );
    } finally {
      setQcActionLoading(false);
    }
  }


  // ============================================================
  // 6. Download GeoJSON Export
  // ============================================================

  async function handleDownloadGeoJSON() {
    const layerToExport = selectedLayer || (layers.length > 0 ? layers[0].layer_name : "");
    if (!layerToExport) {
      setMessage("No PostGIS layer selected for export.");
      return;
    }

    setExporting(true);
    setMessage(`Preparing GeoJSON export for ${layerToExport}...`);

    try {
      const exportUrl = `${API_URL}/layers/${layerToExport}/export?format=geojson`;
      const response = await fetch(exportUrl);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData?.detail?.message || errData?.detail || "Export failed.");
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `${layerToExport}.geojson`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);

      setMessage(`GeoJSON file for ${layerToExport} downloaded successfully.`);
    } catch (error) {
      console.error(error);
      setMessage(
        error instanceof Error ? error.message : "GeoJSON export failed."
      );
    } finally {
      setExporting(false);
    }
  }


  // Shared Button Style Helper
  function buttonStyle(enabled: boolean, background: string): React.CSSProperties {
    return {
      background: enabled ? background : "#9aa9a4",
      color: "white",
      border: "none",
      borderRadius: "8px",
      padding: "11px 18px",
      cursor: enabled ? "pointer" : "not-allowed",
      fontSize: "14px",
      fontWeight: "bold",
    };
  }


  // ============================================================
  // UI Rendering
  // ============================================================

  return (
    <div style={{ minHeight: "100vh", background: "#f4f7f6", fontFamily: "Arial, sans-serif" }}>

      {/* Header */}
      <header style={{ background: "#173f35", color: "white", padding: "18px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "28px" }}>Naksha</h1>
          <p style={{ margin: "6px 0 0", opacity: 0.85 }}>AI-Powered Orthophoto Digitization Platform</p>
        </div>

        {/* Global Export Button */}
        <button
          onClick={handleDownloadGeoJSON}
          disabled={exporting || (!selectedLayer && layers.length === 0)}
          style={{
            background: "#00a86b",
            color: "white",
            border: "none",
            borderRadius: "8px",
            padding: "12px 22px",
            fontSize: "14px",
            fontWeight: "bold",
            cursor: exporting ? "wait" : "pointer",
            boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
          }}
        >
          {exporting ? "Preparing Export..." : "📥 Download GeoJSON"}
        </button>
      </header>

      {/* Main Container */}
      <main style={{ padding: "24px", maxWidth: "1400px", margin: "0 auto" }}>

        {/* Upload & AI Analysis Control Panel */}
        <section style={{ background: "white", borderRadius: "12px", padding: "22px", marginBottom: "20px", boxShadow: "0 2px 10px rgba(0,0,0,0.08)" }}>
          <h2 style={{ marginTop: 0, fontSize: "20px" }}>Orthophoto AI Analysis</h2>
          <p style={{ color: "#555", margin: "0 0 16px 0", fontSize: "14px" }}>
            Upload an orthophoto (.tif, .tiff), select the target feature type, and run automated AI detection.
          </p>

          <div style={{ display: "flex", gap: "20px", alignItems: "flex-end", flexWrap: "wrap", marginBottom: "16px" }}>
            {/* File Upload */}
            <div>
              <label style={{ display: "block", marginBottom: "6px", fontWeight: "bold", fontSize: "14px", color: "#333" }}>
                1. Select Orthophoto Image
              </label>
              <input
                type="file"
                accept=".tif,.tiff,.png,.jpg,.jpeg"
                onChange={handleFileChange}
                disabled={uploading || loading}
                style={{ fontSize: "14px" }}
              />
            </div>

            {/* Feature Type Selector */}
            <div>
              <label style={{ display: "block", marginBottom: "6px", fontWeight: "bold", fontSize: "14px", color: "#333" }}>
                2. Select Feature Type
              </label>
              <select
                value={selectedFeatureType}
                onChange={(e) => setSelectedFeatureType(e.target.value)}
                disabled={loading}
                style={{ padding: "10px 14px", borderRadius: "6px", border: "1px solid #ccc", fontSize: "14px", minWidth: "160px" }}
              >
                <option value="buildings">Buildings</option>
                <option value="roads">Roads</option>
                <option value="trees">Trees</option>
                <option value="water">Water</option>
                <option value="lulc">LULC</option>
                <option value="fields">Fields</option>
              </select>
            </div>

            {/* Run Detection Button */}
            <div>
              <button
                onClick={handleRunDetection}
                disabled={!file || loading || uploading}
                style={buttonStyle(Boolean(file) && !loading && !uploading, "#1f6f5b")}
              >
                {loading ? "Running AI Detection..." : "⚡ Run Detection"}
              </button>
            </div>
          </div>

          {/* System Status Banner */}
          <div style={{ padding: "12px 14px", background: "#eef7f3", borderRadius: "8px", color: "#285f4d", fontSize: "14px" }}>
            {message}
          </div>
        </section>

        {/* Map & Inspection Split Layout */}
        <div style={{ display: "grid", gridTemplateColumns: selectedFeature ? "1fr 350px" : "1fr", gap: "20px" }}>

          {/* Map Container */}
          <section style={{ background: "white", borderRadius: "12px", padding: "16px", boxShadow: "0 2px 10px rgba(0,0,0,0.08)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px", flexWrap: "wrap", gap: "10px" }}>
              <h2 style={{ margin: 0, fontSize: "20px" }}>AI Detection Map</h2>

              {/* PostGIS Layer Dropdown & Export Button */}
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <label style={{ fontSize: "14px", fontWeight: "bold", color: "#444" }}>PostGIS Layer:</label>
                <select
                  value={selectedLayer}
                  onChange={(e) => {
                    const value = e.target.value;
                    setSelectedLayer(value);
                    loadPostGISLayer(value);
                  }}
                  style={{ padding: "8px 12px", borderRadius: "6px", border: "1px solid #ccc", minWidth: "260px", fontSize: "14px" }}
                >
                  <option value="" disabled>Select a PostGIS layer</option>
                  {layers.map((layer) => (
                    <option key={`${layer.layer_name}-${layer.version}-${layer.id}`} value={layer.layer_name}>
                      {layer.feature_type} ({layer.layer_name} v{layer.version}, {layer.feature_count} features)
                    </option>
                  ))}
                </select>

                {layerLoading && <span style={{ fontSize: "13px", color: "#666" }}>Loading...</span>}

                {/* Layer GeoJSON Export */}
                <button
                  onClick={handleDownloadGeoJSON}
                  disabled={exporting || (!selectedLayer && layers.length === 0)}
                  style={{
                    background: "#285f8f",
                    color: "white",
                    border: "none",
                    borderRadius: "6px",
                    padding: "8px 14px",
                    fontSize: "13px",
                    fontWeight: "bold",
                    cursor: "pointer",
                  }}
                >
                  Download GeoJSON
                </button>
              </div>
            </div>

            <div style={{ padding: "8px 12px", background: "#eef7f3", borderRadius: "6px", color: "#285f4d", fontSize: "13px", fontWeight: "bold", marginBottom: "12px" }}>
              💡 Click a detected feature on the map to review it.
            </div>

            {/* Leaflet Map */}
            <MapContainer
              center={mapBounds ? undefined : [46.6578, 16.1166]}
              zoom={17}
              bounds={mapBounds ? mapBounds : undefined}
              style={{ height: "640px", width: "100%", borderRadius: "8px" }}
            >
              <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              <TileLayer
                url="http://127.0.0.1:8080/cog/tiles/WebMercatorQuad/{z}/{x}/{y}?url=%2Fdata%2Fraw%2Fdemo_aoi%2Fdemo_aoi_cog.tif&tilesize=512"
                minZoom={17}
                maxZoom={22}
                opacity={0.85}
              />

              {/* GeoJSON PostGIS Detections Render */}
              {layerGeoJSON && (
                <GeoJSON
                  key={`${selectedLayer}-${layerGeoJSON?.features?.length || 0}`}
                  data={layerGeoJSON}
                  style={(feat: any) => {
                    const featId = feat?.properties?.feature_id || feat?.id;
                    const selId = selectedFeature?.properties?.feature_id || selectedFeature?.id;
                    const isSelected = selId != null && featId === selId;

                    const status = feat?.properties?.qc_status;
                    let stroke = "#7b2cbf";
                    let fill = "#9d4edd";
                    if (status === "accepted") {
                      stroke = "#2e7d32";
                      fill = "#4caf50";
                    } else if (status === "rejected") {
                      stroke = "#c62828";
                      fill = "#ef5350";
                    }
                    return {
                      color: isSelected ? "#ff9800" : stroke,
                      weight: isSelected ? 4 : 2,
                      fillColor: fill,
                      fillOpacity: isSelected ? 0.5 : 0.3,
                    };
                  }}
                  onEachFeature={(feat: any, layer: any) => {
                    const props = feat.properties || {};
                    const confText = props.confidence != null ? `${(props.confidence * 100).toFixed(0)}%` : "Unavailable";
                    layer.bindPopup(`
                      <div style="font-family: Arial, sans-serif;">
                        <strong style="color: #173f35; font-size: 14px;">${props.feature_type || "Feature"}</strong><br/>
                        Feature ID: #${props.feature_id || feat.id || "N/A"}<br/>
                        AI Confidence: <strong>${confText}</strong><br/>
                        QC Status: <strong>${props.qc_status ? props.qc_status.charAt(0).toUpperCase() + props.qc_status.slice(1) : "Pending"}</strong>
                      </div>
                    `);
                    layer.on({
                      click: () => {
                        setSelectedFeature(feat);
                      },
                    });
                  }}
                />
              )}
            </MapContainer>
          </section>

          {/* Feature Review Panel */}
          {selectedFeature && (
            <aside style={{ background: "white", borderRadius: "12px", padding: "20px", boxShadow: "0 2px 10px rgba(0,0,0,0.08)", display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h3 style={{ margin: 0, fontSize: "18px", color: "#173f35" }}>Feature Review</h3>
                <button
                  onClick={() => setSelectedFeature(null)}
                  style={{ background: "none", border: "none", fontSize: "18px", cursor: "pointer", color: "#888" }}
                >
                  ✕
                </button>
              </div>

              {/* Feature Details */}
              <div style={{ background: "#f8f9fa", borderRadius: "8px", padding: "14px", fontSize: "14px" }}>
                <p style={{ margin: "0 0 8px 0" }}>
                  <strong>Feature ID:</strong> #{selectedFeature.properties?.feature_id || selectedFeature.id || "N/A"}
                </p>
                <p style={{ margin: "0 0 8px 0" }}>
                  <strong>Layer:</strong> {selectedFeature.properties?.layer_name || selectedLayer || "N/A"}
                </p>
                <p style={{ margin: "0 0 8px 0" }}>
                  <strong>Feature Type:</strong> {selectedFeature.properties?.feature_type || "N/A"}
                </p>
                <p style={{ margin: 0 }}>
                  <strong>Source Model:</strong> {selectedFeature.properties?.source_model || "N/A"}
                </p>
              </div>

              {/* Confidence Indicator */}
              <div style={{ background: "#eef7f3", borderRadius: "8px", padding: "14px" }}>
                <label style={{ display: "block", fontSize: "13px", color: "#555", marginBottom: "6px", fontWeight: "bold" }}>
                  AI Confidence
                </label>
                {selectedFeature.properties?.confidence != null ? (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "6px" }}>
                      <span style={{ fontSize: "24px", fontWeight: "bold", color: "#173f35" }}>
                        {(selectedFeature.properties.confidence * 100).toFixed(0)}%
                      </span>
                      <span style={{ fontSize: "12px", color: "#666" }}>
                        ({selectedFeature.properties.confidence.toFixed(4)})
                      </span>
                    </div>
                    {/* Progress Bar */}
                    <div style={{ height: "8px", background: "#ddd", borderRadius: "4px", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${Math.min(100, Math.max(0, selectedFeature.properties.confidence * 100))}%`,
                          background:
                            selectedFeature.properties.confidence > 0.75
                              ? "#00a86b"
                              : selectedFeature.properties.confidence > 0.5
                              ? "#f5a623"
                              : "#d9534f",
                        }}
                      />
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: "8px 12px", background: "#e0e0e0", borderRadius: "6px", color: "#555", fontSize: "13px", textAlign: "center", fontWeight: "bold" }}>
                    AI Confidence: Unavailable
                  </div>
                )}
              </div>

              {/* Human QC Action Panel */}
              <div style={{ background: "#fafafa", borderRadius: "8px", padding: "14px", border: "1px solid #eee" }}>
                <label style={{ display: "block", fontSize: "13px", color: "#555", marginBottom: "8px", fontWeight: "bold" }}>
                  Human QC Verification
                </label>

                <div style={{ marginBottom: "12px", fontSize: "14px" }}>
                  Current Status:{" "}
                  <strong style={{
                    color: selectedFeature.properties?.qc_status === "accepted"
                      ? "#2e7d32"
                      : selectedFeature.properties?.qc_status === "rejected"
                      ? "#c62828"
                      : "#f5a623",
                  }}>
                    {selectedFeature.properties?.qc_status
                      ? selectedFeature.properties.qc_status.charAt(0).toUpperCase() + selectedFeature.properties.qc_status.slice(1)
                      : "Pending"}
                  </strong>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                  <button
                    onClick={() => handleQCAction("accept")}
                    disabled={qcActionLoading}
                    style={{
                      background: "#2e7d32",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      padding: "10px",
                      fontSize: "14px",
                      fontWeight: "bold",
                      cursor: qcActionLoading ? "wait" : "pointer",
                    }}
                  >
                    ✓ Accept
                  </button>

                  <button
                    onClick={() => handleQCAction("reject")}
                    disabled={qcActionLoading}
                    style={{
                      background: "#c62828",
                      color: "white",
                      border: "none",
                      borderRadius: "6px",
                      padding: "10px",
                      fontSize: "14px",
                      fontWeight: "bold",
                      cursor: qcActionLoading ? "wait" : "pointer",
                    }}
                  >
                    ✕ Reject
                  </button>
                </div>
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}

// React Root
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
