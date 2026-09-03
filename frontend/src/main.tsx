import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

// ============================================================
// API Configuration
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

// Feature Types Definition
const FEATURE_TYPES = [
  { id: "buildings", label: "Buildings" },
  { id: "roads", label: "Roads" },
  { id: "trees", label: "Trees" },
  { id: "water", label: "Water" },
  { id: "fields", label: "Farms / Fields" },
  { id: "lulc", label: "LULC" },
];

// ============================================================
// GeoJSON Bounds Helper
// ============================================================
function calculateGeoJSONBounds(geojson: any): LatLngBoundsExpression | null {
  const coordinates: number[][] = [];

  function collectCoordinates(value: any): void {
    if (!Array.isArray(value)) return;
    if (value.length >= 2 && typeof value[0] === "number" && typeof value[1] === "number") {
      coordinates.push([value[0], value[1]]);
      return;
    }
    for (const item of value) {
      collectCoordinates(item);
    }
  }

  function collectGeometry(geometry: any): void {
    if (!geometry) return;
    if (geometry.type === "GeometryCollection" && Array.isArray(geometry.geometries)) {
      for (const item of geometry.geometries) {
        collectGeometry(item);
      }
      return;
    }
    if (geometry.coordinates) {
      collectCoordinates(geometry.coordinates);
    }
  }

  if (geojson?.type === "FeatureCollection" && Array.isArray(geojson.features)) {
    for (const feature of geojson.features) {
      collectGeometry(feature?.geometry);
    }
  } else if (geojson?.type === "Feature") {
    collectGeometry(geojson.geometry);
  } else {
    collectGeometry(geojson);
  }

  if (coordinates.length === 0) return null;

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
// Main Application Component
// ============================================================
function App() {
  // State Management
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  const [selectedFeatureType, setSelectedFeatureType] = useState<string>("buildings");
  const [loading, setLoading] = useState(false);
  const [activeOperation, setActiveOperation] = useState("");
  const [message, setMessage] = useState("Select an orthophoto image to begin digitization.");

  const [mapBounds, setMapBounds] = useState<LatLngBoundsExpression | null>(null);

  const [layers, setLayers] = useState<PostGISLayer[]>([]);
  const [selectedLayer, setSelectedLayer] = useState("");
  const [layerGeoJSON, setLayerGeoJSON] = useState<any | null>(null);
  const [layerLoading, setLayerLoading] = useState(false);

  const [selectedFeature, setSelectedFeature] = useState<any | null>(null);
  const [qcActionLoading, setQcActionLoading] = useState(false);
  const [qcFeedback, setQcFeedback] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const [exporting, setExporting] = useState(false);

  // ============================================================
  // 1. Load Available PostGIS Layers on Startup
  // ============================================================
  async function loadLayers() {
    try {
      const response = await fetch(`${API_URL}/layers`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || "Could not load PostGIS layers.");
      }
      const fetchedLayers: PostGISLayer[] = data.layers || [];
      setLayers(fetchedLayers);

      const hasUploadedFarms = fetchedLayers.some((l) => l.layer_name === "uploaded_farms");
      if (hasUploadedFarms && !selectedLayer) {
        setSelectedLayer("uploaded_farms");
        loadPostGISLayer("uploaded_farms");
      }
    } catch (error) {
      console.error("Could not load layers:", error);
    }
  }

  useEffect(() => {
    loadLayers();
  }, []);

  // ============================================================
  // 2. Load Selected PostGIS Layer
  // ============================================================
  async function loadPostGISLayer(layerName: string) {
    if (!layerName) {
      setLayerGeoJSON(null);
      setSelectedFeature(null);
      setQcFeedback(null);
      return;
    }

    setLayerLoading(true);
    setSelectedFeature(null);
    setQcFeedback(null);

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

      setMessage(`Layer "${layerName}" loaded successfully.`);
    } catch (error) {
      console.error(error);
      setMessage(error instanceof Error ? error.message : "Could not load PostGIS layer.");
    } finally {
      setLayerLoading(false);
    }
  }

  // ============================================================
  // 3. File Selection & Ingest Call
  // ============================================================
  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setFile(selectedFile);
    setSelectedFeature(null);
    setQcFeedback(null);

    if (selectedFile) {
      setMessage(`File selected: ${selectedFile.name}. Select target feature type and click Run Detection.`);

      const formData = new FormData();
      formData.append("file", selectedFile);
      fetch(`${API_URL}/ingest`, {
        method: "POST",
        body: formData,
      }).catch((error) => {
        console.warn("Ingest upload warning:", error);
      });
    } else {
      setMessage("Select an orthophoto image to begin digitization.");
    }
  }

  // ============================================================
  // 4. Run AI Detection
  // ============================================================
  async function handleRunDetection() {
    if (!file) {
      setMessage("Please select an orthophoto image file first.");
      return;
    }

    setLoading(true);
    setActiveOperation(`Running ${selectedFeatureType} detection`);
    setMessage(`Uploading orthophoto and running ${selectedFeatureType} AI detection...`);
    setSelectedFeature(null);
    setQcFeedback(null);

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
        const detailMsg = typeof data?.detail === "string"
          ? data.detail
          : data?.detail?.message || data?.detail?.error || "Detection failed.";
        throw new Error(detailMsg);
      }

      await loadLayers();

      const layerTypeMap: Record<string, string> = {
        buildings: "uploaded_buildings",
        roads: "uploaded_roads",
        trees: "uploaded_trees",
        water: "uploaded_water",
        lulc: "uploaded_lulc",
        fields: "uploaded_farms",
        farms: "uploaded_farms",
      };

      const uploadedLayerName = layerTypeMap[selectedFeatureType] || `uploaded_${selectedFeatureType}`;
      setSelectedLayer(uploadedLayerName);
      await loadPostGISLayer(uploadedLayerName);

      setMessage(`${selectedFeatureType} detection completed successfully.`);
    } catch (error) {
      console.error(error);
      setMessage(error instanceof Error ? error.message : "Detection failed.");
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
    setQcFeedback(null);

    try {
      const response = await fetch(`${API_URL}/qc/${featureId}/${action}`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || `Failed to ${action} feature.`);
      }

      const authoritativeStatus = data.qc_status || (action === "accept" ? "accepted" : "rejected");

      setSelectedFeature((prev: any) => ({
        ...prev,
        properties: {
          ...prev?.properties,
          qc_status: authoritativeStatus,
        },
      }));

      if (layerGeoJSON && Array.isArray(layerGeoJSON.features)) {
        const updatedFeatures = layerGeoJSON.features.map((f: any) => {
          if ((f.properties?.feature_id || f.id) === featureId) {
            return {
              ...f,
              properties: {
                ...f.properties,
                qc_status: authoritativeStatus,
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

      if (authoritativeStatus === "accepted") {
        setQcFeedback({ message: `Feature #${featureId} accepted.`, type: "success" });
      } else {
        setQcFeedback({ message: `Feature #${featureId} rejected.`, type: "error" });
      }

      setMessage(`Feature #${featureId} marked as ${authoritativeStatus.toUpperCase()}.`);
    } catch (error) {
      console.error(error);
      const errMsg = error instanceof Error ? error.message : `Failed to ${action} feature.`;
      setQcFeedback({ message: errMsg, type: "error" });
      setMessage(errMsg);
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
    setMessage(`Preparing GeoJSON export for "${layerToExport}"...`);

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

      setMessage(`GeoJSON file for "${layerToExport}" downloaded successfully.`);
    } catch (error) {
      console.error(error);
      setMessage(error instanceof Error ? error.message : "GeoJSON export failed.");
    } finally {
      setExporting(false);
    }
  }

  // ============================================================
  // UI Layout & Render
  // ============================================================
  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#f5f5f7", color: "#1d1d1f" }}>
      {/* ------------------------------------------------------------
          Global Header (Apple Design System Nav Bar)
         ------------------------------------------------------------ */}
      <header
        style={{
          backgroundColor: "rgba(255, 255, 255, 0.85)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          borderBottom: "1px solid #e5e5e7",
          position: "sticky",
          top: 0,
          zIndex: 1000,
          padding: "14px 28px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontSize: "20px", fontWeight: "700", letterSpacing: "-0.5px", color: "#1d1d1f" }}>
            Naksha
          </span>
          <span
            style={{
              fontSize: "11px",
              fontWeight: "600",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              backgroundColor: "#f2f2f7",
              color: "#86868b",
              padding: "4px 8px",
              borderRadius: "9999px",
              border: "1px solid #e5e5e7",
            }}
          >
            AI Platform
          </span>
        </div>

        <button
          onClick={handleDownloadGeoJSON}
          disabled={exporting || (!selectedLayer && layers.length === 0)}
          style={{
            backgroundColor: "#0066cc",
            color: "#ffffff",
            border: "none",
            borderRadius: "9999px",
            padding: "9px 18px",
            fontSize: "13px",
            fontWeight: "500",
            cursor: exporting || (!selectedLayer && layers.length === 0) ? "not-allowed" : "pointer",
            opacity: exporting || (!selectedLayer && layers.length === 0) ? 0.5 : 1,
            transition: "all 0.15s ease",
            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.04)",
          }}
        >
          {exporting ? "Preparing Export..." : "Export GeoJSON"}
        </button>
      </header>

      {/* ------------------------------------------------------------
          Main Workspace Area
         ------------------------------------------------------------ */}
      <main style={{ maxWidth: "1440px", margin: "0 auto", padding: "28px 24px 48px" }}>
        
        {/* Orthophoto AI Control Surface Card */}
        <section
          style={{
            backgroundColor: "#ffffff",
            borderRadius: "16px",
            padding: "24px 28px",
            marginBottom: "24px",
            border: "1px solid #e5e5e7",
            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.04)",
          }}
        >
          <div style={{ marginBottom: "20px" }}>
            <h1 style={{ margin: "0 0 4px 0", fontSize: "21px", fontWeight: "600", letterSpacing: "-0.3px" }}>
              Orthophoto AI Digitization
            </h1>
            <p style={{ margin: 0, fontSize: "14px", color: "#86868b" }}>
              Select an orthophoto file, choose the target feature layer, and execute automated AI segmentation.
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* Step 1: Feature Type Segmented Pill Control */}
            <div>
              <label style={{ display: "block", fontSize: "12px", fontWeight: "600", color: "#86868b", letterSpacing: "0.4px", marginBottom: "8px", textTransform: "uppercase" }}>
                Target Feature Type
              </label>
              <div
                style={{
                  display: "inline-flex",
                  gap: "4px",
                  backgroundColor: "#f2f2f7",
                  padding: "4px",
                  borderRadius: "9999px",
                  border: "1px solid #e5e5e7",
                  flexWrap: "wrap",
                }}
              >
                {FEATURE_TYPES.map((type) => {
                  const isActive = selectedFeatureType === type.id;
                  return (
                    <button
                      key={type.id}
                      onClick={() => setSelectedFeatureType(type.id)}
                      disabled={loading}
                      style={{
                        backgroundColor: isActive ? "#ffffff" : "transparent",
                        color: isActive ? "#0066cc" : "#1d1d1f",
                        border: "none",
                        borderRadius: "9999px",
                        padding: "7px 16px",
                        fontSize: "13px",
                        fontWeight: isActive ? "600" : "500",
                        cursor: loading ? "not-allowed" : "pointer",
                        boxShadow: isActive ? "0 2px 8px rgba(0, 0, 0, 0.04)" : "none",
                        transition: "all 0.15s ease",
                      }}
                    >
                      {type.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Step 2: File Selector & Action Controls */}
            <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <label
                  htmlFor="orthophoto-upload"
                  style={{
                    backgroundColor: "#f2f2f7",
                    color: "#1d1d1f",
                    border: "1px solid #e5e5e7",
                    borderRadius: "8px",
                    padding: "9px 16px",
                    fontSize: "13px",
                    fontWeight: "500",
                    cursor: uploading || loading ? "not-allowed" : "pointer",
                    display: "inline-block",
                  }}
                >
                  {file ? "Change Image" : "Choose Image (.tif)"}
                </label>
                <input
                  id="orthophoto-upload"
                  type="file"
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  onChange={handleFileChange}
                  disabled={uploading || loading}
                  style={{ display: "none" }}
                />
                {file && (
                  <span style={{ fontSize: "13px", color: "#86868b", maxWidth: "240px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {file.name}
                  </span>
                )}
              </div>

              <button
                onClick={handleRunDetection}
                disabled={!file || loading || uploading}
                style={{
                  backgroundColor: !file || loading || uploading ? "#e5e5e7" : "#0066cc",
                  color: !file || loading || uploading ? "#86868b" : "#ffffff",
                  border: "none",
                  borderRadius: "9999px",
                  padding: "10px 22px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: !file || loading || uploading ? "not-allowed" : "pointer",
                  transition: "all 0.15s ease",
                  boxShadow: !file || loading || uploading ? "none" : "0 2px 8px rgba(0, 0, 0, 0.04)",
                }}
              >
                {loading ? "Processing AI Detection..." : "Run AI Detection"}
              </button>
            </div>
          </div>

          {/* System Notification Pill */}
          <div
            style={{
              marginTop: "20px",
              padding: "10px 16px",
              backgroundColor: "#e8f2ff",
              borderRadius: "8px",
              color: "#0066cc",
              fontSize: "13px",
              fontWeight: "500",
              border: "1px solid #cce0ff",
            }}
          >
            {message}
          </div>
        </section>

        {/* ------------------------------------------------------------
            Map & Review Panel Split Workbench
           ------------------------------------------------------------ */}
        <div style={{ display: "grid", gridTemplateColumns: selectedFeature ? "1fr 380px" : "1fr", gap: "24px" }}>
          
          {/* Map Viewport Container */}
          <section
            style={{
              backgroundColor: "#ffffff",
              borderRadius: "16px",
              padding: "20px",
              border: "1px solid #e5e5e7",
              boxShadow: "0 4px 16px rgba(0, 0, 0, 0.06)",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {/* Map Header & PostGIS Selector Bar */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <h2 style={{ margin: 0, fontSize: "17px", fontWeight: "600" }}>AI Detection Map</h2>
                {layerLoading && (
                  <span style={{ fontSize: "12px", color: "#0066cc", fontWeight: "500" }}>Loading layer...</span>
                )}
              </div>

              {/* PostGIS Layer Dropdown */}
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <label style={{ fontSize: "13px", color: "#86868b", fontWeight: "500" }}>PostGIS Layer:</label>
                <select
                  value={selectedLayer}
                  onChange={(e) => {
                    const value = e.target.value;
                    setSelectedLayer(value);
                    loadPostGISLayer(value);
                  }}
                  style={{
                    backgroundColor: "#f2f2f7",
                    color: "#1d1d1f",
                    border: "1px solid #e5e5e7",
                    borderRadius: "8px",
                    padding: "7px 12px",
                    fontSize: "13px",
                    fontWeight: "500",
                    minWidth: "260px",
                    outline: "none",
                  }}
                >
                  <option value="" disabled>Select a PostGIS layer</option>
                  {layers.map((layer) => (
                    <option key={`${layer.layer_name}-${layer.version}-${layer.id}`} value={layer.layer_name}>
                      {layer.feature_type} ({layer.layer_name} v{layer.version}, {layer.feature_count} features)
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Instruction Tip */}
            <div
              style={{
                padding: "8px 14px",
                backgroundColor: "#f2f2f7",
                borderRadius: "8px",
                color: "#1d1d1f",
                fontSize: "12px",
                fontWeight: "500",
                marginBottom: "16px",
                border: "1px solid #e5e5e7",
              }}
            >
              Select any detected feature polygon on the map to review details, view AI confidence scores, and perform human QC verification.
            </div>

            {/* Leaflet Map Frame */}
            <div style={{ height: "640px", width: "100%", borderRadius: "12px", overflow: "hidden", border: "1px solid #e5e5e7" }}>
              <MapContainer
                center={mapBounds ? undefined : [46.6578, 16.1166]}
                zoom={17}
                bounds={mapBounds ? mapBounds : undefined}
                style={{ height: "100%", width: "100%" }}
              >
                <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                <TileLayer
                  url="http://127.0.0.1:8080/cog/tiles/WebMercatorQuad/{z}/{x}/{y}?url=%2Fdata%2Fraw%2Fdemo_aoi%2Fdemo_aoi_cog.tif&tilesize=512"
                  minZoom={17}
                  maxZoom={22}
                  opacity={0.85}
                />

                {layerGeoJSON && (
                  <GeoJSON
                    key={`${selectedLayer}-${layerGeoJSON?.features?.length || 0}`}
                    data={layerGeoJSON}
                    style={(feat: any) => {
                      const featId = feat?.properties?.feature_id || feat?.id;
                      const selId = selectedFeature?.properties?.feature_id || selectedFeature?.id;
                      const isSelected = selId != null && featId === selId;

                      const status = feat?.properties?.qc_status;
                      let stroke = "#0066cc";
                      let fill = "#0066cc";
                      if (status === "accepted") {
                        stroke = "#28cd41";
                        fill = "#28cd41";
                      } else if (status === "rejected") {
                        stroke = "#ff3b30";
                        fill = "#ff3b30";
                      }
                      return {
                        color: isSelected ? "#ff9500" : stroke,
                        weight: isSelected ? 4 : 2,
                        fillColor: fill,
                        fillOpacity: isSelected ? 0.5 : 0.25,
                      };
                    }}
                    onEachFeature={(feat: any, layer: any) => {
                      const props = feat.properties || {};
                      const confText = props.confidence != null ? `${(props.confidence * 100).toFixed(0)}%` : "Unavailable";
                      layer.bindPopup(`
                        <div style="font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif; padding: 4px;">
                          <strong style="color: #1d1d1f; font-size: 14px;">${props.feature_type || "Feature"}</strong><br/>
                          <span style="font-size: 12px; color: #86868b;">ID: #${props.feature_id || feat.id || "N/A"}</span><br/>
                          <span style="font-size: 12px; color: #1d1d1f;">AI Confidence: <strong>${confText}</strong></span><br/>
                          <span style="font-size: 12px; color: #86868b;">QC Status: <strong>${props.qc_status ? props.qc_status.charAt(0).toUpperCase() + props.qc_status.slice(1) : "Pending"}</strong></span>
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
            </div>
          </section>

          {/* ------------------------------------------------------------
              Feature Review Sidebar Panel
             ------------------------------------------------------------ */}
          {selectedFeature && (
            <aside
              style={{
                backgroundColor: "#ffffff",
                borderRadius: "16px",
                padding: "24px",
                border: "1px solid #e5e5e7",
                boxShadow: "0 4px 16px rgba(0, 0, 0, 0.06)",
                display: "flex",
                flexDirection: "column",
                gap: "20px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h3 style={{ margin: 0, fontSize: "17px", fontWeight: "600" }}>Feature Review</h3>
                <button
                  onClick={() => setSelectedFeature(null)}
                  style={{
                    backgroundColor: "#f2f2f7",
                    border: "none",
                    borderRadius: "9999px",
                    width: "28px",
                    height: "28px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "14px",
                    cursor: "pointer",
                    color: "#86868b",
                  }}
                >
                  ✕
                </button>
              </div>

              {/* Feature Details Section */}
              <div style={{ backgroundColor: "#fbfbfd", borderRadius: "12px", padding: "16px", border: "1px solid #e5e5e7", fontSize: "13px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <span style={{ color: "#86868b" }}>Feature ID</span>
                  <span style={{ fontWeight: "600" }}>#{selectedFeature.properties?.feature_id || selectedFeature.id || "N/A"}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <span style={{ color: "#86868b" }}>Layer Name</span>
                  <span style={{ fontWeight: "600" }}>{selectedFeature.properties?.layer_name || selectedLayer || "N/A"}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <span style={{ color: "#86868b" }}>Feature Type</span>
                  <span style={{ fontWeight: "600", textTransform: "capitalize" }}>{selectedFeature.properties?.feature_type || "N/A"}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#86868b" }}>Source Model</span>
                  <span style={{ fontWeight: "600" }}>{selectedFeature.properties?.source_model || "N/A"}</span>
                </div>
              </div>

              {/* Real AI Confidence Indicator Section */}
              <div style={{ backgroundColor: "#fbfbfd", borderRadius: "12px", padding: "16px", border: "1px solid #e5e5e7" }}>
                <label style={{ display: "block", fontSize: "12px", color: "#86868b", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: "8px" }}>
                  AI Confidence Score
                </label>

                {selectedFeature.properties?.confidence != null ? (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "8px" }}>
                      <span style={{ fontSize: "28px", fontWeight: "700", color: "#1d1d1f", letterSpacing: "-0.5px" }}>
                        {(selectedFeature.properties.confidence * 100).toFixed(0)}%
                      </span>
                      <span style={{ fontSize: "12px", color: "#86868b" }}>
                        ({selectedFeature.properties.confidence.toFixed(4)})
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div style={{ height: "6px", backgroundColor: "#e5e5e7", borderRadius: "9999px", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${Math.min(100, Math.max(0, selectedFeature.properties.confidence * 100))}%`,
                          backgroundColor:
                            selectedFeature.properties.confidence > 0.75
                              ? "#28cd41"
                              : selectedFeature.properties.confidence > 0.5
                              ? "#ff9500"
                              : "#ff3b30",
                          borderRadius: "9999px",
                          transition: "width 0.3s ease",
                        }}
                      />
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: "8px 12px", backgroundColor: "#f2f2f7", borderRadius: "8px", color: "#86868b", fontSize: "13px", textAlign: "center", fontWeight: "500" }}>
                    AI Confidence: Unavailable
                  </div>
                )}
              </div>

              {/* Human Quality Control Verification Panel */}
              <div style={{ backgroundColor: "#fbfbfd", borderRadius: "12px", padding: "16px", border: "1px solid #e5e5e7" }}>
                <label style={{ display: "block", fontSize: "12px", color: "#86868b", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: "8px" }}>
                  Human QC Verification
                </label>

                <div style={{ marginBottom: "12px", fontSize: "13px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ color: "#86868b" }}>Current Status</span>
                  <span
                    style={{
                      fontWeight: "600",
                      padding: "3px 10px",
                      borderRadius: "9999px",
                      fontSize: "12px",
                      backgroundColor:
                        selectedFeature.properties?.qc_status === "accepted"
                          ? "#eafaf1"
                          : selectedFeature.properties?.qc_status === "rejected"
                          ? "#ffebeb"
                          : "#fff7e6",
                      color:
                        selectedFeature.properties?.qc_status === "accepted"
                          ? "#28cd41"
                          : selectedFeature.properties?.qc_status === "rejected"
                          ? "#ff3b30"
                          : "#ff9500",
                    }}
                  >
                    {selectedFeature.properties?.qc_status
                      ? selectedFeature.properties.qc_status.charAt(0).toUpperCase() + selectedFeature.properties.qc_status.slice(1)
                      : "Pending"}
                  </span>
                </div>

                {/* Feedback Toast */}
                {qcFeedback && (
                  <div
                    style={{
                      padding: "8px 12px",
                      borderRadius: "8px",
                      fontSize: "12px",
                      fontWeight: "500",
                      marginBottom: "12px",
                      backgroundColor: qcFeedback.type === "success" ? "#eafaf1" : "#ffebeb",
                      color: qcFeedback.type === "success" ? "#28cd41" : "#ff3b30",
                      border: `1px solid ${qcFeedback.type === "success" ? "#c3f2d2" : "#ffd1d1"}`,
                    }}
                  >
                    {qcFeedback.message}
                  </div>
                )}

                {/* Action Buttons */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                  <button
                    onClick={() => handleQCAction("accept")}
                    disabled={qcActionLoading || selectedFeature.properties?.qc_status === "accepted"}
                    style={{
                      backgroundColor: selectedFeature.properties?.qc_status === "accepted" ? "#f2f2f7" : "#28cd41",
                      color: selectedFeature.properties?.qc_status === "accepted" ? "#86868b" : "#ffffff",
                      border: "none",
                      borderRadius: "9999px",
                      padding: "9px",
                      fontSize: "13px",
                      fontWeight: "500",
                      cursor: qcActionLoading || selectedFeature.properties?.qc_status === "accepted" ? "not-allowed" : "pointer",
                      transition: "all 0.15s ease",
                    }}
                  >
                    ✓ Accept
                  </button>

                  <button
                    onClick={() => handleQCAction("reject")}
                    disabled={qcActionLoading || selectedFeature.properties?.qc_status === "rejected"}
                    style={{
                      backgroundColor: selectedFeature.properties?.qc_status === "rejected" ? "#f2f2f7" : "#ff3b30",
                      color: selectedFeature.properties?.qc_status === "rejected" ? "#86868b" : "#ffffff",
                      border: "none",
                      borderRadius: "9999px",
                      padding: "9px",
                      fontSize: "13px",
                      fontWeight: "500",
                      cursor: qcActionLoading || selectedFeature.properties?.qc_status === "rejected" ? "not-allowed" : "pointer",
                      transition: "all 0.15s ease",
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

// React Root Mount
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
