import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import {
  MapContainer,
  TileLayer,
  GeoJSON,
  ImageOverlay,
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

type BuildingStats = {
  job_id: string;
  pixels: number;
  coverage: number;
  prediction_min?: number;
  prediction_max?: number;
  prediction_mean?: number;
};

type RoadStats = {
  job_id: string;
  pixels: number;
  coverage: number;
  prediction_min?: number;
  prediction_max?: number;
  prediction_mean?: number;
};

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


  function collectCoordinates(
    value: any
  ): void {

    if (!Array.isArray(value)) {
      return;
    }

    if (
      value.length >= 2 &&
      typeof value[0] === "number" &&
      typeof value[1] === "number"
    ) {
      coordinates.push([
        value[0],
        value[1],
      ]);

      return;
    }

    for (const item of value) {
      collectCoordinates(item);
    }
  }


  function collectGeometry(
    geometry: any
  ): void {

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
      collectCoordinates(
        geometry.coordinates
      );
    }
  }


  if (
    geojson?.type === "FeatureCollection" &&
    Array.isArray(geojson.features)
  ) {

    for (const feature of geojson.features) {
      collectGeometry(
        feature?.geometry
      );
    }

  } else if (
    geojson?.type === "Feature"
  ) {

    collectGeometry(
      geojson.geometry
    );

  } else {

    collectGeometry(
      geojson
    );
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

    minLon = Math.min(
      minLon,
      lon
    );

    maxLon = Math.max(
      maxLon,
      lon
    );

    minLat = Math.min(
      minLat,
      lat
    );

    maxLat = Math.max(
      maxLat,
      lat
    );
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

  // ==========================================================
  // File / Field state
  // ==========================================================

  const [file, setFile] =
    useState<File | null>(null);

  const [geojson, setGeojson] =
    useState<GeoJsonObject | null>(null);

  const [fieldJobId, setFieldJobId] =
    useState<string | null>(null);

  const [existingJobId, setExistingJobId] =
    useState("da8807fb");

  const [fieldCount, setFieldCount] =
    useState(0);


  // ==========================================================
  // Building state
  // ==========================================================

  const [buildingMaskUrl, setBuildingMaskUrl] =
    useState<string | null>(null);

  const [buildingStats, setBuildingStats] =
    useState<BuildingStats | null>(null);


  // ==========================================================
  // Road state
  // ==========================================================

  const [roadMaskUrl, setRoadMaskUrl] =
    useState<string | null>(null);

  const [roadStats, setRoadStats] =
    useState<RoadStats | null>(null);


  // ==========================================================
  // General UI state
  // ==========================================================

  const [loading, setLoading] =
    useState(false);

  const [activeOperation, setActiveOperation] =
    useState("");

  const [message, setMessage] =
    useState(
      "Select an orthophoto to begin."
    );

  const [mapBounds, setMapBounds] =
    useState<LatLngBoundsExpression | null>(
      null
    );


  // ==========================================================
  // PostGIS layer state
  // ==========================================================

  const [layers, setLayers] =
    useState<PostGISLayer[]>([]);

  const [selectedLayer, setSelectedLayer] =
    useState("");

  const [layerGeoJSON, setLayerGeoJSON] =
    useState<GeoJsonObject | null>(null);

  const [layerLoading, setLayerLoading] =
    useState(false);


  // ==========================================================
  // File selection
  // ==========================================================

  function handleFileChange(
    event: React.ChangeEvent<HTMLInputElement>
  ) {

    const selectedFile =
      event.target.files?.[0] ?? null;

    setFile(selectedFile);

    setGeojson(null);

    setBuildingMaskUrl(null);

    setRoadMaskUrl(null);

    setBuildingStats(null);

    setRoadStats(null);

    setFieldJobId(null);

    setFieldCount(0);

    setLayerGeoJSON(null);

    setSelectedLayer("");

    setMapBounds(null);


    if (selectedFile) {

      setMessage(
        `Selected file: ${selectedFile.name}`
      );

    } else {

      setMessage(
        "Select an orthophoto to begin."
      );
    }
  }


  // ============================================================
  // Load available PostGIS layers
  // ============================================================

  async function loadLayers() {

    try {

      const response =
        await fetch(
          `${API_URL}/layers`
        );

      const data =
        await response.json();

      if (!response.ok) {

        throw new Error(
          data?.detail ||
          "Could not load PostGIS layers."
        );
      }

      setLayers(
        data.layers || []
      );

    } catch (error) {

      console.error(
        "Could not load layers:",
        error
      );
    }
  }


  // ============================================================
  // Load selected PostGIS layer
  // ============================================================

  async function loadPostGISLayer(
    layerName: string
  ) {

    if (!layerName) {

      setLayerGeoJSON(null);

      return;
    }


    setLayerLoading(true);


    try {

      const response =
        await fetch(
          `${API_URL}/layers/${layerName}`
        );

      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data?.detail ||
          "Could not load selected layer."
        );
      }


      setLayerGeoJSON(
        data
      );


      const bounds =
        calculateGeoJSONBounds(
          data
        );


      if (bounds) {

        setMapBounds(
          bounds
        );
      }


      setMessage(
        `${layerName} loaded successfully.`
      );

    } catch (error) {

      console.error(error);

      setMessage(
        error instanceof Error
          ? error.message
          : "Could not load PostGIS layer."
      );

    } finally {

      setLayerLoading(false);
    }
  }


  // ============================================================
  // Load PostGIS layers when frontend starts
  // ============================================================

  useEffect(() => {

    loadLayers();

  }, []);


  // ============================================================
  // Load existing field result
  // ============================================================

  async function loadExistingResult() {

    if (!existingJobId.trim()) {

      setMessage(
        "Please enter a Job ID."
      );

      return;
    }


    setLoading(true);

    setActiveOperation(
      "Loading field result"
    );

    setGeojson(null);

    setFieldCount(0);


    try {

      setMessage(
        "Loading existing field result..."
      );


      const response =
        await fetch(
          `${API_URL}/inference/fields/${existingJobId.trim()}/result`
        );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data?.detail ||
          "Could not load field result."
        );
      }


      setGeojson(
        data
      );


      setFieldJobId(
        existingJobId.trim()
      );


      const count =
        Array.isArray(data.features)
          ? data.features.length
          : 0;


      setFieldCount(
        count
      );


      const bounds =
        calculateGeoJSONBounds(
          data
        );


      if (bounds) {

        setMapBounds(
          bounds
        );
      }


      setMessage(
        `Field result loaded successfully. ${count} field(s) detected.`
      );

    } catch (error) {

      console.error(error);

      setMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong."
      );

    } finally {

      setLoading(false);

      setActiveOperation("");
    }
  }


  // ============================================================
  // Field inference
  // ============================================================

  async function runFieldInference() {

    if (!file) {

      setMessage(
        "Please select a .tif or .tiff file first."
      );

      return;
    }


    setLoading(true);

    setActiveOperation(
      "Running field detection"
    );

    setGeojson(null);

    setFieldJobId(null);

    setFieldCount(0);


    try {

      setMessage(
        "Uploading orthophoto and running AI field detection..."
      );


      const formData =
        new FormData();


      formData.append(
        "file",
        file
      );


      const inferenceResponse =
        await fetch(
          `${API_URL}/inference/fields`,
          {
            method: "POST",
            body: formData,
          }
        );


      const inferenceData =
        await inferenceResponse.json();


      if (!inferenceResponse.ok) {

        throw new Error(
          inferenceData?.detail?.message ||
          inferenceData?.detail ||
          "Field inference failed."
        );
      }


      const newJobId =
        inferenceData.job_id;


      setFieldJobId(
        newJobId
      );


      setMessage(
        `Field AI completed. Loading result for job ${newJobId}...`
      );


      const resultResponse =
        await fetch(
          `${API_URL}/inference/fields/${newJobId}/result`
        );


      const resultData =
        await resultResponse.json();


      if (!resultResponse.ok) {

        throw new Error(
          resultData?.detail ||
          "Could not load field result."
        );
      }


      setGeojson(
        resultData
      );


      const count =
        Array.isArray(
          resultData.features
        )
          ? resultData.features.length
          : 0;


      setFieldCount(
        count
      );


      const bounds =
        calculateGeoJSONBounds(
          resultData
        );


      if (bounds) {

        setMapBounds(
          bounds
        );
      }


      setMessage(
        `Field detection completed successfully. ${count} field(s) detected.`
      );

    } catch (error) {

      console.error(error);

      setMessage(
        error instanceof Error
          ? error.message
          : "Field detection failed."
      );

    } finally {

      setLoading(false);

      setActiveOperation("");
    }
  }


  // ============================================================
  // Building inference
  // ============================================================

  async function runBuildingInference() {

    if (!file) {

      setMessage(
        "Please select a .tif or .tiff file first."
      );

      return;
    }


    setLoading(true);

    setActiveOperation(
      "Running building detection"
    );

    setBuildingMaskUrl(null);

    setBuildingStats(null);


    try {

      setMessage(
        "Uploading orthophoto and running Building U-Net..."
      );


      const formData =
        new FormData();


      formData.append(
        "file",
        file
      );


      const response =
        await fetch(
          `${API_URL}/inference/buildings`,
          {
            method: "POST",
            body: formData,
          }
        );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data?.detail?.message ||
          data?.detail ||
          "Building inference failed."
        );
      }


      const jobId =
        data.job_id;


      const statistics =
        data.statistics;


      setBuildingStats({

        job_id: jobId,

        pixels:
          statistics?.building_pixels ?? 0,

        coverage:
          statistics?.building_coverage_percent ?? 0,

        prediction_min:
          statistics?.prediction_min,

        prediction_max:
          statistics?.prediction_max,

        prediction_mean:
          statistics?.prediction_mean,
      });


      setBuildingMaskUrl(
        `${API_URL}/inference/buildings/${jobId}/mask`
      );


      setMessage(
        `Building detection completed. ${statistics?.building_coverage_percent ?? 0}% building coverage detected.`
      );

    } catch (error) {

      console.error(error);

      setMessage(
        error instanceof Error
          ? error.message
          : "Building detection failed."
      );

    } finally {

      setLoading(false);

      setActiveOperation("");
    }
  }


  // ============================================================
  // Road inference
  // ============================================================

  async function runRoadInference() {

    if (!file) {

      setMessage(
        "Please select a .tif or .tiff file first."
      );

      return;
    }


    setLoading(true);

    setActiveOperation(
      "Running road detection"
    );

    setRoadMaskUrl(null);

    setRoadStats(null);


    try {

      setMessage(
        "Uploading orthophoto and running Road DLinkNet34..."
      );


      const formData =
        new FormData();


      formData.append(
        "file",
        file
      );


      const response =
        await fetch(
          `${API_URL}/inference/roads`,
          {
            method: "POST",
            body: formData,
          }
        );


      const data =
        await response.json();


      if (!response.ok) {

        throw new Error(
          data?.detail?.message ||
          data?.detail ||
          "Road inference failed."
        );
      }


      const jobId =
        data.job_id;


      const statistics =
        data.statistics;


      setRoadStats({

        job_id: jobId,

        pixels:
          statistics?.road_pixels ?? 0,

        coverage:
          statistics?.road_coverage_percent ?? 0,

        prediction_min:
          statistics?.prediction_min,

        prediction_max:
          statistics?.prediction_max,

        prediction_mean:
          statistics?.prediction_mean,
      });


      setRoadMaskUrl(
        `${API_URL}/inference/roads/${jobId}/mask`
      );


      setMessage(
        `Road detection completed. ${statistics?.road_coverage_percent ?? 0}% road coverage detected.`
      );

    } catch (error) {

      console.error(error);

      setMessage(
        error instanceof Error
          ? error.message
          : "Road detection failed."
      );

    } finally {

      setLoading(false);

      setActiveOperation("");
    }
  }


  // ============================================================
  // Shared button style
  // ============================================================

  function buttonStyle(
    enabled: boolean,
    background: string
  ): React.CSSProperties {

    return {

      background:
        enabled
          ? background
          : "#9aa9a4",

      color: "white",

      border: "none",

      borderRadius: "8px",

      padding: "11px 18px",

      cursor:
        enabled
          ? "pointer"
          : "not-allowed",

      fontSize: "14px",

      marginRight: "10px",

      marginBottom: "10px",
    };
  }


  // ============================================================
  // UI
  // ============================================================

  return (

    <div
      style={{
        minHeight: "100vh",
        background: "#f4f7f6",
        fontFamily:
          "Arial, sans-serif",
      }}
    >

      {/* ======================================================
          Header
      ====================================================== */}

      <header
        style={{
          background: "#173f35",
          color: "white",
          padding: "18px 32px",
        }}
      >

        <h1
          style={{
            margin: 0,
            fontSize: "28px",
          }}
        >
          Naksha
        </h1>

        <p
          style={{
            margin: "6px 0 0",
            opacity: 0.85,
          }}
        >
          AI-Powered Orthophoto Digitization Platform
        </p>

      </header>


      {/* ======================================================
          Main
      ====================================================== */}

      <main
        style={{
          padding: "24px",
          maxWidth: "1400px",
          margin: "0 auto",
        }}
      >

        {/* ====================================================
            Upload / AI Analysis Panel
        ==================================================== */}

        <section
          style={{
            background: "white",
            borderRadius: "12px",
            padding: "22px",
            marginBottom: "20px",
            boxShadow:
              "0 2px 10px rgba(0,0,0,0.08)",
          }}
        >

          <h2
            style={{
              marginTop: 0,
            }}
          >
            Orthophoto AI Analysis
          </h2>

          <p
            style={{
              color: "#555",
            }}
          >
            Upload a GeoTIFF orthophoto and run
            AI-based field, building, and road
            detection.
          </p>


          {/* File upload */}

          <input
            type="file"
            accept=".tif,.tiff"
            onChange={handleFileChange}
            style={{
              marginBottom: "15px",
            }}
          />


          {file && (

            <div
              style={{
                marginBottom: "18px",
                padding: "10px 12px",
                background: "#eef7f3",
                borderRadius: "7px",
                color: "#285f4d",
              }}
            >

              <strong>
                Selected file:
              </strong>{" "}

              {file.name}

            </div>

          )}


          {/* Detection buttons */}

          <div
            style={{
              marginTop: "10px",
            }}
          >

            <button
              onClick={
                runFieldInference
              }
              disabled={
                !file ||
                loading
              }
              style={buttonStyle(
                Boolean(file) &&
                  !loading,
                "#1f6f5b"
              )}
            >

              {loading &&
              activeOperation ===
                "Running field detection"
                ? "Running Field AI..."
                : "Detect Fields"}

            </button>


            <button
              onClick={
                runBuildingInference
              }
              disabled={
                !file ||
                loading
              }
              style={buttonStyle(
                Boolean(file) &&
                  !loading,
                "#8a5a00"
              )}
            >

              {loading &&
              activeOperation ===
                "Running building detection"
                ? "Running Building AI..."
                : "Detect Buildings"}

            </button>


            <button
              onClick={
                runRoadInference
              }
              disabled={
                !file ||
                loading
              }
              style={buttonStyle(
                Boolean(file) &&
                  !loading,
                "#285f8f"
              )}
            >

              {loading &&
              activeOperation ===
                "Running road detection"
                ? "Running Road AI..."
                : "Detect Roads"}

            </button>

          </div>


          {/* Status */}

          <div
            style={{
              marginTop: "12px",
              padding: "12px",
              background: "#f5f5f5",
              borderRadius: "8px",
              color: "#444",
            }}
          >
            {message}
          </div>


          {/* Field statistics */}

          {fieldJobId && (

            <div
              style={{
                marginTop: "12px",
                padding: "12px",
                background: "#eef7f3",
                borderRadius: "8px",
              }}
            >

              <strong>
                Field Job ID:
              </strong>{" "}

              {fieldJobId}

              <br />

              <strong>
                Detected Fields:
              </strong>{" "}

              {fieldCount}

            </div>

          )}


          {/* Building statistics */}

          {buildingStats && (

            <div
              style={{
                marginTop: "12px",
                padding: "12px",
                background: "#fff7e6",
                borderRadius: "8px",
              }}
            >

              <strong>
                Building Detection
              </strong>

              <br />

              Job ID:{" "}
              {buildingStats.job_id}

              <br />

              Building Pixels:{" "}
              {buildingStats.pixels.toLocaleString()}

              <br />

              Building Coverage:{" "}
              {buildingStats.coverage}%

              {buildingStats.prediction_max !==
                undefined && (

                <>
                  <br />

                  Prediction Max:{" "}
                  {buildingStats.prediction_max.toFixed(
                    4
                  )}
                </>

              )}

            </div>

          )}


          {/* Road statistics */}

          {roadStats && (

            <div
              style={{
                marginTop: "12px",
                padding: "12px",
                background: "#eaf3fb",
                borderRadius: "8px",
              }}
            >

              <strong>
                Road Detection
              </strong>

              <br />

              Job ID:{" "}
              {roadStats.job_id}

              <br />

              Road Pixels:{" "}
              {roadStats.pixels.toLocaleString()}

              <br />

              Road Coverage:{" "}
              {roadStats.coverage}%

              {roadStats.prediction_max !==
                undefined && (

                <>
                  <br />

                  Prediction Max:{" "}
                  {roadStats.prediction_max.toFixed(
                    4
                  )}
                </>

              )}

            </div>

          )}

        </section>


        {/* ====================================================
            Existing Field Result
        ==================================================== */}

        <section
          style={{
            background: "white",
            borderRadius: "12px",
            padding: "22px",
            marginBottom: "20px",
            boxShadow:
              "0 2px 10px rgba(0,0,0,0.08)",
          }}
        >

          <h3
            style={{
              marginTop: 0,
            }}
          >
            Load Existing Field Result
          </h3>

          <p
            style={{
              color: "#666",
              fontSize: "14px",
            }}
          >
            Enter an existing Field Job ID to
            display previously generated field
            boundaries.
          </p>


          <input
            type="text"
            value={existingJobId}
            onChange={(event) =>
              setExistingJobId(
                event.target.value
              )
            }
            placeholder="Enter Field Job ID"
            style={{
              padding: "10px",
              width: "220px",
              border:
                "1px solid #ccc",
              borderRadius: "6px",
              marginRight: "10px",
            }}
          />


          <button
            onClick={
              loadExistingResult
            }
            disabled={loading}
            style={buttonStyle(
              !loading,
              "#285f8f"
            )}
          >

            {loading &&
            activeOperation ===
              "Loading field result"
              ? "Loading..."
              : "Load Field Result"}

          </button>

        </section>


        {/* ====================================================
            Map
        ==================================================== */}

        <section
          style={{
            background: "white",
            borderRadius: "12px",
            padding: "12px",
            boxShadow:
              "0 2px 10px rgba(0,0,0,0.08)",
          }}
        >

          <h2
            style={{
              paddingLeft: "10px",
              marginTop: "8px",
            }}
          >
            AI Detection Map
          </h2>


          <p
            style={{
              paddingLeft: "10px",
              color: "#666",
              fontSize: "14px",
            }}
          >
            Field boundaries, building masks,
            road masks, and PostGIS layers are
            displayed here.
          </p>


          {/* ==================================================
              PostGIS Layer Control
          ================================================== */}

          <div
            style={{
              margin: "10px",
              padding: "14px",
              background: "#f4f7f6",
              borderRadius: "8px",
              border:
                "1px solid #d8e2de",
            }}
          >

            <strong>
              PostGIS Layers
            </strong>


            <div
              style={{
                marginTop: "10px",
              }}
            >

              <select
                value={selectedLayer}
                onChange={(event) => {
                  const value = event.target.value;
                  setSelectedLayer(value);
                  loadPostGISLayer(value);
                }}
                style={{
                  padding: "9px 12px",
                  borderRadius: "6px",
                  border: "1px solid #ccc",
                  minWidth: "260px",
                }}
              >
                <option value="" disabled>
                  Select a PostGIS layer
                </option>

                {layers.map((layer) => (
                  <option
                    key={`${layer.layer_name}-${layer.version}-${layer.id}`}
                    value={layer.layer_name}
                  >
                    {layer.feature_type} (v{layer.version}, {layer.feature_count} features)
                  </option>
                ))}
              </select>


              {layerLoading && (

                <span
                  style={{
                    marginLeft: "12px",
                    color: "#666",
                  }}
                >
                  Loading layer...
                </span>

              )}

            </div>

          </div>


          {/* ==================================================
              Leaflet Map
          ================================================== */}

          <MapContainer
            center={
              mapBounds
                ? undefined
                : [
                    46.6578,
                    16.1166,
                  ]
            }
            zoom={17}
            bounds={
              mapBounds
                ? mapBounds
                : undefined
            }
            style={{
              height: "650px",
              width: "100%",
              borderRadius: "8px",
            }}
          >

            <TileLayer
              attribution=
                '&copy; OpenStreetMap contributors'
              url=
                "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
<TileLayer
  url={
    "http" +
    "://127.0.0.1:8080/cog/tiles/WebMercatorQuad/{z}/{x}/{y}" +
    "?url=%2Fdata%2Fraw%2Fdemo_aoi%2Fdemo_aoi_cog.tif" +
    "&tilesize=512"
  }
  minZoom={17}
  maxZoom={22}
  opacity={0.85}
/>


            {/* ================================================
                Field GeoJSON
            ================================================= */}

            {geojson && (

              <GeoJSON
                data={
                  geojson
                }
                style={() => ({
                  color: "#00a86b",
                  weight: 3,
                  fillColor:
                    "#38b879",
                  fillOpacity: 0.20,
                })}
              />

            )}


            {/* ================================================
                PostGIS GeoJSON
            ================================================= */}

            {layerGeoJSON && (

              <GeoJSON
                key={
                  selectedLayer
                }
                data={
                  layerGeoJSON
                }
                style={() => ({
                  color: "#7b2cbf",
                  weight: 2,
                  fillColor:
                    "#9d4edd",
                  fillOpacity: 0.25,
                })}
                onEachFeature={(
                  feature,
                  layer
                ) => {

                  const properties =
                    feature.properties ||
                    {};

                  layer.bindPopup(
                    `
                    <strong>
                      ${
                        properties.feature_type ||
                        "Feature"
                      }
                    </strong>
                    <br />
                    Confidence:
                    ${
                      properties.confidence ??
                      "N/A"
                    }
                    `
                  );
                }}
              />

            )}


            {/* ================================================
                Building mask
            ================================================= */}

            {buildingMaskUrl &&
              mapBounds && (

                <ImageOverlay
                  url={
                    buildingMaskUrl
                  }
                  bounds={
                    mapBounds
                  }
                  opacity={0.65}
                  zIndex={20}
                />

              )}


            {/* ================================================
                Road mask
            ================================================= */}

            {roadMaskUrl &&
              mapBounds && (

                <ImageOverlay
                  url={
                    roadMaskUrl
                  }
                  bounds={
                    mapBounds
                  }
                  opacity={0.70}
                  zIndex={30}
                />

              )}

          </MapContainer>

        </section>

      </main>

    </div>
  );
}


// ============================================================
// React root
// ============================================================

createRoot(
  document.getElementById("root")!
).render(

  <React.StrictMode>
    <App />
  </React.StrictMode>

);

