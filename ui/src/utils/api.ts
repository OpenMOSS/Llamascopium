import { Feature } from "@/types/feature";
import { buildBt4DictionaryFromAnalysisName } from "@/utils/bt4Sae";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";

const API_BASE = import.meta.env.VITE_BACKEND_URL || "";
const featureCache = new Map<string, Feature>();
const pendingFeatureRequests = new Map<string, Promise<Feature | null>>();

type FeatureFetchOptions = {
  forceRefresh?: boolean;
};

const parseFeatureResponse = async (response: Response): Promise<Feature> => {
  const arrayBuffer = await response.arrayBuffer();
  const decoded = decode(new Uint8Array(arrayBuffer)) as Record<string, unknown>;
  const camelCased = camelcaseKeys(decoded, {
    deep: true,
    stopPaths: ["context"],
  });

  return camelCased as Feature;
};

const fetchFeatureByResolvedDictionaryName = async (
  dictionaryName: string,
  featureId: number,
  options: FeatureFetchOptions = {},
): Promise<Feature | null> => {
  const cacheKey = `${dictionaryName}::${featureId}`;

  if (!options.forceRefresh) {
    const cached = featureCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    const pending = pendingFeatureRequests.get(cacheKey);
    if (pending) {
      return pending;
    }
  } else {
    featureCache.delete(cacheKey);
    pendingFeatureRequests.delete(cacheKey);
  }

  const request = (async () => {
    try {
      const response = await fetch(
        `${API_BASE}/dictionaries/${encodeURIComponent(dictionaryName)}/features/${featureId}`,
        {
          method: "GET",
          headers: {
            Accept: "application/x-msgpack",
          },
        },
      );

      if (!response.ok) {
        console.warn(`Failed to fetch feature from ${dictionaryName}: ${response.status} ${response.statusText}`);
        return null;
      }

      const feature = await parseFeatureResponse(response);
      featureCache.set(cacheKey, feature);
      return feature;
    } catch (error) {
      console.error(`Error fetching feature ${featureId} from dictionary ${dictionaryName}:`, error);
      return null;
    }
  })().finally(() => {
    pendingFeatureRequests.delete(cacheKey);
  });

  pendingFeatureRequests.set(cacheKey, request);
  return request;
};

export const fetchFeature = async (
  analysisName: string,
  layer: number,
  featureId: number,
  options: FeatureFetchOptions = {},
): Promise<Feature | null> => {
  const formattedAnalysisName = analysisName.replace("{}", layer.toString());
  return fetchFeatureByResolvedDictionaryName(formattedAnalysisName, featureId, options);
};

export const fetchFeatureByDictionaryName = async (
  dictionaryName: string,
  featureId: number,
  options: FeatureFetchOptions = {},
): Promise<Feature | null> => {
  return fetchFeatureByResolvedDictionaryName(dictionaryName, featureId, options);
};

/**
 * Get dictionary name from circuit JSON metadata.
 * Uses lorsa_analysis_name / tc_analysis_name to build full dictionary names.
 */
export const getDictionaryName = (metadata: any, layer: number, isLorsa: boolean): string => {
  if (isLorsa) {
    return buildBt4DictionaryFromAnalysisName(metadata?.lorsa_analysis_name, layer, "lorsa");
  } else {
    return buildBt4DictionaryFromAnalysisName(
      metadata?.tc_analysis_name || metadata?.clt_analysis_name,
      layer,
      "tc",
    );
  }
};

// Circuit Annotation API
export interface CircuitFeature {
  sae_name: string;
  sae_series: string;
  layer: number;
  feature_index: number;
  feature_type: "transcoder" | "lorsa";
  interpretation?: string;
  level?: number; // Circuit level (independent of layer, for visualization)
  feature_id?: string; // Unique identifier for this feature in the circuit
}

export interface CircuitEdge {
  source_feature_id: string;
  target_feature_id: string;
  weight: number;
  interpretation?: string;
}

export interface CircuitAnnotation {
  circuit_id: string;
  circuit_interpretation: string;
  sae_combo_id: string;
  features: CircuitFeature[];
  edges?: CircuitEdge[];
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

const circuitTaxonomyCircuitCache = new Map<string, CircuitTaxonomyCircuitDetail>();
const pendingCircuitTaxonomyCircuitRequests = new Map<
  string,
  Promise<CircuitTaxonomyCircuitDetail>
>();

export const createCircuitAnnotation = async (
  circuitInterpretation: string,
  saeComboId: string,
  features: CircuitFeature[],
  edges?: CircuitEdge[],
  metadata?: Record<string, any>
): Promise<CircuitAnnotation> => {
  const response = await fetch(`${API_BASE}/circuit_annotations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      circuit_interpretation: circuitInterpretation,
      sae_combo_id: saeComboId,
      features,
      edges,
      metadata,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to create circuit annotation: ${response.status}`);
  }

  return response.json();
};

export const getCircuitAnnotation = async (circuitId: string): Promise<CircuitAnnotation> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to get circuit annotation: ${response.status}`);
  }

  return response.json();
};

export const listCircuitAnnotations = async (
  saeComboId?: string,
  limit: number = 100,
  skip: number = 0
): Promise<{ circuits: CircuitAnnotation[]; total_count: number }> => {
  const params = new URLSearchParams();
  if (saeComboId) params.append("sae_combo_id", saeComboId);
  params.append("limit", limit.toString());
  params.append("skip", skip.toString());

  const response = await fetch(`${API_BASE}/circuit_annotations?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to list circuit annotations: ${response.status}`);
  }

  return response.json();
};

export const updateCircuitInterpretation = async (
  circuitId: string,
  circuitInterpretation: string
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/interpretation`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      circuit_interpretation: circuitInterpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to update circuit interpretation: ${response.status}`);
  }
};

export const addFeatureToCircuit = async (
  circuitId: string,
  feature: CircuitFeature
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/features`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(feature),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to add feature to circuit: ${response.status}`);
  }
};

export interface CircuitTaxonomyDirectoryOption {
  id: string;
  label: string;
  combo_id: string;
  root_id: string;
  file_count: number;
}

export interface CircuitTaxonomyCircuitSummary {
  file_name: string;
  index: number;
  prompt: string | null;
  target_move: string | null;
  predicted_move_uci: string | null;
  slug: string | null;
  feature_count: number;
  error?: string;
}

export interface CircuitTaxonomyFeatureRef {
  node_id: string;
  layer: number;
  feature_index: number;
  feature_type: "lorsa" | "transcoder";
  dictionary_name: string;
  ctx_idx: number | null;
  label: string;
}

export interface CircuitTaxonomyCircuitDetail {
  directory_id: string;
  file_name: string;
  circuit_index: number;
  total_circuits: number;
  total_features: number;
  first_unannotated_feature_index: number | null;
  features: CircuitTaxonomyFeatureRef[];
  graph_data: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface CircuitTaxonomyResumeTarget {
  directory_id: string;
  completed: boolean;
  file_name: string | null;
  circuit_index: number | null;
  feature_index: number | null;
}

export interface CircuitTaxonomyAnnotateResponse {
  status: "unchanged" | "conflict" | "updated";
  taxonomy: string;
  existing_taxonomy?: string | null;
  existing_text?: string;
  proposed_text?: string;
  overwritten?: boolean;
  interpretation?: Record<string, unknown> | null;
}

export interface CircuitTaxonomySaveDirectoryEvidenceResponse {
  status: "saved";
  directory_id: string;
  path: string;
  relative_path: string;
  item_count: number;
  file_count?: number;
  files?: Array<{
    file_name: string;
    item_count: number;
    path: string;
    relative_path: string;
  }>;
}

export interface CircuitTaxonomyBatchAnnotateResponse {
  status: "completed";
  item_count: number;
  counts: Record<string, number>;
  results: Array<Record<string, unknown>>;
}

export interface CircuitTaxonomyReviewStateResponse {
  status: "missing" | "loaded" | "saved" | "snapshot_saved";
  proposals?: unknown[];
  active_review_index: number;
  updated_at: string | null;
  relative_path?: string;
  proposal_count?: number;
  saved_at?: string;
  snapshot_path?: string;
  snapshot_relative_path?: string;
}

export const fetchCircuitTaxonomyDirectories = async (): Promise<{
  directories: CircuitTaxonomyDirectoryOption[];
  taxonomy_labels: string[];
}> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/directories`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const fetchCircuitTaxonomyCircuits = async (
  directoryId: string,
): Promise<{ directory_id: string; circuits: CircuitTaxonomyCircuitSummary[]; total_circuits: number }> => {
  const response = await fetch(
    `${API_BASE}/circuit_taxonomy/circuits?directory_id=${encodeURIComponent(directoryId)}`,
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const fetchCircuitTaxonomyCircuit = async (
  directoryId: string,
  fileName: string,
): Promise<CircuitTaxonomyCircuitDetail> => {
  const cacheKey = `${directoryId}::${fileName}`;
  const cached = circuitTaxonomyCircuitCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const pending = pendingCircuitTaxonomyCircuitRequests.get(cacheKey);
  if (pending) {
    return pending;
  }

  const request = (async () => {
    const response = await fetch(
      `${API_BASE}/circuit_taxonomy/circuit?directory_id=${encodeURIComponent(directoryId)}&file_name=${encodeURIComponent(fileName)}`,
    );
    if (!response.ok) {
      throw new Error(await response.text());
    }

    const detail = (await response.json()) as CircuitTaxonomyCircuitDetail;
    circuitTaxonomyCircuitCache.set(cacheKey, detail);
    return detail;
  })().finally(() => {
    pendingCircuitTaxonomyCircuitRequests.delete(cacheKey);
  });

  pendingCircuitTaxonomyCircuitRequests.set(cacheKey, request);
  return request;
};

export const fetchCircuitTaxonomyResumeTarget = async (
  directoryId: string,
  fileName?: string,
  startFeatureIndex?: number,
): Promise<CircuitTaxonomyResumeTarget> => {
  const params = new URLSearchParams();
  params.set("directory_id", directoryId);
  if (fileName) {
    params.set("file_name", fileName);
  }
  if (startFeatureIndex !== undefined) {
    params.set("start_feature_index", String(startFeatureIndex));
  }

  const response = await fetch(`${API_BASE}/circuit_taxonomy/resume?${params.toString()}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const buildCircuitTaxonomyEvidenceExportUrl = (
  directoryId: string,
  options: {
    fileName?: string;
    startFeatureIndex?: number;
    limit?: number;
    maxSamples?: number;
    topSquares?: number;
    topZ?: number;
    includeAnnotated?: boolean;
    singleFileOnly?: boolean;
  } = {},
): string => {
  const params = new URLSearchParams();
  params.set("directory_id", directoryId);
  if (options.fileName) {
    params.set("file_name", options.fileName);
  }
  if (options.startFeatureIndex !== undefined) {
    params.set("start_feature_index", String(options.startFeatureIndex));
  }
  if (options.limit !== undefined) {
    params.set("limit", String(options.limit));
  }
  if (options.maxSamples !== undefined) {
    params.set("max_samples", String(options.maxSamples));
  }
  if (options.topSquares !== undefined) {
    params.set("top_squares", String(options.topSquares));
  }
  if (options.topZ !== undefined) {
    params.set("top_z", String(options.topZ));
  }
  if (options.includeAnnotated !== undefined) {
    params.set("include_annotated", String(options.includeAnnotated));
  }
  if (options.singleFileOnly !== undefined) {
    params.set("single_file_only", String(options.singleFileOnly));
  }

  return `${API_BASE}/circuit_taxonomy/export_evidence?${params.toString()}`;
};

export const saveCircuitTaxonomyDirectoryEvidence = async (
  directoryId: string,
  options: {
    limit?: number;
    maxSamples?: number;
    topSquares?: number;
    topZ?: number;
    includeAnnotated?: boolean;
  } = {},
): Promise<CircuitTaxonomySaveDirectoryEvidenceResponse> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/save_directory_evidence`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      directory_id: directoryId,
      ...(options.limit === undefined ? {} : { limit: options.limit }),
      max_samples: options.maxSamples ?? 6,
      top_squares: options.topSquares ?? 8,
      top_z: options.topZ ?? 12,
      include_annotated: options.includeAnnotated ?? false,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const saveAllCircuitTaxonomyEvidence = async (
  directoryId: string,
  options: {
    maxSamples?: number;
    topSquares?: number;
    topZ?: number;
    includeAnnotated?: boolean;
  } = {},
): Promise<CircuitTaxonomySaveDirectoryEvidenceResponse> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/save_all_circuit_evidence`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      directory_id: directoryId,
      max_samples: options.maxSamples ?? 6,
      top_squares: options.topSquares ?? 8,
      top_z: options.topZ ?? 12,
      include_annotated: options.includeAnnotated ?? true,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const fetchCircuitTaxonomyReviewState = async (): Promise<CircuitTaxonomyReviewStateResponse> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/review_state`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const saveCircuitTaxonomyReviewState = async (
  proposals: unknown[],
  activeReviewIndex: number,
): Promise<CircuitTaxonomyReviewStateResponse> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/review_state`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      proposals,
      active_review_index: activeReviewIndex,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const snapshotCircuitTaxonomyReviewState = async (
  proposals: unknown[],
  activeReviewIndex: number,
): Promise<CircuitTaxonomyReviewStateResponse> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/review_state/snapshot`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      proposals,
      active_review_index: activeReviewIndex,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const annotateCircuitTaxonomyFeature = async (
  dictionaryName: string,
  featureIndex: number,
  taxonomy: string,
  overwrite: boolean = false,
): Promise<CircuitTaxonomyAnnotateResponse> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/annotate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      dictionary_name: dictionaryName,
      feature_index: featureIndex,
      taxonomy,
      overwrite,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const annotateCircuitTaxonomyFeatures = async (
  items: Array<{
    dictionary_name?: string;
    dictionaryName?: string;
    feature_index?: number;
    featureIndex?: number;
    taxonomy: string;
  }>,
  overwrite: boolean = true,
): Promise<CircuitTaxonomyBatchAnnotateResponse> => {
  const response = await fetch(`${API_BASE}/circuit_taxonomy/annotate_batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      items,
      overwrite,
    }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

export const removeFeatureFromCircuit = async (
  circuitId: string,
  saeName: string,
  saeSeries: string,
  layer: number,
  featureIndex: number,
  featureType: "transcoder" | "lorsa"
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/features`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sae_name: saeName,
      sae_series: saeSeries,
      layer,
      feature_index: featureIndex,
      feature_type: featureType,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to remove feature from circuit: ${response.status}`);
  }
};

export const updateFeatureInterpretationInCircuit = async (
  circuitId: string,
  saeName: string,
  saeSeries: string,
  layer: number,
  featureIndex: number,
  featureType: "transcoder" | "lorsa",
  interpretation: string
): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/features/interpretation`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sae_name: saeName,
      sae_series: saeSeries,
      layer,
      feature_index: featureIndex,
      feature_type: featureType,
      interpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to update feature interpretation: ${response.status}`);
  }
};

export const getCircuitsByFeature = async (
  saeName: string,
  saeSeries: string,
  layer: number,
  featureIndex: number,
  featureType?: "transcoder" | "lorsa"
): Promise<{ circuits: CircuitAnnotation[] }> => {
  const params = new URLSearchParams();
  params.append("sae_name", saeName);
  params.append("sae_series", saeSeries);
  params.append("layer", layer.toString());
  params.append("feature_index", featureIndex.toString());
  if (featureType) params.append("feature_type", featureType);

  const response = await fetch(`${API_BASE}/circuit_annotations/by_feature?${params}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to get circuits by feature: ${response.status}`);
  }

  return response.json();
};

export const deleteCircuitAnnotation = async (circuitId: string): Promise<void> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to delete circuit annotation: ${response.status}`);
  }
};

// Edge management APIs
export const addEdgeToCircuit = async (
  circuitId: string,
  sourceFeatureId: string,
  targetFeatureId: string,
  weight: number = 0.0,
  interpretation?: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/edges`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_feature_id: sourceFeatureId,
      target_feature_id: targetFeatureId,
      weight,
      interpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to add edge: ${response.status}`);
  }

  return response.json();
};

export const removeEdgeFromCircuit = async (
  circuitId: string,
  sourceFeatureId: string,
  targetFeatureId: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/edges`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_feature_id: sourceFeatureId,
      target_feature_id: targetFeatureId,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to remove edge: ${response.status}`);
  }

  return response.json();
};

export const updateEdgeWeight = async (
  circuitId: string,
  sourceFeatureId: string,
  targetFeatureId: string,
  weight: number,
  interpretation?: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE}/circuit_annotations/${circuitId}/edges`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source_feature_id: sourceFeatureId,
      target_feature_id: targetFeatureId,
      weight,
      interpretation,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to update edge weight: ${response.status}`);
  }

  return response.json();
};

export const setFeatureLevel = async (
  circuitId: string,
  featureId: string,
  level: number
): Promise<{ message: string }> => {
  const response = await fetch(
    `${API_BASE}/circuit_annotations/${circuitId}/features/${featureId}/level`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        level,
      }),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Failed to set feature level: ${response.status}`);
  }

  return response.json();
}; 
