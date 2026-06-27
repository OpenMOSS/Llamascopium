/**
 * Hook for circuit-related backend API calls.
 * Centralizes fetch logic for analyze_fen_all_positions and related endpoints.
 */

import { useCallback } from "react";
import { NodeActivationData, normalizeZPattern } from "@/utils/activationUtils";

const getBackendUrl = (): string => import.meta.env.VITE_BACKEND_URL;

export interface UseCircuitBackendOptions {
  /** Called when loading state changes (e.g. setLoadingAllPositions) */
  setLoadingAllPositions?: (loading: boolean) => void;
  /** Graph data to resolve node metadata (nodeType, clerp) for the result. May be null when no graph loaded. */
  linkGraphData?: { nodes?: Array<{ nodeId: string; feature_type?: string; clerp?: string }> } | null;
}

const appendZPatternPairs = (
  targetIndices: number[][],
  targetValues: number[],
  indices?: number[][],
  values?: number[]
) => {
  if (!indices || !values) return;

  const looksLikePairList = Array.isArray(indices[0]) && indices[0].length === 2;
  if (looksLikePairList) {
    for (let i = 0; i < Math.min(indices.length, values.length); i++) {
      const pair = indices[i];
      if (!Array.isArray(pair) || pair.length < 2) continue;
      targetIndices.push([Number(pair[0]), Number(pair[1])]);
      targetValues.push(Number(values[i]) || 0);
    }
    return;
  }

  if (indices.length >= 2 && Array.isArray(indices[0]) && Array.isArray(indices[1])) {
    const sources = indices[0];
    const targets = indices[1];
    for (let i = 0; i < Math.min(sources.length, targets.length, values.length); i++) {
      targetIndices.push([Number(sources[i]), Number(targets[i])]);
      targetValues.push(Number(values[i]) || 0);
    }
  }
};

/**
 * Fetch activation data for all positions from backend.
 * Merges activations across positions by taking max absolute value per cell.
 * Also keeps z_pattern pairs from every query position so hover can show
 * source -> target connections in all-positions mode.
 */
export const fetchAllPositionsFromBackend = async (
  dictionary: string,
  featureIndex: number,
  fen: string,
  nodeMetadata?: { nodeType?: string; clerp?: string }
): Promise<NodeActivationData | null> => {
  try {
    const response = await fetch(
      `${getBackendUrl()}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen_all_positions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ fen: fen.trim() }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const data = await response.json();

    const mergedActivations = new Array(64).fill(0);
    const mergedZPatternIndices: number[][] = [];
    const mergedZPatternValues: number[] = [];
    if (data.positions && Array.isArray(data.positions)) {
      for (const posData of data.positions) {
        if (posData.activations && Array.isArray(posData.activations) && posData.activations.length === 64) {
          for (let i = 0; i < 64; i++) {
            const newValue = posData.activations[i];
            if (Math.abs(newValue) > Math.abs(mergedActivations[i])) {
              mergedActivations[i] = newValue;
            }
          }
        }

        const { zPatternIndices, zPatternValues } = normalizeZPattern(
          posData.z_pattern_indices ?? posData.zPatternIndices,
          posData.z_pattern_values ?? posData.zPatternValues
        );
        appendZPatternPairs(mergedZPatternIndices, mergedZPatternValues, zPatternIndices, zPatternValues);
      }
    }

    return {
      activations: mergedActivations,
      zPatternIndices: mergedZPatternIndices.length > 0 ? mergedZPatternIndices : undefined,
      zPatternValues: mergedZPatternValues.length > 0 ? mergedZPatternValues : undefined,
      nodeType: nodeMetadata?.nodeType,
      clerp: nodeMetadata?.clerp,
    };
  } catch (error) {
    console.error("fetchAllPositionsFromBackend failed:", error);
    return null;
  }
};

export interface ZPatternResult {
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
}

const parseAnalyzeFenResponse = (data: any): ZPatternResult => {
  let activations: number[] | undefined = undefined;
  if (Array.isArray(data?.feature_acts_indices) && Array.isArray(data?.feature_acts_values)) {
    activations = new Array(64).fill(0);
    const indices = data.feature_acts_indices;
    const values = data.feature_acts_values;
    for (let i = 0; i < Math.min(indices.length, values.length); i++) {
      const idx = Number(indices[i]);
      const value = Number(values[i]) || 0;
      if (idx >= 0 && idx < 64) activations[idx] = value;
    }
  }

  const { zPatternIndices, zPatternValues } = normalizeZPattern(
    data?.z_pattern_indices ?? data?.zPatternIndices,
    data?.z_pattern_values ?? data?.zPatternValues
  );

  return { activations, zPatternIndices, zPatternValues };
};

/**
 * Fetch current-FEN feature activations for single-position mode.
 * This works for both Lorsa and Transcoder. Lorsa responses may include
 * z_pattern, while Transcoder responses only include activations.
 */
export const fetchFeatureActivationFromBackend = async (
  dictionary: string,
  featureIndex: number,
  fen: string,
  signal?: AbortSignal
): Promise<ZPatternResult | null> => {
  try {
    const response = await fetch(
      `${getBackendUrl()}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ fen: fen.trim() }),
        signal,
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    return parseAnalyzeFenResponse(await response.json());
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return null;
    console.error("fetchFeatureActivationFromBackend failed:", error);
    return null;
  }
};

/**
 * Fetch z_pattern for a Lorsa feature at a specific query position (single-position display).
 * Calls analyze_fen_all_positions and extracts z_pattern for the given queryPos.
 */
export const fetchZPatternForPosFromBackend = async (
  dictionary: string,
  featureIndex: number,
  fen: string,
  queryPos: number,
  signal?: AbortSignal
): Promise<ZPatternResult | null> => {
  try {
    const response = await fetch(
      `${getBackendUrl()}/dictionaries/${dictionary}/features/${featureIndex}/analyze_fen_all_positions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ fen: fen.trim() }),
        signal,
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const positions = data?.positions;
    if (!Array.isArray(positions)) return null;

    const posData = positions.find((p: { position?: number }) => Number(p?.position) === queryPos);
    if (!posData) return null;

    const activations = Array.isArray(posData.activations) && posData.activations.length === 64
      ? posData.activations.map((value: unknown) => Number(value) || 0)
      : undefined;
    const { zPatternIndices, zPatternValues } = normalizeZPattern(
      (posData as { z_pattern_indices?: unknown }).z_pattern_indices,
      (posData as { z_pattern_values?: unknown }).z_pattern_values
    );

    return { activations, zPatternIndices, zPatternValues };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") return null;
    console.error("fetchZPatternForPosFromBackend failed:", error);
    return null;
  }
};

/**
 * Hook that provides fetchAllPositionsFromBackend with loading state and node metadata.
 */
export const useCircuitBackend = (options: UseCircuitBackendOptions = {}) => {
  const { setLoadingAllPositions, linkGraphData } = options;

  const fetchAllPositions = useCallback(
    async (
      nodeId: string,
      fen: string,
      dictionary: string,
      featureIndex: number
    ): Promise<NodeActivationData | null> => {
      setLoadingAllPositions?.(true);
      try {
        const currentNode = linkGraphData?.nodes?.find((n: any) => n.nodeId === nodeId);
        const nodeMetadata = currentNode
          ? { nodeType: currentNode.feature_type, clerp: (currentNode as any)?.clerp }
          : undefined;

        return await fetchAllPositionsFromBackend(dictionary, featureIndex, fen, nodeMetadata);
      } finally {
        setLoadingAllPositions?.(false);
      }
    },
    [setLoadingAllPositions, linkGraphData]
  );

  return {
    fetchAllPositionsFromBackend: fetchAllPositions,
    fetchFeatureActivationFromBackend,
    fetchZPatternForPosFromBackend,
  };
};
