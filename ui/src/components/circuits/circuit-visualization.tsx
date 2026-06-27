import { useState, useCallback, useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import { useCircuitState } from "@/contexts/AppStateContext";
import { LinkGraphContainer } from "./link-graph-container";
import { NodeConnections } from "./node-connections";
import { CircuitJsonData } from "./link-graph/utils";
import { Node } from "./link-graph/types";
import { Feature } from "@/types/feature";
import { FeatureCard } from "@/components/feature/feature-card";
import { ChessBoard } from "@/components/chess/chess-board";
import React from "react";
import { SaeComboLoader } from "@/components/common/SaeComboLoader";
import { CustomFenInput } from "@/components/feature/custom-fen-input";
import { PosFeatureCard } from "@/components/feature/pos-feature-card";
import { CircuitInterpretationCard } from "./circuit-interpretation-card";
import { useCircuitFileUpload } from "@/hooks/useCircuitFileUpload";
import { FileUploadZone } from "./FileUploadZone";
import { useActivationData } from "@/hooks/useActivationData";
import { useFenExtraction } from "@/hooks/useFenExtraction";
import { useDictionaryName } from "@/hooks/useDictionaryName";
import { useCircuitStateReducer } from "@/hooks/useCircuitStateReducer";
import { useCircuitBackend } from "@/hooks/useCircuitBackend";
import { UNIQUE_GRAPH_COLORS, POSITION_MAPPING_HIGHLIGHT_COLOR } from "@/utils/graphMergeUtils";
import { parseNodeId, normalizeZPattern, getAllPositionsActivationDataFromJson, findNodesArray } from "@/utils/activationUtils";
import { NodeActivationData } from "@/utils/activationUtils";

/**
 * Format probability for display
 * Intelligently formats probability values based on magnitude
 */
const formatProbability = (prob: number): string => {
  const percentage = prob * 100;
  if (Math.abs(percentage) >= 0.01) {
    return `${percentage.toFixed(2)}%`;
  } else if (Math.abs(percentage) >= 0.0001) {
    return `${percentage.toFixed(4)}%`;
  } else if (prob !== 0) {
    return `${percentage.toExponential(2)}%`;
  } else {
    return `${percentage.toFixed(2)}%`; // For true 0, display 0.00%
  }
};

const maskActivationsToPosition = (activations: number[] | undefined, position: number): number[] | undefined => {
  if (!Array.isArray(activations) || position < 0 || position >= 64) return activations;
  const masked = new Array(64).fill(0);
  masked[position] = Number(activations[position]) || 0;
  return masked;
};

const filterZPatternToSourcePosition = (
  zPatternIndices: number[][] | undefined,
  zPatternValues: number[] | undefined,
  sourcePosition: number
): { zPatternIndices?: number[][]; zPatternValues?: number[] } => {
  if (!Array.isArray(zPatternIndices) || !Array.isArray(zPatternValues)) {
    return {};
  }

  const filteredIndices: number[][] = [];
  const filteredValues: number[] = [];
  const looksLikePairList = Array.isArray(zPatternIndices[0]) && zPatternIndices[0].length === 2;

  if (looksLikePairList) {
    for (let i = 0; i < Math.min(zPatternIndices.length, zPatternValues.length); i++) {
      const pair = zPatternIndices[i];
      if (Number(pair?.[0]) !== sourcePosition) continue;
      filteredIndices.push([Number(pair[0]), Number(pair[1])]);
      filteredValues.push(Number(zPatternValues[i]) || 0);
    }
  } else if (zPatternIndices.length >= 2) {
    const sources = zPatternIndices[0];
    const targets = zPatternIndices[1];
    for (let i = 0; i < Math.min(sources.length, targets.length, zPatternValues.length); i++) {
      if (Number(sources[i]) !== sourcePosition) continue;
      filteredIndices.push([Number(sources[i]), Number(targets[i])]);
      filteredValues.push(Number(zPatternValues[i]) || 0);
    }
  }

  return {
    zPatternIndices: filteredIndices.length > 0 ? filteredIndices : undefined,
    zPatternValues: filteredValues.length > 0 ? filteredValues : undefined,
  };
};

const restrictActivationDataToPosition = (
  data: NodeActivationData | null | undefined,
  position: number
): NodeActivationData | null | undefined => {
  if (!data) return data;
  const filteredZPattern = filterZPatternToSourcePosition(
    data.zPatternIndices,
    data.zPatternValues,
    position
  );
  return {
    ...data,
    activations: maskActivationsToPosition(data.activations, position),
    zPatternIndices: filteredZPattern.zPatternIndices,
    zPatternValues: filteredZPattern.zPatternValues,
  };
};

export const CircuitVisualization = () => {
  const {
    circuitData: linkGraphData,
    isLoading,
    error,
    clickedId,
    hoveredId,
    pinnedIds,
    hiddenIds,
    setCircuitData: setLinkGraphData,
    setLoading,
    setError,
    setClickedId,
    setHoveredId,
    setPinnedIds,
    setHiddenIds,
  } = useCircuitState();

  // Use reducer for state management instead of multiple useState calls
  const circuitState = useCircuitStateReducer();
  const {
    state: {
      file,
      activation,
      display,
      featureDiffing,
      positionMapping,
      dense,
      sync,
      clerp,
      steering,
      posFeature,
    },
    actions,
  } = circuitState;

  // Local UI state (not managed by reducer)
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [backendActivationError, setBackendActivationError] = useState<string | null>(null);

  // Extract FEN and moves using custom hook
  const fenExtraction = useFenExtraction({
    linkGraphData,
    originalCircuitJson: file.originalCircuitJson,
    multiOriginalJsons: file.multiOriginalJsons,
  });

  // Get dictionary names using custom hook
  const dictionaryName = useDictionaryName({ linkGraphData });

  // Get activation data using custom hook
  const activationDataHook = useActivationData({
    originalCircuitJson: file.originalCircuitJson,
    updateCounter: clerp.updateCounter,
  });

  // Circuit backend API
  const { fetchAllPositionsFromBackend, fetchFeatureActivationFromBackend } = useCircuitBackend({
    setLoadingAllPositions: actions.activation.setLoadingAllPositions,
    linkGraphData,
  });

  // Create aliases for easier access to reducer state (for backward compatibility during refactoring)
  const originalCircuitJson = file.originalCircuitJson;
  const originalFileName = file.originalFileName;
  const updateCounter = clerp.updateCounter;
  const hasUnsavedChanges = file.hasUnsavedChanges;
  const saveHistory = file.saveHistory;
  const multiOriginalJsons = file.multiOriginalJsons;
  const topActivations = activation.topActivations;
  const loadingTopActivations = activation.loadingTopActivations;
  const tokenPredictions = activation.tokenPredictions;
  const loadingTokenPredictions = activation.loadingTokenPredictions;
  const allPositionsActivationData = activation.allPositionsActivationData;
  const loadingAllPositions = activation.loadingAllPositions;
  const multiGraphActivationData = activation.multiGraphActivationData;
  const loadingBackendZPattern = activation.loadingBackendZPattern;
  const backendZPatternByNode = activation.backendZPatternByNode;
  const showAllPositions = display.showAllPositions;
  const showSubgraph = display.showSubgraph;
  const subgraphData = display.subgraphData;
  const subgraphRootNodeId = display.subgraphRootNodeId;
  const showDiffingLogs = display.showDiffingLogs;
  const perturbedFen = featureDiffing.perturbedFen;
  const isComparingFens = featureDiffing.isComparingFens;
  const inactiveNodes = featureDiffing.inactiveNodes;
  const diffingLogs = featureDiffing.diffingLogs;
  const enablePositionMapping = positionMapping.enablePositionMapping;
  const positionMappingSelections = positionMapping.positionMappingSelections;
  const draftPositionMappingSelections = positionMapping.draftPositionMappingSelections;
  const positionMappingApplyNonce = positionMapping.positionMappingApplyNonce;
  const denseNodes = dense.denseNodes;
  const denseThreshold = dense.denseThreshold;
  const checkingDenseFeatures = dense.checkingDenseFeatures;
  const syncingToBackend = sync.syncingToBackend;
  const syncingFromBackend = sync.syncingFromBackend;
  const editingClerp = clerp.editingClerp;
  const isSaving = clerp.isSaving;
  const steeringScale = steering.steeringScale;
  const steeringScaleInput = steering.steeringScaleInput;
  const posFeatureLayer = posFeature.posFeatureLayer;
  const posFeaturePositions = posFeature.posFeaturePositions;
  const posFeatureComponentType = posFeature.posFeatureComponentType;

  // Create aliases for setter functions (for backward compatibility during refactoring)
  const setOriginalCircuitJson = actions.file.setOriginalJson;
  // setUpdateCounter needs to support functional updates: (prev) => prev + 1
  const setUpdateCounter = useCallback((updater: number | ((prev: number) => number)) => {
    if (typeof updater === 'function') {
      const newValue = updater(updateCounter);
      actions.clerp.setUpdateCounter(newValue);
    } else {
      actions.clerp.setUpdateCounter(updater);
    }
  }, [updateCounter, actions.clerp]);
  const setHasUnsavedChanges = actions.file.setHasUnsavedChanges;
  // setSaveHistory needs to support functional updates: (prev) => [...prev, newEntry]
  const setSaveHistory = useCallback((updater: string[] | ((prev: string[]) => string[])) => {
    if (typeof updater === 'function') {
      const newHistory = updater(saveHistory);
      // Clear and rebuild history
      actions.file.clearSaveHistory();
      newHistory.forEach(entry => actions.file.addSaveHistory(entry));
    } else {
      actions.file.clearSaveHistory();
      updater.forEach(entry => actions.file.addSaveHistory(entry));
    }
  }, [saveHistory, actions.file]);
  const setTopActivations = actions.activation.setTopActivations;
  const setLoadingTopActivations = actions.activation.setLoadingTopActivations;
  const setTokenPredictions = actions.activation.setTokenPredictions;
  const setLoadingTokenPredictions = actions.activation.setLoadingTokenPredictions;
  const setAllPositionsActivationData = actions.activation.setAllPositionsData;
  const setLoadingAllPositions = actions.activation.setLoadingAllPositions;
  const setMultiGraphActivationData = actions.activation.setMultiGraphData;
  const setLoadingBackendZPattern = actions.activation.setLoadingBackendZPattern;
  const setBackendZPatternByNode = actions.activation.setBackendZPatternByNode;
  const setShowAllPositions = actions.display.setShowAllPositions;
  const setShowSubgraph = actions.display.setShowSubgraph;
  const setSubgraphData = actions.display.setSubgraphData;
  const setSubgraphRootNodeId = actions.display.setSubgraphRootNodeId;
  const setShowDiffingLogs = actions.display.setShowDiffingLogs;
  const setPerturbedFen = actions.featureDiffing.setPerturbedFen;
  const setIsComparingFens = actions.featureDiffing.setIsComparingFens;
  const setInactiveNodes = actions.featureDiffing.setInactiveNodes;
  // setDiffingLogs needs to support functional updates: (prev) => [...prev, newEntry] or []
  // Note: reducer's addDiffingLog automatically adds timestamp and formats message, so we extract just the message part
  const setDiffingLogs = useCallback((updater: Array<{timestamp: number; message: string}> | ((prev: Array<{timestamp: number; message: string}>) => Array<{timestamp: number; message: string}>)) => {
    if (typeof updater === 'function') {
      const newLogs = updater(diffingLogs);
      // Clear and rebuild logs
      actions.featureDiffing.clearDiffingLogs();
      // Extract message from logEntry (reducer will add timestamp and format)
      newLogs.forEach(log => {
        // Remove the timestamp prefix if present: "[HH:MM:SS] message" -> "message"
        const message = log.message.replace(/^\[\d{1,2}:\d{2}:\d{2}\]\s*/, '');
        actions.featureDiffing.addDiffingLog(message);
      });
    } else {
      actions.featureDiffing.clearDiffingLogs();
      updater.forEach(log => {
        const message = log.message.replace(/^\[\d{1,2}:\d{2}:\d{2}\]\s*/, '');
        actions.featureDiffing.addDiffingLog(message);
      });
    }
  }, [diffingLogs, actions.featureDiffing]);
  const setEnablePositionMapping = actions.positionMapping.setEnablePositionMapping;
  // setPositionMappingSelections needs to support functional updates: (prev) => Record<number, number>
  const setPositionMappingSelections = useCallback((updater: Record<number, number> | ((prev: Record<number, number>) => Record<number, number>)) => {
    if (typeof updater === 'function') {
      const newSelections = updater(positionMappingSelections);
      actions.positionMapping.setPositionMappingSelections(newSelections);
    } else {
      actions.positionMapping.setPositionMappingSelections(updater);
    }
  }, [positionMappingSelections, actions.positionMapping]);
  // setDraftPositionMappingSelections needs to support functional updates: (prev) => Record<number, number>
  const setDraftPositionMappingSelections = useCallback((updater: Record<number, number> | ((prev: Record<number, number>) => Record<number, number>)) => {
    if (typeof updater === 'function') {
      const newSelections = updater(draftPositionMappingSelections);
      actions.positionMapping.setDraftPositionMappingSelections(newSelections);
    } else {
      actions.positionMapping.setDraftPositionMappingSelections(updater);
    }
  }, [draftPositionMappingSelections, actions.positionMapping]);
  // setPositionMappingApplyNonce needs to support functional updates: (x) => x + 1
  const setPositionMappingApplyNonce = useCallback((updater: number | ((prev: number) => number)) => {
    if (typeof updater === 'function') {
      const newValue = updater(positionMappingApplyNonce);
      actions.positionMapping.setPositionMappingApplyNonce(newValue);
    } else {
      actions.positionMapping.setPositionMappingApplyNonce(updater);
    }
  }, [positionMappingApplyNonce, actions.positionMapping]);
  const setDenseNodes = actions.dense.setDenseNodes;
  const setDenseThreshold = actions.dense.setDenseThreshold;
  const setCheckingDenseFeatures = actions.dense.setCheckingDenseFeatures;
  const setSyncingToBackend = actions.sync.setSyncingToBackend;
  const setSyncingFromBackend = actions.sync.setSyncingFromBackend;
  const setEditingClerp = actions.clerp.setEditingClerp;
  const setIsSaving = actions.clerp.setIsSaving;
  const setSteeringScale = actions.steering.setSteeringScale;
  const setSteeringScaleInput = actions.steering.setSteeringScaleInput;
  const setPosFeatureLayer = actions.posFeature.setPosFeatureLayer;
  const setPosFeaturePositions = actions.posFeature.setPosFeaturePositions;
  const setPosFeatureComponentType = actions.posFeature.setPosFeatureComponentType;

  const parseNodeIdParts = useCallback((nodeId: string) => {
    const parsed = parseNodeId(nodeId);
    return { rawLayer: parsed.rawLayer, featureOrHead: parsed.featureOrHead, ctxIdx: parsed.ctxIdx };
  }, []);

  // When multi-graph file list changes, initialize position selections (default 0)
  useEffect(() => {
    const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    if (!names || names.length <= 1) return;
    setPositionMappingSelections((prev) => {
      const next: Record<number, number> = { ...prev };
      for (let i = 0; i < names.length; i++) {
        if (typeof next[i] !== "number" || !Number.isFinite(next[i])) {
          next[i] = 0;
        }
      }
      return next;
    });
  }, [linkGraphData, setPositionMappingSelections]);

  // Sync draft state: when multi-graph changes or first entry, make draft default to applied selections
  useEffect(() => {
    const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    if (!names || names.length <= 1) return;
    setDraftPositionMappingSelections((prev) => {
      const next: Record<number, number> = { ...prev };
      for (let i = 0; i < names.length; i++) {
        const applied = positionMappingSelections[i];
        const v = (typeof applied === "number" && Number.isFinite(applied)) ? applied : 0;
        if (typeof next[i] !== "number" || !Number.isFinite(next[i])) {
          next[i] = v;
        }
      }
      return next;
    });
  }, [linkGraphData, positionMappingSelections, setDraftPositionMappingSelections]);

  const handleFeatureClick = useCallback((node: Node, isMetaKey: boolean) => {
    if (isMetaKey) {
      // Toggle pinned state
      const newPinnedIds = pinnedIds.includes(node.nodeId)
        ? pinnedIds.filter(id => id !== node.nodeId)
        : [...pinnedIds, node.nodeId];
      setPinnedIds(newPinnedIds);
    } else {
      // Background click (empty nodeId) should clear selection
      if (!node.nodeId) {
        setClickedId(null);
        return;
      }

      const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
      const isMultiFile = !!(names && names.length > 1);

      // Single-file: toggle selection on same-node click; multi-file: always set (no toggle to avoid flicker)
      const newClickedId = isMultiFile
        ? node.nodeId
        : (node.nodeId === clickedId ? null : node.nodeId);
      console.log("[CircuitVisualization] setClickedId:", { newClickedId, prevClickedId: clickedId });
      setClickedId(newClickedId);
    }
  }, [clickedId, pinnedIds, linkGraphData, setClickedId, setPinnedIds]);

  const handleFeatureHover = useCallback((nodeId: string | null) => {
    // Only update if the hovered ID has actually changed
    if (nodeId !== hoveredId) {
      setHoveredId(nodeId);
    }
  }, [hoveredId, setHoveredId]);

  const handleFeatureSelect = useCallback((feature: Feature | null) => {
    setSelectedFeature(feature);
  }, []);

  const handleSelectedFeatureUpdated = useCallback((feature: Feature) => {
    setSelectedFeature(feature);
  }, []);

  const handleConnectedFeaturesSelect = useCallback((features: Feature[]) => {
    setConnectedFeatures(features);
  }, []);

  const handleConnectedFeaturesLoading = useCallback((_loading: boolean) => {
  }, []);

  // Use file upload hook
  const { handleSingleFileUpload: handleSingleFileUploadHook, handleMultiFilesUpload: handleMultiFilesUploadHook } = useCircuitFileUpload({
    onSuccess: useCallback((data: any, originalJson: any, fileName: string, multiJsons?: { json: CircuitJsonData; fileName: string }[]) => {
      setLinkGraphData(data);
      // Reset circuit state when loading new data
      setClickedId(null);
      setHoveredId(null);
      setPinnedIds([]);
      setHiddenIds([]);
      setSelectedFeature(null);
      setConnectedFeatures([]);
      
      // Update reducer state
      actions.file.setOriginalJson(originalJson);
      actions.file.setOriginalFileName(fileName);
      actions.file.setMultiOriginalJsons(multiJsons || []);
      actions.clerp.setEditingClerp('');
      actions.file.setHasUnsavedChanges(false);
      actions.file.clearSaveHistory();
      
      // Reset feature diffing state
      actions.featureDiffing.setPerturbedFen('');
      actions.featureDiffing.setInactiveNodes(new Set());
      actions.featureDiffing.clearDiffingLogs();
      actions.display.setShowDiffingLogs(false);
      
      // Reset subgraph state
      actions.display.setShowSubgraph(false);
      actions.display.setSubgraphData(null);
      actions.display.setSubgraphRootNodeId(null);
      
      // Reset activation display mode
      actions.display.setShowAllPositions(false);
      actions.activation.setAllPositionsData(null);
      actions.activation.setMultiGraphData({});
    }, [setLinkGraphData, setClickedId, setHoveredId, setPinnedIds, setHiddenIds, setSelectedFeature, setConnectedFeatures, actions]),
    onError: setError,
    setLoading,
  });

  // Wrapper functions for compatibility
  const handleSingleFileUpload = useCallback(async (file: File) => {
    await handleSingleFileUploadHook(file);
  }, [handleSingleFileUploadHook]);

  const handleMultiFilesUpload = useCallback(async (files: FileList | File[]) => {
    await handleMultiFilesUploadHook(files);
  }, [handleMultiFilesUploadHook]);


  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      if (files.length === 1) {
        handleSingleFileUpload(files[0]);
      } else {
        handleMultiFilesUpload(files);
      }
    }
  }, [handleSingleFileUpload, handleMultiFilesUpload]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      if (files.length === 1) {
        handleSingleFileUpload(files[0]);
      } else {
        handleMultiFilesUpload(files);
      }
    }
  }, [handleSingleFileUpload, handleMultiFilesUpload]);

  // Extract FEN from circuit data (using hook)
  const extractFenFromPrompt = fenExtraction.extractFenFromPrompt;

  // Extract FEN and moves from circuit JSON (using hooks)
  const extractFenFromCircuitJson = fenExtraction.extractFenFromCircuitJson;
  const extractOutputMove = fenExtraction.extractOutputMove;
  const extractOutputMoveFromCircuitJson = fenExtraction.extractOutputMoveFromCircuitJson;
 
  // Get node activation data (using hook)
  const getNodeActivationData = useCallback((nodeId: string | null): NodeActivationData => {
    return activationDataHook.getNodeActivationData(nodeId);
  }, [activationDataHook]);

  // Get node activation data from JSON (using hook)
  const getNodeActivationDataFromJson = activationDataHook.getNodeActivationDataFromJson;

  // Get dictionary name (using hook)
  const getDictionaryName = dictionaryName.getDictionaryName;

  // Get SAE name for Circuit (using hook)
  const getSaeNameForCircuit = dictionaryName.getSaeNameForCircuit;
 

  // Synchronous version: get activation data for all positions from file (for multi-file mode)
  const getAllPositionsActivationDataSync = useCallback((nodeId: string | null, jsonData?: any): NodeActivationData | null => {
    const dataToSearch = jsonData || file.originalCircuitJson;
    if (!nodeId || !dataToSearch) return null;
    return getAllPositionsActivationDataFromJson(dataToSearch, nodeId);
  }, [originalCircuitJson]);

  /**
   * Fetches activation data for a feature across all positions.
   * Multi-file mode: fetches per graph and merges into multiGraphActivationData.
   * Single-file mode: fetches from backend and returns merged result.
   */
  const getAllPositionsActivationData = useCallback(async (nodeId: string | null, _jsonData?: any): Promise<NodeActivationData | null> => {
    if (!nodeId) return null;

    const parsed = parseNodeId(nodeId);
    const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
    const featureTypeForNode = currentNode?.feature_type;
    const isLorsa = featureTypeForNode?.toLowerCase() === 'lorsa';
    const dictionary = getDictionaryName(parsed.layerForActivation, isLorsa);
    const isMultiFile = !!(linkGraphData?.metadata?.sourceFileNames && linkGraphData.metadata.sourceFileNames.length > 1);

    if (isMultiFile) {
      setLoadingAllPositions(true);
      const multiResults: Record<number, NodeActivationData> = {};

      for (let i = 0; i < multiOriginalJsons.length; i++) {
        const jsonData = multiOriginalJsons[i].json;
        const fileFen = extractFenFromCircuitJson(jsonData);

        if (fileFen) {
          const result = await fetchAllPositionsFromBackend(nodeId, fileFen, dictionary, parsed.featureOrHead);
          if (result) {
            multiResults[i] = {
              ...result,
              nodeType: featureTypeForNode,
              clerp: (currentNode as any)?.clerp,
            };
          }
        }
      }

      setMultiGraphActivationData(multiResults);
      setLoadingAllPositions(false);
      return null;
    }

    const fen = extractFenFromPrompt();
    if (!fen) return null;

    const result = await fetchAllPositionsFromBackend(nodeId, fen, dictionary, parsed.featureOrHead);
    if (!result) return null;

    return {
      ...result,
      nodeType: featureTypeForNode,
      clerp: (currentNode as any)?.clerp,
    };
  }, [linkGraphData, extractFenFromPrompt, getDictionaryName, fetchAllPositionsFromBackend, multiOriginalJsons, setMultiGraphActivationData, setLoadingAllPositions, extractFenFromCircuitJson]);

  const fen = extractFenFromPrompt();
  const outputMove = extractOutputMove();
  // Memoize to prevent new object reference every render (causes useEffect/useMemo loops)
  // Use activationDataHook.getNodeActivationData directly - it's stable (useCallback in hook)
  const nodeActivationData = useMemo(
    () => activationDataHook.getNodeActivationData(clickedId),
    [activationDataHook.getNodeActivationData, clickedId]
  );

  /** Reset display mode to single position when switching nodes. */
  useEffect(() => {
    setShowAllPositions(false);
    setAllPositionsActivationData(null);
    setMultiGraphActivationData({});
  }, [clickedId]);

  /** Fetch current-FEN activation data from backend in single-file, single-position mode. */
  useEffect(() => {
    setBackendZPatternByNode(null);
    setBackendActivationError(null);

    if (!clickedId || showAllPositions) return;

    const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    if (names && names.length > 1) return;

    const currentNode = (linkGraphData?.nodes || []).find((n: any) => n?.nodeId === clickedId);
    const featureType = typeof currentNode?.feature_type === "string" ? currentNode.feature_type.toLowerCase() : "";

    const fenLocal = extractFenFromPrompt();
    if (!fenLocal) return;

    const parts = clickedId.split("_");
    const rawLayer = Number(parts[0]) || 0;
    const featureIndex = Number(parts[1]) || 0;
    const layerIdx = Math.floor(rawLayer / 2);
    const dictionary = getDictionaryName(layerIdx, featureType === "lorsa");

    const controller = new AbortController();
    setLoadingBackendZPattern(true);

    fetchFeatureActivationFromBackend(dictionary, featureIndex, fenLocal, controller.signal)
      .then((data) => {
        if (!data) {
          setBackendActivationError("Backend did not return feature activation data.");
          return;
        }
        const ctxIdx = Number(parts[2]) || 0;
        if (!data.activations) {
          setBackendActivationError("Backend response did not include activations.");
        } else if ((Number(data.activations[ctxIdx]) || 0) === 0) {
          setBackendActivationError("Current node position has zero activation for this feature.");
        } else {
          setBackendActivationError(null);
        }
        setBackendZPatternByNode({
          nodeId: clickedId,
          activations: data.activations,
          zPatternIndices: data.zPatternIndices,
          zPatternValues: data.zPatternValues,
        });
      })
      .finally(() => {
        setLoadingBackendZPattern(false);
      });

    return () => {
      controller.abort();
    };
  }, [
    clickedId,
    showAllPositions,
    linkGraphData,
    extractFenFromPrompt,
    getDictionaryName,
    fetchFeatureActivationFromBackend,
  ]);

  /** Update all-positions activation data when node or mode changes. */
  useEffect(() => {
    if (clickedId && showAllPositions) {
      const names = (linkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
      const isMultiFile = !!(names && names.length > 1);

      if (!isMultiFile) {
        setLoadingAllPositions(true);
        getAllPositionsActivationData(clickedId)
          .then(setAllPositionsActivationData)
          .catch((err) => {
            console.error("Failed to fetch all-positions activation data:", err);
            setAllPositionsActivationData(null);
          })
          .finally(() => setLoadingAllPositions(false));
      } else {
        getAllPositionsActivationData(clickedId);
      }
    } else {
      setAllPositionsActivationData(null);
      setLoadingAllPositions(false);
    }
  }, [clickedId, showAllPositions, linkGraphData, getAllPositionsActivationData]);

  /** Activation data to display. Uses backend z_pattern in single-position mode to override JSON when available. */
  const displayActivationData = useMemo(() => {
    const base = (showAllPositions && allPositionsActivationData)
      ? allPositionsActivationData
      : nodeActivationData;

    if (!showAllPositions && clickedId) {
      const clickedPosition = parseNodeId(clickedId).ctxIdx;
      if (backendZPatternByNode?.nodeId === clickedId) {
        const backendActivations = backendZPatternByNode.activations;
        const hasBackendPositionActivation = (Number(backendActivations?.[clickedPosition]) || 0) !== 0;
        return restrictActivationDataToPosition({
          ...base,
          activations: hasBackendPositionActivation ? backendActivations : base?.activations,
          zPatternIndices: backendZPatternByNode.zPatternIndices,
          zPatternValues: backendZPatternByNode.zPatternValues,
        }, clickedPosition);
      }
      return restrictActivationDataToPosition(base, clickedPosition);
    }

    return base;
  }, [showAllPositions, allPositionsActivationData, nodeActivationData, clickedId, backendZPatternByNode]);

  /** Sync editor with selected node's clerp; reset when no node selected. */
  useEffect(() => {
    if (clickedId && nodeActivationData) {
      setEditingClerp(nodeActivationData.clerp || "");
    } else {
      setEditingClerp("");
      setShowAllPositions(false);
      setAllPositionsActivationData(null);
      setMultiGraphActivationData({});
    }
  }, [clickedId, nodeActivationData?.clerp, updateCounter]);

  const handleSaveClerp = useCallback(async () => {
    if (!clickedId || !originalCircuitJson) return;

    const trimmedClerp = editingClerp.trim();
    
    setIsSaving(true);
    
    try {
      const updatedCircuitJson = JSON.parse(JSON.stringify(originalCircuitJson));
      const nodesToSearch = findNodesArray(updatedCircuitJson);
      let updated = false;

      for (const node of nodesToSearch) {
        if (node && typeof node === "object" && node.node_id === clickedId) {
          node.clerp = trimmedClerp;
          updated = true;
          break;
        }
      }

      if (!updated) {
        throw new Error(`Node not found (node_id: ${clickedId})`);
      }

      setOriginalCircuitJson(updatedCircuitJson);
      setUpdateCounter((prev) => prev + 1);
      setHasUnsavedChanges(true);

      const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const baseName = (originalFileName || "circuit_data.json").replace(".json", "");
      const updatedFileName = `${baseName}_updated_${timestamp}.json`;

      const blob = new Blob([JSON.stringify(updatedCircuitJson, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = updatedFileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      const historyEntry = `${new Date().toLocaleTimeString()}: Node ${clickedId} - ${trimmedClerp.length === 0 ? "cleared clerp" : `updated: ${trimmedClerp.substring(0, 30)}...`}`;
      setSaveHistory((prev) => [...prev, historyEntry]);

      alert(
        `Clerp saved and downloaded.${trimmedClerp.length === 0 ? " (empty)" : ""}\n\n` +
          `File: ${updatedFileName}\n\n` +
          `Tip: Replace the original file or re-upload to this page.`
      );
    } catch (err) {
      console.error("Save failed:", err);
      alert("Save failed: " + (err instanceof Error ? err.message : "Unknown error"));
    } finally {
      setIsSaving(false);
    }
  }, [clickedId, originalCircuitJson, editingClerp, originalFileName, setOriginalCircuitJson, updateCounter]);

  /** Quick export of current circuit state to JSON file. */
  const handleQuickExport = useCallback(() => {
    if (!originalCircuitJson) return;
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const fileName = originalFileName || 'circuit_data.json';
    const baseName = fileName.replace('.json', '');
    const exportFileName = `${baseName}_export_${timestamp}.json`;
    
    const jsonString = JSON.stringify(originalCircuitJson, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = exportFileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    setHasUnsavedChanges(false);
    alert(`File exported to Downloads:\n${exportFileName}\n\nTip: Replace the original file or drag the new file here to reload.`);
  }, [originalCircuitJson, originalFileName]);

  /** Fetches top activation data for a node from the backend. */
  const fetchTopActivations = useCallback(async (nodeId: string) => {
    if (!nodeId) return;
    
    setLoadingTopActivations(true);
    try {
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
      const isLorsa = currentNode?.feature_type?.toLowerCase() === 'lorsa';
      
      const dictionary = getDictionaryName(layerIdx, isLorsa);
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/features/${featureIndex}`,
        {
          method: "GET",
          headers: {
            Accept: "application/x-msgpack",
          },
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const arrayBuffer = await response.arrayBuffer();
      const decoded = await import("@msgpack/msgpack").then(module => module.decode(new Uint8Array(arrayBuffer)));
      const camelcaseKeys = await import("camelcase-keys").then(module => module.default);
      
      // Parse response data
      const camelData = camelcaseKeys(decoded as Record<string, unknown>, {
        deep: true,
        stopPaths: ["sample_groups.samples.context"],
      }) as any;
      
      // Extract sample data
      const sampleGroups = camelData?.sampleGroups || camelData?.sample_groups || [];
      const allSamples: any[] = [];
      
      for (const group of sampleGroups) {
        if (group.samples && Array.isArray(group.samples)) {
          allSamples.push(...group.samples);
        }
      }
      
      // Find samples containing FEN and extract activations
      const chessSamples: any[] = [];
      
      for (const sample of allSamples) {
        if (sample.text) {
          const lines = sample.text.split('\n');
          
          for (const line of lines) {
            const trimmed = line.trim();
            
            // Check for FEN format
            if (trimmed.includes('/')) {
              const parts = trimmed.split(/\s+/);
              
              if (parts.length >= 6) {
                const [boardPart, activeColor] = parts;
                const boardRows = boardPart.split('/');
                
                if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
                  // Validate FEN format
                  let isValidBoard = true;
                  let totalSquares = 0;
                  
                  for (const row of boardRows) {
                    if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) {
                      isValidBoard = false;
                      break;
                    }
                    
                    let rowSquares = 0;
                    for (const char of row) {
                      if (/\d/.test(char)) {
                        rowSquares += parseInt(char);
                      } else {
                        rowSquares += 1;
                      }
                    }
                    totalSquares += rowSquares;
                  }
                  
                  if (isValidBoard && totalSquares === 64) {
                    // Map sparse activation data to 64-square board
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0; // Use max activation for ranking
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      // Create 64-square activation array
                      activationsArray = new Array(64).fill(0);
                      
                      // Map sparse activations to board positions and find max
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        // Ensure index is within valid range
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          // Use max activation (aligned with feature page logic)
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                      
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation, // Used for sorting
                      activations: activationsArray,
                      ...normalizeZPattern(sample.zPatternIndices, sample.zPatternValues),
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0
                    });
                    
                    break; // Stop after first valid FEN
                  }
                }
              }
            }
          }
        }
      }
      
      // Sort by max activation and take top 8 (aligned with feature page logic)
      const topSamples = chessSamples
        .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
        .slice(0, 8);
      
      
      setTopActivations(topSamples);
      
    } catch (error) {
      console.error("Failed to fetch top activation data:", error);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [linkGraphData, getDictionaryName]);

  /** Fetches Token Predictions data from backend */
  const fetchTokenPredictions = useCallback(async (nodeId: string, currentSteeringScale?: number) => {
    if (!nodeId || !fen) return;
    
    setLoadingTokenPredictions(true);
    try {
      // Parse feature info from nodeId
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const pos = Number(parts[2]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      const currentNode = linkGraphData?.nodes.find(n => n.nodeId === nodeId);
      const featureType = currentNode?.feature_type?.toLowerCase() === 'lorsa' ? 'lorsa' : 'transcoder';
      const saeComboId = typeof window !== 'undefined'
        ? window.localStorage.getItem("bt4_sae_combo_id") || undefined
        : undefined;
      
      // Use passed steeringScale or value from current state
      const scaleToUse = currentSteeringScale !== undefined ? currentSteeringScale : steeringScale;
      
      console.log("Fetching Token Predictions:", {
        nodeId,
        layerIdx,
        featureIndex,
        pos,
        featureType,
        fen,
        steering_scale: scaleToUse
      });
      
      // Call backend API for steering analysis (supports steering_scale)
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/steering_analysis`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            fen: fen,
            feature_type: featureType,
            layer: layerIdx,
            pos: pos,
            feature: featureIndex,
            steering_scale: scaleToUse,
            metadata: {
              ...(linkGraphData?.metadata || {}),
              sae_combo_id: saeComboId,
            }
          })
        }
      );
      
      if (!response.ok) {
        const errorText = await response.text();
        const normalizedError = errorText.toLowerCase();
        const needsPreload =
          normalizedError.includes("not preloaded") ||
          normalizedError.includes("not loaded") ||
          normalizedError.includes("preload_models") ||
          normalizedError.includes("still loading");
        if (needsPreload) {
          alert("Please load TC/Lorsa combo (SaeComboLoader) above first, then use steering.");
        }
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      const result = await response.json();
      
      console.log("Token Predictions fetched:", result);
      
      setTokenPredictions(result);
      
    } catch (error) {
      console.error("Failed to fetch Token Predictions:", error);
      setTokenPredictions(null);
    } finally {
      setLoadingTokenPredictions(false);
    }
  }, [fen, linkGraphData, steeringScale]);

  /** Fetch Top Activation data when node is clicked (Token Predictions is manual) */
  useEffect(() => {
    if (clickedId) {
      fetchTopActivations(clickedId);
    } else {
      setTopActivations([]);
      setTokenPredictions(null);
    }
  }, [clickedId, fetchTopActivations]);

  /** Sync clerps to backend interpretations */
  const syncClerpsToBackend = useCallback(async () => {
    if (!originalCircuitJson || !originalCircuitJson.nodes) {
      alert("No node data available");
      return;
    }
    
    setSyncingToBackend(true);
    try {
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const lorsaAnalysisName = metadata?.lorsa_analysis_name;
      const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
      
      const nodes = originalCircuitJson.nodes.map((node: any) => {
        const parts = node.node_id.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.node_id,
          clerp: node.clerp || '',
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      console.log('Sync interpretations to backend:', {
        totalNodes: nodes.length,
        nodesWithClerp: nodes.filter((n: any) => n.clerp).length,
        lorsaAnalysisName,
        tcAnalysisName
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_clerps_to_interpretations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ Synchronization completed:', result);
      
      alert(
        `Interpretation synchronized to backend successfully!\n\n` +
        `Statistics:\n` +
        `- Total nodes: ${result.total_nodes}\n` +
        `- Successfully synchronized: ${result.synced}\n` +
        `- Skipped (no clerp): ${result.skipped}\n` +
        `- Failed: ${result.errors}`
      );
      
    } catch (error) {
      console.error('Synchronization failed:', error);
      alert(`Synchronization failed: ${error}`);
    } finally {
      setSyncingToBackend(false);
    }
  }, [originalCircuitJson, linkGraphData]);

  const syncClerpsFromBackend = useCallback(async () => {
    if (!originalCircuitJson || !originalCircuitJson.nodes) {
      alert("No node data available");
      return;
    }
    
    setSyncingFromBackend(true);
    try {
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const lorsaAnalysisName = metadata?.lorsa_analysis_name;
      const tcAnalysisName = metadata?.tc_analysis_name || metadata?.clt_analysis_name;
      
      // Prepare node data
      const nodes = originalCircuitJson.nodes.map((node: any) => {
        const parts = node.node_id.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.node_id,
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      console.log('Sync interpretations to clerps from backend:', {
        totalNodes: nodes.length,
        lorsaAnalysisName,
        tcAnalysisName
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/sync_interpretations_to_clerps`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ Synchronization completed:', result);
      
      // Update original JSON data
      const updatedCircuitJson = JSON.parse(JSON.stringify(originalCircuitJson));
      
      // Update clerp based on returned updated_nodes
      const updatedNodesMap = new Map(
        result.updated_nodes.map((n: any) => [n.node_id, n.clerp])
      );
      
      let updatedCount = 0;
      updatedCircuitJson.nodes.forEach((node: any) => {
        if (updatedNodesMap.has(node.node_id)) {
          const newClerp = updatedNodesMap.get(node.node_id);
          if (newClerp) {
            node.clerp = newClerp;
            updatedCount++;
          }
        }
      });
      
      // Update state
      setOriginalCircuitJson(updatedCircuitJson);
      setUpdateCounter(prev => prev + 1);
      setHasUnsavedChanges(true);
      
      alert(
        `Interpretation synchronized from backend successfully!\n\n` +
        `Statistics:\n` +
        `- Total nodes: ${result.total_nodes}\n` +
        `- Found interpretation: ${result.found}\n` +
        `- Not found: ${result.not_found}\n` +
        `- Actual updated: ${updatedCount}\n\n` +
        `Suggestion: Click "Export" button to save the updated file`
      );
      
    } catch (error) {
      console.error('Synchronization failed:', error);
      alert(`Synchronization failed: ${error}`);
    } finally {
      setSyncingFromBackend(false);
    }
  }, [originalCircuitJson, linkGraphData, setOriginalCircuitJson, setUpdateCounter, setHasUnsavedChanges]);

  const checkDenseFeatures = useCallback(async () => {
    if (!linkGraphData || !linkGraphData.nodes) {
      console.warn('No available node data');
      return;
    }
    
    setCheckingDenseFeatures(true);
    try {
      const threshold = denseThreshold === '' ? null : parseInt(denseThreshold);
      
      // Extract all node information from linkGraphData
      const nodes = linkGraphData.nodes.map(node => {
        // Parse layer and feature from nodeId
        const parts = node.nodeId.split('_');
        const rawLayer = parseInt(parts[0]) || 0;
        const featureIdx = parseInt(parts[1]) || 0;
        const layerForActivation = Math.floor(rawLayer / 2);
        
        return {
          node_id: node.nodeId,
          feature: featureIdx,
          layer: layerForActivation,
          feature_type: node.feature_type || ''
        };
      });
      
      // Extract model name from metadata and convert to analysis_name
      const metadata = (linkGraphData.metadata || {}) as any;
      const lorsaAnalysisNameRaw = metadata.lorsa_analysis_name;
      const tcAnalysisNameRaw = metadata.tc_analysis_name || metadata.clt_analysis_name;
      // Read sae_series from metadata, use default value if not present
      const saeSeries = (metadata as any).sae_series || 'BT4-exp128';
      
      // Build template based on analysis_name
      // Format: If analysis_name is "BT4_lorsa_k30_e16", template should be "BT4_lorsa_L{}A_k30_e16"
      // If analysis_name is "BT4_lorsa" (default), template should be "BT4_lorsa_L{}A"
      let lorsaAnalysisName = undefined;
      let tcAnalysisName = undefined;
      
      if (lorsaAnalysisNameRaw) {
        if (lorsaAnalysisNameRaw.includes('BT4_lorsa')) {
          // Extract suffix (if present): "BT4_lorsa_k30_e16" -> "k30_e16"
          const suffix = lorsaAnalysisNameRaw.replace('BT4_lorsa', '').replace(/^_/, '');
          if (suffix) {
            lorsaAnalysisName = `BT4_lorsa_L{}A_${suffix}`;
          } else {
            lorsaAnalysisName = 'BT4_lorsa_L{}A';
          }
        } else if (lorsaAnalysisNameRaw.includes('T82')) {
          lorsaAnalysisName = 'lc0-lorsa-L{}';
        }
      }
      
      if (tcAnalysisNameRaw) {
        if (tcAnalysisNameRaw.includes('BT4_tc')) {
          // Extract suffix (if present): "BT4_tc_k30_e16" -> "k30_e16"
          const suffix = tcAnalysisNameRaw.replace('BT4_tc', '').replace(/^_/, '');
          if (suffix) {
            tcAnalysisName = `BT4_tc_L{}M_${suffix}`;
          } else {
            tcAnalysisName = 'BT4_tc_L{}M';
          }
        } else if (tcAnalysisNameRaw.includes('T82')) {
          tcAnalysisName = 'lc0_L{}M_16x_k30_lr2e-03_auxk_sparseadam';
        }
      }
      
      console.log('Check dense features:', {
        totalNodes: nodes.length,
        threshold: threshold,
        saeSeries: saeSeries,
        lorsaAnalysisNameRaw: lorsaAnalysisNameRaw,
        tcAnalysisNameRaw: tcAnalysisNameRaw,
        lorsaAnalysisName: lorsaAnalysisName,
        tcAnalysisName: tcAnalysisName,
        sampleNodes: nodes.slice(0, 3)
      });
      
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/check_dense_features`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            nodes: nodes,
            threshold: threshold,
            sae_series: saeSeries,
            lorsa_analysis_name: lorsaAnalysisName,
            tc_analysis_name: tcAnalysisName
          })
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const result = await response.json();
      
      console.log('✅ Dense features checked successfully:', {
        denseNodeCount: result.dense_nodes.length,
        totalNodes: result.total_nodes,
        threshold: result.threshold
      });
      
      setDenseNodes(new Set(result.dense_nodes));
      
    } catch (error) {
      console.error('Check dense features failed:', error);
      alert(`Check dense features failed: ${error}`);
    } finally {
      setCheckingDenseFeatures(false);
    }
  }, [linkGraphData, denseThreshold]);

  // Apply dense node color overlay
  const applyDenseNodeColors = useCallback((data: any) => {
    if (!data || !data.nodes || denseNodes.size === 0) {
      return data;
    }
    
    return {
      ...data,
      nodes: data.nodes.map((node: any) => {
        if (denseNodes.has(node.nodeId)) {
          return {
            ...node,
            nodeColor: '#000000',  // Black
            isDense: true 
          };
        }
        return node;
      })
    };
  }, [denseNodes]);

  // Apply inactive node color overlay (golden)
  const applyInactiveNodeColors = useCallback((data: any) => {
    if (!data || !data.nodes || inactiveNodes.size === 0) {
      return data;
    }
    
    return {
      ...data,
      nodes: data.nodes.map((node: any) => {
        if (inactiveNodes.has(node.nodeId)) {
          return {
            ...node,
            nodeColor: '#FFD700',  // Golden
            isInactive: true
          };
        }
        return node;
      })
    };
  }, [inactiveNodes]);

  const applyPositionMappingHighlights = useCallback((data: any) => {
    if (!enablePositionMapping) return data;
    if (!data || !data.nodes || !data.metadata?.sourceFileNames || data.metadata.sourceFileNames.length <= 1) {
      return data;
    }

    type Hit = { nodeId: string; sourceIdx: number };
    const buckets = new Map<string, Hit[]>();

    for (const node of data.nodes as any[]) {
      const nodeId = String(node.nodeId);
      const { rawLayer, featureOrHead, ctxIdx } = parseNodeIdParts(nodeId);

      const ftRaw = typeof node.feature_type === "string" ? node.feature_type : "";
      const ft = ftRaw.toLowerCase();
      if (ft.includes("embedding") || ft.includes("logit") || ft.includes("error")) continue;

      const srcs: number[] = Array.isArray(node.sourceIndices)
        ? node.sourceIndices
        : (typeof node.sourceIndex === "number" ? [node.sourceIndex] : []);

      for (const s of srcs) {
        const sel = positionMappingSelections[s];
        if (typeof sel !== "number") continue;
        if (ctxIdx !== sel) continue;
        const typeNorm = ft.includes("lorsa") ? "lorsa" : "tc";
        const key = `${rawLayer}_${featureOrHead}_${typeNorm}`;
        const arr = buckets.get(key) || [];
        arr.push({ nodeId, sourceIdx: s });
        buckets.set(key, arr);
      }
    }

    const highlightNodeIds = new Set<string>();
    buckets.forEach((hits) => {
      const uniqSources = new Set(hits.map((h) => h.sourceIdx));
      if (uniqSources.size >= 2) {
        for (const h of hits) highlightNodeIds.add(h.nodeId);
      }
    });

    if (highlightNodeIds.size === 0) return data;

    return {
      ...data,
      nodes: (data.nodes as any[]).map((node) => {
        if ((node as any)?.isDense === true) {
          return node;
        }
        if (highlightNodeIds.has(String(node.nodeId))) {
          return {
            ...node,
            nodeColor: POSITION_MAPPING_HIGHLIGHT_COLOR,
            isPositionMapped: true,
          };
        }
        return node;
      }),
    };
  }, [enablePositionMapping, parseNodeIdParts, positionMappingSelections, POSITION_MAPPING_HIGHLIGHT_COLOR]);

  const displayLinkGraphData = useMemo(() => {
    let data = linkGraphData;
    data = applyDenseNodeColors(data);
    data = applyInactiveNodeColors(data);
    data = applyPositionMappingHighlights(data);
    return data;
  }, [linkGraphData, applyDenseNodeColors, applyInactiveNodeColors, applyPositionMappingHighlights]);

  // Count the number of nodes currently highlighted (for user feedback "has it worked")
  const positionMappedCount = useMemo(() => {
    const names = (displayLinkGraphData as any)?.metadata?.sourceFileNames as string[] | undefined;
    if (!enablePositionMapping || !names || names.length <= 1) return 0;
    return (displayLinkGraphData?.nodes || []).filter((n: any) => (n as any)?.isPositionMapped === true).length;
  }, [displayLinkGraphData, enablePositionMapping]);

  // Compare FEN activation differences
  const compareFenActivations = useCallback(async () => {
    if (!originalCircuitJson || !perturbedFen.trim()) {
      alert('Please upload graph file and input perturbed FEN');
      return;
    }

    // Get original FEN
    const originalFen = extractFenFromPrompt();
    if (!originalFen) {
      alert('Cannot extract original FEN from graph file');
      return;
    }

    setIsComparingFens(true);
    setDiffingLogs([]);
    setShowDiffingLogs(true);

    const addLog = (message: string) => {
      const logEntry = {
        timestamp: Date.now(),
        message: `[${new Date().toLocaleTimeString()}] ${message}`
      };
      setDiffingLogs(prev => [...prev, logEntry]);
      console.log(logEntry.message);
    };

    try {
      addLog('Comparing FEN activation differences...');
      addLog(`Original FEN: ${originalFen}`);
      addLog(`Perturbed FEN: ${perturbedFen}`);

      // Extract model name from metadata
      const metadata = (linkGraphData?.metadata || originalCircuitJson?.metadata) as any;
      const modelName = metadata?.model_name || 'lc0/BT4-1024x15x32h';

      addLog(`Using model: ${modelName}`);

      // Call backend API
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/circuit/compare_fen_activations`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            graph_json: originalCircuitJson,
            original_fen: originalFen,
            perturbed_fen: perturbedFen.trim(),
            model_name: modelName,
            activation_threshold: 0.0
          })
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `HTTP ${response.status}: ${errorText}`;
        try {
          const errorJson = JSON.parse(errorText);
          if (errorJson.detail) {
            errorMessage = errorJson.detail;
          }
        } catch {
          // If cannot parse JSON, use original error text
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      
      addLog(`✅ Comparison completed:`);
      addLog(`   - Total nodes: ${result.total_nodes}`);
      addLog(`   - Inactive nodes: ${result.inactive_nodes_count}`);

      // Update inactive nodes set
      const inactiveNodeIds = new Set<string>(
        result.inactive_nodes.map((node: any) => String(node.node_id))
      );
      setInactiveNodes(inactiveNodeIds);

      // Display statistics
      if (result.statistics) {
        addLog(`By layer statistics:`);
        Object.entries(result.statistics.by_layer).forEach(([layer, count]) => {
          addLog(`   Layer ${layer}: ${count} nodes`);
        });
        addLog(`By type statistics:`);
        Object.entries(result.statistics.by_type).forEach(([type, count]) => {
          addLog(`   ${type}: ${count} nodes`);
        });
      }

      alert(
        `FEN activation difference comparison completed!\n\n` +
        `Statistics:\n` +
        `- Total nodes: ${result.total_nodes}\n` +
        `- Inactive nodes: ${result.inactive_nodes_count}\n\n` +
        `Inactive nodes are marked in golden color in the graph`
      );

    } catch (error) {
      console.error('Comparison failed:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      addLog(`Comparison failed: ${errorMessage}`);
      alert(`Comparison failed: ${errorMessage}`);
    } finally {
      setIsComparingFens(false);
    }
  }, [originalCircuitJson, perturbedFen, linkGraphData, extractFenFromPrompt]);

  // Recursively find all upstream nodes of a node (including their upstream nodes)
  const findUpstreamNodes = useCallback((nodeId: string, graphData: any): Set<string> => {
    const upstreamNodes = new Set<string>();
    const visited = new Set<string>();
    
    const traverse = (currentNodeId: string) => {
      if (visited.has(currentNodeId)) return;
      visited.add(currentNodeId);
      upstreamNodes.add(currentNodeId);
      
      // Find all edges pointing to the current node (incoming edges)
      const incomingLinks = graphData.links.filter((link: any) => link.target === currentNodeId);
      
      // Recursively find the upstream nodes of each source node
      for (const link of incomingLinks) {
        traverse(link.source);
      }
    };
    
    traverse(nodeId);
    return upstreamNodes;
  }, []);

  // Create subgraph data
  const createSubgraph = useCallback((rootNodeId: string, graphData: any) => {
    const upstreamNodeIds = findUpstreamNodes(rootNodeId, graphData);
    
    // Filter nodes: only keep upstream nodes
    const subgraphNodes = graphData.nodes.filter((node: any) => 
      upstreamNodeIds.has(node.nodeId)
    );
    
    // Filter edges: only keep edges where both ends are in the subgraph
    const subgraphLinks = graphData.links.filter((link: any) => 
      upstreamNodeIds.has(link.source) && upstreamNodeIds.has(link.target)
    );
    
    // Create subgraph data structure
    const subgraph = {
      nodes: subgraphNodes,
      links: subgraphLinks,
      metadata: {
        ...graphData.metadata,
        subgraphRoot: rootNodeId,
        originalNodeCount: graphData.nodes.length,
        subgraphNodeCount: subgraphNodes.length,
        originalLinkCount: graphData.links.length,
        subgraphLinkCount: subgraphLinks.length,
        createdAt: new Date().toISOString(),
        isSubgraph: true
      }
    };
    
    console.log('Create subgraph:', {
      rootNodeId,
      totalUpstreamNodes: upstreamNodeIds.size,
      subgraphNodes: subgraphNodes.length,
      subgraphLinks: subgraphLinks.length,
      originalNodes: graphData.nodes.length,
      originalLinks: graphData.links.length
    });
    
    return subgraph;
  }, [findUpstreamNodes]);

  // Display subgraph
  const handleShowSubgraph = useCallback(() => {
    if (!clickedId || !displayLinkGraphData) return;
    
    const subgraph = createSubgraph(clickedId, displayLinkGraphData);
    setSubgraphData(subgraph);
    setSubgraphRootNodeId(clickedId);
    setShowSubgraph(true);
    
    console.log('Display subgraph mode:', {
      rootNodeId: clickedId,
      nodeCount: subgraph.nodes.length,
      linkCount: subgraph.links.length
    });
  }, [clickedId, displayLinkGraphData, createSubgraph]);

  // Exit subgraph mode
  const handleExitSubgraph = useCallback(() => {
    setShowSubgraph(false);
    setSubgraphData(null);
    setSubgraphRootNodeId(null);
    console.log('Exit subgraph mode');
  }, []);

  // Save subgraph as JSON file
  const handleSaveSubgraph = useCallback(() => {
    if (!subgraphData || !subgraphRootNodeId) return;
    
    // Get complete node information from original data (including activations, z_pattern, etc.)
    const enrichSubgraphWithOriginalData = (subgraph: any) => {
      if (!originalCircuitJson) return subgraph;
      
      const enrichedNodes = subgraph.nodes.map((node: any) => {
        // Find corresponding complete node data in original JSON
        const originalNodeData = getNodeActivationDataFromJson(originalCircuitJson, node.nodeId);
        
        return {
          ...node,
          // Add complete information from original data
          activations: originalNodeData.activations,
          zPatternIndices: originalNodeData.zPatternIndices,
          zPatternValues: originalNodeData.zPatternValues,
          clerp: originalNodeData.clerp,
          // Keep all original fields
          ...(originalCircuitJson.nodes?.find((n: any) => n.node_id === node.nodeId) || {})
        };
      });
      
      return {
        ...subgraph,
        nodes: enrichedNodes
      };
    };
    
    const enrichedSubgraph = enrichSubgraphWithOriginalData(subgraphData);
    
    // Generate file name
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const rootNodeForFilename = subgraphRootNodeId.replace(/[^a-zA-Z0-9]/g, '_');
    const fileName = `subgraph_${rootNodeForFilename}_${timestamp}.json`;
    
    // Create download
    const jsonString = JSON.stringify(enrichedSubgraph, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    console.log('Subgraph saved:', {
      fileName,
      rootNodeId: subgraphRootNodeId,
      nodeCount: enrichedSubgraph.nodes.length,
      linkCount: enrichedSubgraph.links.length
    });
    
    alert(
      `Subgraph saved!\n\n` +
      `File name: ${fileName}\n` +
      `Root node: ${subgraphRootNodeId}\n` +
      `Statistics:\n` +
      `  - Node count: ${enrichedSubgraph.nodes.length}\n` +
      `  - Edge count: ${enrichedSubgraph.links.length}\n` +
      `  - Contains complete activation data and z_pattern information\n\n` +
      `File saved to Downloads folder`
    );
    
  }, [subgraphData, subgraphRootNodeId, originalCircuitJson, getNodeActivationDataFromJson]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <h3 className="text-lg font-semibold text-red-600 mb-2">Failed to load circuit visualization</h3>
          <p className="text-gray-600">{error}</p>
          <button 
            onClick={() => setError(null)}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading circuit visualization...</p>
        </div>
      </div>
    );
  }

  if (!linkGraphData) {
    return (
      <div className="space-y-6">
        <SaeComboLoader />

        {/* Header */}
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Circuit Visualization</h2>
        </div>

        <FileUploadZone
          isDragOver={isDragOver}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onFileInput={handleFileInput}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6 w-full max-w-full overflow-hidden">
      <SaeComboLoader />

      {/* Header */}
      <div className="flex flex-wrap items-start gap-3">
        <div className="flex items-center space-x-2 min-w-0">
          <h2 className="text-l font-bold whitespace-nowrap">Prompt:</h2>
          <h2 className="text-l truncate">{displayLinkGraphData?.metadata?.prompt_tokens?.join(' ') || ''}</h2>
        </div>
        <div className="flex flex-wrap items-start justify-end gap-3 ml-auto">
          <button
            onClick={() => setLinkGraphData(null)}
            className="px-4 py-2 text-sm font-medium bg-blue-500 text-white rounded-lg border-2 border-blue-600 hover:bg-blue-600 hover:border-blue-700 shadow-md transition-all"
          >
            Upload New File
          </button>
          {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-yellow-50 rounded-md border border-yellow-200">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-700">Perturb FEN:</label>
                <input
                  type="text"
                  value={perturbedFen}
                  onChange={(e) => setPerturbedFen(e.target.value)}
                  placeholder="Enter perturbed FEN..."
                  className="w-64 px-2 py-1 text-sm border border-gray-300 rounded"
                  disabled={isComparingFens}
                  title="Enter perturbed FEN string, for comparing activation differences"
                />
                <button
                  onClick={compareFenActivations}
                  disabled={isComparingFens || !perturbedFen.trim()}
                  className="px-3 py-1 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                  title="Compare activation differences between original FEN and perturbed FEN"
                >
                  {isComparingFens ? (
                    <>
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                      Comparing...
                    </>
                  ) : (
                    'Compare Activation Differences'
                  )}
                </button>
                {inactiveNodes.size > 0 && (
                  <span className="text-sm text-yellow-700 font-medium">
                    {inactiveNodes.size} inactive nodes
                  </span>
                )}
                <button
                  onClick={() => setShowDiffingLogs(!showDiffingLogs)}
                  className="px-2 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                  title="Show/hide comparison logs"
                >
                  {showDiffingLogs ? 'Hide Logs' : 'Show Logs'}
                </button>
              </div>
            </div>
          )}
          {/* Clerp sync controls */}
          <div className="flex items-center space-x-2 px-3 py-1 bg-blue-50 rounded-md border border-blue-200">
            <button
              onClick={syncClerpsToBackend}
              disabled={syncingToBackend || !originalCircuitJson}
              className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="Synchronize clerp of all nodes in JSON to interpretation in backend MongoDB"
            >
              {syncingToBackend ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  Uploading...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Upload Clerp
                </>
              )}
            </button>
            <button
              onClick={syncClerpsFromBackend}
              disabled={syncingFromBackend || !originalCircuitJson}
              className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="Read interpretation from backend MongoDB and synchronize to clerp of nodes in JSON"
            >
              {syncingFromBackend ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  Downloading...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                  </svg>
                  Download Clerp
                </>
              )}
            </button>
          </div>
          
          {/* Dense Feature check controls */}
          <div className="flex items-center space-x-2 px-3 py-1 bg-gray-100 rounded-md">
            <label className="text-sm text-gray-700">Dense threshold:</label>
            <input
              type="number"
              value={denseThreshold}
              onChange={(e) => setDenseThreshold(e.target.value)}
              placeholder="No limit"
              className="w-24 px-2 py-1 text-sm border border-gray-300 rounded"
              title="Activation threshold, empty means no limit (all nodes kept)"
            />
            <button
              onClick={checkDenseFeatures}
              disabled={checkingDenseFeatures}
              className="px-3 py-1 text-sm bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
              title="Check which nodes are dense feature"
            >
              {checkingDenseFeatures ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                  Checking...
                </>
              ) : (
                'Check Dense'
              )}
            </button>
            {denseNodes.size > 0 && (
              <span className="text-sm text-purple-700 font-medium">
                {denseNodes.size} dense nodes
              </span>
            )}
          </div>
          
          {/* Color-file name legend (displayed when multiple files) */}
          {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
            <div className="hidden md:flex items-center space-x-3 mr-4">
              {displayLinkGraphData.metadata.sourceFileNames.map((name, idx) => (
                <div key={idx} className="flex items-center space-x-1 text-xs">
                  <span
                    className="inline-block rounded-full"
                    style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                    title={name}
                  />
                  <span className="text-gray-600 truncate max-w-[140px]" title={name}>{name}</span>
                </div>
              ))}
            </div>
          )}

          {/* Position mapping highlight switch (displayed when multiple files) */}
          {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
            <div className="flex flex-wrap items-center gap-2 px-3 py-1 bg-purple-50 rounded-md border border-purple-200">
              <label className="text-sm text-purple-800 font-medium flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={enablePositionMapping}
                  onChange={(e) => setEnablePositionMapping(e.target.checked)}
                />
                Position mapping highlight
              </label>
              <span className="text-xs text-purple-700">
                Select one pos (0-63) for each file, highlight nodes on different pos but same (layer, feature) in different files
              </span>
              {enablePositionMapping && (
                <span className="text-xs text-purple-700">
                  Current matches: <span className="font-semibold">{positionMappedCount}</span> nodes
                </span>
              )}
            </div>
          )}
          {hasUnsavedChanges && (
            <div className="flex items-center space-x-2 px-3 py-1 bg-orange-100 text-orange-800 rounded-md text-sm">
              <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
              <span>Unsaved changes</span>
              <button
                onClick={handleQuickExport}
                className="ml-2 px-2 py-1 bg-orange-200 hover:bg-orange-300 text-orange-900 rounded text-xs transition-colors"
                title="Export all changes immediately"
              >
                Export
              </button>
            </div>
          )}
          {saveHistory.length > 0 && (
            <div className="relative group">
              <button className="px-3 py-1 text-sm bg-green-100 text-green-800 rounded hover:bg-green-200 transition-colors">
                Save History ({saveHistory.length})
              </button>
              <div className="absolute right-0 top-full mt-1 w-80 bg-white border rounded-lg shadow-lg z-10 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <div className="p-3">
                  <h4 className="font-medium text-gray-900 mb-2">Recent changes:</h4>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {saveHistory.slice(-5).reverse().map((entry, index) => (
                      <div key={index} className="text-xs text-gray-600 p-2 bg-gray-50 rounded">
                        {entry}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chess Board Display - Single file */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && fen && (
        <div className="flex justify-center gap-6 mb-6">
          {/* Circuit board state - left */}
          <div className="bg-white rounded-lg border shadow-sm p-4 pb-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-center flex-1">
                Circuit Board State
                {clickedId && displayActivationData && (
                  <span className="text-sm font-normal text-blue-600 ml-2">
                    (Node: {clickedId}{displayActivationData.nodeType ? ` - ${displayActivationData.nodeType.toUpperCase()}` : ''})
                  </span>
                )}
              </h3>
              {clickedId && (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setShowAllPositions(!showAllPositions)}
                    className={`px-3 py-1 text-sm rounded transition-colors ${
                      showAllPositions
                        ? 'bg-blue-500 text-white hover:bg-blue-600'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    title={showAllPositions ? 'Single position activation' : 'All positions activation (merged)'}
                  >
                    {showAllPositions ? 'Single Position Mode' : 'All Positions Mode'}
                  </button>
                </div>
              )}
            </div>
            {outputMove && (
              <div className="text-center mb-2 text-sm text-green-600 font-medium">
                Output move: {outputMove} 🎯
              </div>
            )}
            {clickedId && showAllPositions && loadingAllPositions && (
              <div className="text-center mb-2 text-sm text-blue-600">
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                  <span>Loading activation data from backend...</span>
                </div>
              </div>
            )}
            {clickedId && !showAllPositions && loadingBackendZPattern && (
              <div className="text-center mb-2 text-sm text-blue-600">
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                  <span>Loading feature activation from backend...</span>
                </div>
              </div>
            )}
            {clickedId && !showAllPositions && backendActivationError && (
              <div className="text-center mb-2 text-sm text-red-600">
                {backendActivationError}
              </div>
            )}
            {clickedId && displayActivationData && displayActivationData.activations && (
              <div className="text-center mb-2 text-sm text-purple-600">
                {showAllPositions ? (
                  <>
                    All positions merged activation: {displayActivationData.activations.filter((v: number) => v !== 0).length} non-zero activations
                  </>
                ) : (
                  <>
                    Activation data: {displayActivationData.activations.filter((v: number) => v !== 0).length} non-zero activations
                    {displayActivationData.zPatternIndices && displayActivationData.zPatternValues && 
                      `, ${displayActivationData.zPatternValues.length} Z pattern connections`
                    }
                  </>
                )}
              </div>
            )}
            <ChessBoard
              fen={fen}
              size="medium"
              showCoordinates={true}
              move={outputMove || undefined}
              activations={displayActivationData?.activations}
              zPatternIndices={displayActivationData?.zPatternIndices}
              zPatternValues={displayActivationData?.zPatternValues}
              flip_activation={Boolean(fen && fen.split(' ')[1] === 'b')}
              autoFlipWhenBlack={true}
              sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
              analysisName={displayActivationData?.nodeType || 'Circuit Node'}
              moveColor={(clickedId ? (displayLinkGraphData.nodes.find(n => n.nodeId === clickedId)?.nodeColor) : undefined) as any}
              showSelfPlay={true}
            />
          </div>

          {/* Custom FEN analysis - right */}
          {clickedId && (() => {
            // Parse layer and featureIndex from nodeId
            const parts = clickedId.split('_');
            const rawLayer = parseInt(parts[0]) || 0;
            const featureIdx = parseInt(parts[1]) || 0;
            const layerIdx = Math.floor(rawLayer / 2);
            
            // Determine node type and corresponding dictionary name
            const currentNode = displayLinkGraphData?.nodes.find(n => n.nodeId === clickedId);
            const isLorsa = currentNode?.feature_type?.toLowerCase() === 'lorsa';
            const dictionary = getDictionaryName(layerIdx, isLorsa);
            
            return (
              <CustomFenInput
                dictionary={dictionary}
                featureIndex={featureIdx}
                disabled={false}
              />
            );
          })()}
        </div>
      )}

      {/* Position Feature Analysis: section entry + config row + PosFeatureCard (single title inside card) */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata?.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && fen && (
        <div className="w-full max-w-6xl mx-auto mb-6">
          <h3 className="text-lg font-semibold mb-3">Position Feature Analysis</h3>
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <label className="text-sm font-medium">Layer:</label>
            <input
              type="number"
              min={0}
              max={14}
              value={posFeatureLayer}
              onChange={(e) => setPosFeatureLayer(parseInt(e.target.value) || 0)}
              className="w-20 px-2 py-1 border rounded text-sm"
            />
            <label className="text-sm font-medium">Positions:</label>
            <input
              type="text"
              placeholder="e.g. 36 or 16,20,34"
              value={posFeaturePositions}
              onChange={(e) => setPosFeaturePositions(e.target.value)}
              className="w-48 px-2 py-1 border rounded text-sm"
            />
            <label className="text-sm font-medium">Component:</label>
            <select
              value={posFeatureComponentType}
              onChange={(e) => setPosFeatureComponentType(e.target.value as "attn" | "mlp")}
              className="px-2 py-1 border rounded text-sm"
            >
              <option value="attn">Attention</option>
              <option value="mlp">MLP</option>
            </select>
          </div>
          {posFeaturePositions.trim() && (() => {
            const parsedPositions = posFeaturePositions
              .split(",")
              .map((p) => parseInt(p.trim(), 10))
              .filter((p) => !isNaN(p) && p >= 0 && p < 64);
            if (parsedPositions.length === 0) return null;
            return (
              <PosFeatureCard
                fen={fen}
                layer={posFeatureLayer}
                positions={parsedPositions}
                componentType={posFeatureComponentType}
                saeComboId={
                  typeof window !== "undefined"
                    ? window.localStorage.getItem("bt4_sae_combo_id") || undefined
                    : undefined
                }
              />
            );
          })()}
        </div>
      )}

      {/* Graph Feature Diffing Log Display - Only show when single graph */}
      {displayLinkGraphData && (!displayLinkGraphData.metadata.sourceFileNames || displayLinkGraphData.metadata.sourceFileNames.length <= 1) && showDiffingLogs && (
        <div className="w-full border rounded-lg overflow-hidden mb-6">
          <div className="bg-yellow-800 text-white px-4 py-2 flex items-center justify-between">
            <h3 className="font-semibold">FEN Activation Diff Log</h3>
            <div className="flex gap-2">
              <button
                onClick={() => setDiffingLogs([])}
                className="px-2 py-1 text-sm bg-yellow-700 hover:bg-yellow-600 text-white rounded transition-colors"
              >
                Clear Log
              </button>
              <button
                onClick={() => setShowDiffingLogs(false)}
                className="px-2 py-1 text-sm bg-yellow-700 hover:bg-yellow-600 text-white rounded transition-colors"
              >
                Hide
              </button>
            </div>
          </div>
          <div 
            id="diffing-logs-container"
            className="bg-gray-900 text-green-400 p-4 font-mono text-sm max-h-64 overflow-y-auto"
          >
            <div className="space-y-1">
              {diffingLogs.length === 0 ? (
                <div className="text-gray-500">No logs yet...</div>
              ) : (
                diffingLogs.map((log, index) => (
                  <div key={index} className="whitespace-pre-wrap">
                    {log.message}
                  </div>
                ))
              )}
              {isComparingFens && (
                <div className="text-yellow-400 animate-pulse">Comparing...</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Chess Board Display - Multiple files: Render a chess board for each source file, and display activations by source */}
      {displayLinkGraphData && displayLinkGraphData.metadata.sourceFileNames && displayLinkGraphData.metadata.sourceFileNames.length > 1 && (
        <div className="space-y-4 mb-6">
          {/* Multiple files: position mapping selector */}
          {enablePositionMapping && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
              <div className="flex flex-wrap items-center gap-3">
                <div className="text-sm font-medium text-purple-900">Position mapping selector (one per file)</div>
                <div className="text-xs text-purple-700">
                  Note: First input pos (draft) below, then click "Apply mapping" to take effect and refresh the graph (does not change node merging rules)
                </div>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  onClick={() => {
                    // Apply: Write draft to actual selection, and force refresh once
                    setPositionMappingSelections(() => {
                      const next: Record<number, number> = {};
                      const names = displayLinkGraphData?.metadata?.sourceFileNames || [];
                      for (let i = 0; i < names.length; i++) {
                        const v = draftPositionMappingSelections[i];
                        next[i] = (typeof v === "number" && Number.isFinite(v)) ? Math.max(0, Math.min(63, v)) : 0;
                      }
                      return next;
                    });
                    setPositionMappingApplyNonce((x) => x + 1);
                  }}
                  className="px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors"
                  title="Apply current input pos to highlight logic, and refresh graph"
                >
                  Apply mapping
                </button>
                <button
                  onClick={() => {
                    // Reset draft to applied value
                    setDraftPositionMappingSelections((prev) => {
                      const names = displayLinkGraphData?.metadata?.sourceFileNames || [];
                      const next: Record<number, number> = { ...prev };
                      for (let i = 0; i < names.length; i++) {
                        const applied = positionMappingSelections[i];
                        next[i] = (typeof applied === "number" && Number.isFinite(applied)) ? applied : 0;
                      }
                      return next;
                    });
                  }}
                  className="px-3 py-1 text-sm bg-white text-purple-800 border border-purple-300 rounded hover:bg-purple-100 transition-colors"
                  title="Undo unapplied modifications"
                >
                  Undo input
                </button>
                <span className="text-xs text-purple-700">
                  Applied hits: <span className="font-semibold">{positionMappedCount}</span>
                </span>
              </div>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                {displayLinkGraphData.metadata.sourceFileNames.map((name: string, idx: number) => (
                  <div key={`pos-map-${idx}`} className="flex items-center gap-3 bg-white border rounded p-2">
                    <span
                      className="inline-block rounded-full"
                      style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                      title={name}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="text-xs text-gray-600 truncate" title={name}>{name}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-gray-700">pos</label>
                      <input
                        type="number"
                        min={0}
                        max={63}
                        className="w-20 px-2 py-1 text-sm border rounded"
                        value={draftPositionMappingSelections[idx] ?? positionMappingSelections[idx] ?? 0}
                        onChange={(e) => {
                          const v = parseInt(e.target.value);
                          setDraftPositionMappingSelections((prev) => ({
                            ...prev,
                            [idx]: Number.isFinite(v) ? Math.max(0, Math.min(63, v)) : 0,
                          }));
                        }}
                        title="Choose position (0-63) for this file for alignment/highlight"
                      />
                      <span className="text-xs text-purple-700">↦ Highlight</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-xs text-purple-700">
                Highlight color: <span className="font-mono">{POSITION_MAPPING_HIGHLIGHT_COLOR}</span>
              </div>
            </div>
          )}

          {/* All positions mode switch button (multiple files) */}
          {clickedId && (
            <div className="flex justify-center">
              <button
                onClick={() => setShowAllPositions(!showAllPositions)}
                className={`px-4 py-2 text-sm rounded transition-colors ${
                  showAllPositions
                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                title={showAllPositions ? 'Show activation of single position' : 'Show activation of all positions (merged)'}
              >
                {showAllPositions ? 'Single position mode' : 'All positions mode'}
              </button>
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {multiOriginalJsons.map((entry, idx) => {
              const fileFen = extractFenFromCircuitJson(entry.json);
              if (!fileFen) return null;
              const fileMove = extractOutputMoveFromCircuitJson(entry.json);
              // Check if current selected node belongs to this file
              const currentNode = clickedId ? displayLinkGraphData.nodes.find(n => n.nodeId === clickedId) : null;
              const belongs = currentNode && (currentNode.sourceIndices?.includes(idx) || currentNode.sourceIndex === idx);
              
              // Get activation data for this file
              let perFileActivation: NodeActivationData = { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
              if (clickedId) {
                if (showAllPositions) {
                  // In all positions mode, get activation data for each file
                  const multiGraphData = multiGraphActivationData[idx];
                  if (multiGraphData) {
                    perFileActivation = multiGraphData;
                  } else {
                    // Fall back to get data from JSON file
                    const allPosData = getAllPositionsActivationDataSync(clickedId, entry.json);
                    perFileActivation = allPosData || perFileActivation;
                  }
                } else {
                  // Get activation data for single position
                  perFileActivation = getNodeActivationDataFromJson(entry.json, clickedId);
                  perFileActivation = restrictActivationDataToPosition(
                    perFileActivation,
                    parseNodeId(clickedId).ctxIdx
                  ) || perFileActivation;
                }
              }
              
              return (
                <div key={idx} className="bg-white rounded-lg border shadow-sm p-4 pb-8">
                  <h3 className="text-md font-semibold mb-3 flex items-center justify-center">
                    <span
                      className="inline-block rounded-full mr-2"
                      style={{ width: 10, height: 10, backgroundColor: UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length] }}
                      title={entry.fileName}
                    />
                    <span className="truncate" title={entry.fileName}>{entry.fileName}</span>
                    {clickedId && belongs && (
                      <span className="text-xs font-normal text-blue-600 ml-2">(Contains this node)</span>
                    )}
                  </h3>
                  {fileMove && (
                    <div className="text-center mb-2 text-sm text-green-600 font-medium">
                      Output policy: {fileMove}
                    </div>
                  )}
                  {clickedId && belongs && showAllPositions && loadingAllPositions && (
                    <div className="text-center mb-2 text-sm text-blue-600">
                      <div className="flex items-center justify-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                        <span>Getting activation data from backend...</span>
                      </div>
                    </div>
                  )}
                  {clickedId && belongs && perFileActivation.activations && (
                    <div className="text-center mb-2 text-sm text-purple-600">
                      {showAllPositions ? (
                        <>
                          All positions merged activation: {perFileActivation.activations.filter((v: number) => v !== 0).length} non-zero activations
                        </>
                      ) : (
                        <>
                          Activation data: {perFileActivation.activations.filter((v: number) => v !== 0).length} non-zero activations
                          {perFileActivation.zPatternIndices && perFileActivation.zPatternValues &&
                            `, ${perFileActivation.zPatternValues.length} Z-pattern`}
                        </>
                      )}
                    </div>
                  )}
                  <ChessBoard
                    fen={fileFen}
                    size="medium"
                    showCoordinates={true}
                    move={fileMove || undefined}
                    activations={perFileActivation.activations}
                    zPatternIndices={perFileActivation.zPatternIndices}
                    zPatternValues={perFileActivation.zPatternValues}
                    flip_activation={Boolean(fileFen && fileFen.split(' ')[1] === 'b')}
                    autoFlipWhenBlack={true}
                    sampleIndex={clickedId ? parseInt(clickedId.split('_')[1]) : undefined}
                    analysisName={(perFileActivation?.nodeType || 'Circuit Node') + ` @${idx+1}`}
                    moveColor={UNIQUE_GRAPH_COLORS[idx % UNIQUE_GRAPH_COLORS.length]}
                    showSelfPlay={true}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Circuit Visualization Layout */}
      <div className="space-y-6 w-full max-w-full overflow-hidden">
        {/* Subgraph mode control bar */}
        {clickedId && (
          <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-blue-900">Selected node:</span>
                <code className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm font-mono">
                  {clickedId}
                </code>
              </div>
              
              {!showSubgraph ? (
                <button
                  onClick={handleShowSubgraph}
                  className="px-4 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors flex items-center"
                  title="Subgraph mode with this node as root"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                  </svg>
                  Show subgraph
                </button>
              ) : (
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-2 text-sm text-green-700">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className="font-medium">Subgraph mode</span>
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                      {subgraphData?.nodes.length || 0} nodes
                    </span>
                  </div>
                  
                  <button
                    onClick={handleSaveSubgraph}
                    className="px-3 py-1 bg-green-500 text-white text-sm font-medium rounded hover:bg-green-600 transition-colors flex items-center"
                    title="Save subgraph as JSON file (includes complete activation data and z_pattern)"
                  >
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Save subgraph
                  </button>
                  
                  <button
                    onClick={handleExitSubgraph}
                    className="px-3 py-1 bg-gray-500 text-white text-sm font-medium rounded hover:bg-gray-600 transition-colors flex items-center"
                    title="Exit subgraph mode, show full graph"
                  >
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    Exit subgraph
                  </button>
                </div>
              )}
            </div>
            
            {showSubgraph && subgraphData && (
              <div className="text-xs text-gray-600 space-y-1">
                <div className="flex items-center space-x-2">
                  <span>Root node:</span>
                  <code className="px-1 bg-gray-100 rounded">{subgraphRootNodeId}</code>
                </div>
                <div className="flex items-center space-x-4">
                  <span>
                    Nodes: {subgraphData.nodes.length}/{subgraphData.metadata?.originalNodeCount || 0}
                    <span className="text-green-600 ml-1">
                      ({((subgraphData.nodes.length / (subgraphData.metadata?.originalNodeCount || 1)) * 100).toFixed(1)}%)
                    </span>
                  </span>
                  <span>
                    Edges: {subgraphData.links.length}/{subgraphData.metadata?.originalLinkCount || 0}
                    <span className="text-blue-600 ml-1">
                      ({((subgraphData.links.length / (subgraphData.metadata?.originalLinkCount || 1)) * 100).toFixed(1)}%)
                    </span>
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Top Row: Link Graph and Node Connections side by side */}
        <div className="flex gap-6 h-[700px] w-full max-w-full overflow-hidden">
          {/* Link Graph Component - Left Side */}
          <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            <div className="w-full h-full overflow-hidden relative">
              {(showSubgraph ? subgraphData : displayLinkGraphData) && (
                <LinkGraphContainer 
                  key={`${showSubgraph ? "sub" : "full"}-${positionMappingApplyNonce}`}
                  data={showSubgraph ? subgraphData : displayLinkGraphData} 
                  onNodeClick={handleFeatureClick}
                  onNodeHover={handleFeatureHover}
                  onFeatureSelect={handleFeatureSelect}
                  onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                  onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                  clickedId={clickedId}
                  hoveredId={hoveredId}
                  pinnedIds={pinnedIds}
                />
              )}
            </div>
          </div>

          {/* Node Connections Component - Right Side */}
          <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
            {(showSubgraph ? subgraphData : displayLinkGraphData) && (
              <NodeConnections
                data={showSubgraph ? subgraphData : displayLinkGraphData}
                clickedId={clickedId}
                hoveredId={hoveredId}
                pinnedIds={pinnedIds}
                hiddenIds={hiddenIds}
                onFeatureClick={handleFeatureClick}
                onFeatureSelect={handleFeatureSelect}
                onFeatureHover={handleFeatureHover}
              />
            )}
          </div>
        </div>

        {/* Top Activation Section */}
        {clickedId && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Top Activation Board</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Node: {clickedId}</span>
                {loadingTopActivations && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">Loading...</span>
                  </div>
                )}
              </div>
            </div>
            
            {loadingTopActivations ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">Getting Top Activation data...</p>
                </div>
              </div>
            ) : topActivations.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {topActivations.map((sample, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-3 border">
                    <div className="text-center mb-2">
                      <div className="text-sm font-medium text-gray-700">
                        Top #{index + 1}
                      </div>
                      <div className="text-xs text-gray-500">
                        Max Activation: {sample.activationStrength.toFixed(3)}
                      </div>
                    </div>
                    <ChessBoard
                      fen={sample.fen}
                      size="small"
                      showCoordinates={true}
                      activations={sample.activations}
                      zPatternIndices={sample.zPatternIndices}
                      zPatternValues={sample.zPatternValues}
                      sampleIndex={sample.sampleIndex}
                      analysisName={`Context ${sample.contextId}`}
                      flip_activation={Boolean(sample.fen && sample.fen.split(' ')[1] === 'b')}
                      autoFlipWhenBlack={true}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>No activation samples containing chessboard found</p>
              </div>
            )}
          </div>
        )}

        {/* Circuit Interpretation Section */}
        {clickedId && displayLinkGraphData && (() => {
          const currentNode = displayLinkGraphData.nodes.find(n => n.nodeId === clickedId);
          if (!currentNode) return null;
          
          const parts = clickedId.split('_');
          const rawLayer = parseInt(parts[0]) || 0;
          const featureIndex = parseInt(parts[1]) || 0;
          const layerIdx = Math.floor(rawLayer / 2);
          
          return (
            <CircuitInterpretationCard
              node={{
                nodeId: clickedId,
                layer: layerIdx,
                feature: featureIndex,
                feature_type: currentNode.feature_type || '',
              }}
              saeComboId={
                (typeof window !== 'undefined' 
                  ? window.localStorage.getItem("bt4_sae_combo_id") 
                  : null) || 'k_30_e_16'
              }
              saeSeries="BT4-exp128"
              getSaeName={getSaeNameForCircuit}
            />
          );
        })()}

        {/* Token Predictions Section */}
        {clickedId && (
          <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Token Predictions</h3>
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <label className="text-sm text-gray-600" htmlFor="steering-scale-input">steering_scale:</label>
                  <input
                    id="steering-scale-input"
                    type="number"
                    step={0.1}
                    className="w-24 px-2 py-1 border rounded text-sm"
                    value={steeringScaleInput}
                    onChange={(e) => {
                      const inputValue = e.target.value;
                      setSteeringScaleInput(inputValue);
                      const v = parseFloat(inputValue);
                      if (Number.isFinite(v)) {
                        setSteeringScale(v);
                      }
                    }}
                    onBlur={() => {
                      if (steeringScaleInput === '' || steeringScaleInput === '-') {
                        setSteeringScale(0);
                        setSteeringScaleInput('0');
                      }
                    }}
                    title="Adjust steering scale, supports negative input"
                  />
                </div>
                <button
                  onClick={() => clickedId && fetchTokenPredictions(clickedId)}
                  disabled={loadingTokenPredictions || !clickedId || !fen}
                  className="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                  title="Run feature intervention analysis"
                >
                  {loadingTokenPredictions ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Analyzing...
                    </>
                  ) : (
                    'Start Analysis'
                  )}
                </button>
                {loadingTokenPredictions && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">Analyzing...</span>
                  </div>
                )}
              </div>
            </div>

            {loadingTokenPredictions ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">Running feature intervention analysis...</p>
                </div>
              </div>
            ) : tokenPredictions ? (
              <div className="space-y-4">
                <div className="bg-gray-50 rounded-lg p-3 border">
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 text-sm">
                    <div>
                      <span className="text-gray-600">steering_scale:</span>
                      <span className="ml-1 font-medium">{Number(steeringScale).toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Total legal moves:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.total_legal_moves}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Average probability difference:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.avg_prob_diff?.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Average Logit difference:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.avg_logit_diff?.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Original Value:</span>
                      <span className="ml-1 font-medium">{tokenPredictions.statistics?.original_value?.toFixed(4)}</span>
                    </div>
                    <div>
                      <span className={`text-gray-600 ${tokenPredictions.statistics?.value_diff > 0 ? 'text-green-600' : tokenPredictions.statistics?.value_diff < 0 ? 'text-red-600' : ''}`}>
                        Value change:
                        <span className="ml-1 font-medium">
                          {tokenPredictions.statistics?.value_diff > 0 ? '+' : ''}{tokenPredictions.statistics?.value_diff?.toFixed(4)}
                        </span>
                      </span>
                    </div>
                  </div>
                </div>

                {tokenPredictions.promoting_moves && tokenPredictions.promoting_moves.length > 0 && (
                  <div className="bg-white rounded-lg p-3 border">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">Probability difference maximum (increase most) Top 5</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                      {tokenPredictions.promoting_moves.map((move: any, index: number) => (
                        <div key={index} className="bg-gray-50 rounded p-3 border">
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-800 mb-1">{move.uci}</div>
                            <div className="text-xs text-gray-600 space-y-1">
                              <div>Rank: #{index + 1}</div>
                              <div>Probability difference: <span className="font-medium">{formatProbability(move.prob_diff)}</span></div>
                              <div>Original probability: {formatProbability(move.original_prob)}</div>
                              <div>Modified probability: {formatProbability(move.modified_prob)}</div>
                              <div>Logit difference: {move.diff?.toFixed(4)}</div>
                              <div>Original Logit: {move.original_logit?.toFixed(4)}</div>
                              <div>Modified Logit: {move.modified_logit?.toFixed(4)}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {tokenPredictions.inhibiting_moves && tokenPredictions.inhibiting_moves.length > 0 && (
                  <div className="bg-white rounded-lg p-3 border">
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">Probability difference minimum (decrease most) Top 5</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
                      {tokenPredictions.inhibiting_moves.map((move: any, index: number) => (
                        <div key={index} className="bg-gray-50 rounded p-3 border">
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-800 mb-1">{move.uci}</div>
                            <div className="text-xs text-gray-600 space-y-1">
                              <div>Rank: #{index + 1}</div>
                              <div>Probability difference: <span className="font-medium">{formatProbability(move.prob_diff)}</span></div>
                              <div>Original probability: {formatProbability(move.original_prob)}</div>
                              <div>Modified probability: {formatProbability(move.modified_prob)}</div>
                              <div>Logit difference: {move.diff?.toFixed(4)}</div>
                              <div>Original Logit: {move.original_logit?.toFixed(4)}</div>
                              <div>Modified Logit: {move.modified_logit?.toFixed(4)}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <p>Click "Start Analysis" button to run Token Predictions analysis</p>
                <p className="text-sm mt-2">Please load TC/Lorsa combination (SaeComboLoader) above</p>
              </div>
            )}
          </div>
        )}

        {/* Bottom Row: Feature Card below Link Graph Container */}
        {clickedId && displayLinkGraphData && (() => {
          const currentNode = displayLinkGraphData.nodes.find(
            node => String(node.nodeId ?? node.id ?? "") === String(clickedId)
          );
          
          if (!currentNode) {
            return null;
          }
          
          const parseNodeId = (nodeId: string) => {
            const parts = nodeId.split('_');
            if (parts.length >= 2) {
              const rawLayer = parseInt(parts[0]) || 0;
              return {
                layerIdx: Math.floor(rawLayer / 2),
                featureIndex: parseInt(parts[1]) || 0
              };
            }
            return { layerIdx: 0, featureIndex: 0 };
          };
          
          const { layerIdx, featureIndex } = parseNodeId(currentNode.nodeId ?? currentNode.id ?? "");
          const isLorsa = currentNode.feature_type?.toLowerCase() === 'lorsa';
          
          
          const dictionary = getDictionaryName(layerIdx, isLorsa);
          
          const nodeTypeDisplay = isLorsa ? 'LORSA' : 'SAE';
          
          return (
            <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Selected Feature Details</h3>
                <div className="flex items-center space-x-4">
                  {connectedFeatures.length > 0 && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-600">Connected features:</span>
                      <span className="px-2 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
                        {connectedFeatures.length}
                      </span>
                    </div>
                  )}
                  {currentNode && featureIndex !== undefined && (
                    <Link
                      to={`/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${featureIndex}`}
                      className="inline-flex items-center px-3 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors"
                      title={`Go to L${layerIdx} ${nodeTypeDisplay} Feature #${featureIndex}`}
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                      View L{layerIdx} {nodeTypeDisplay} #{featureIndex}
                    </Link>
                  )}
                </div>
              </div>
              {selectedFeature ? (
                <FeatureCard
                  feature={selectedFeature}
                  onFeatureUpdated={handleSelectedFeatureUpdated}
                />
              ) : (
                <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
                  <div className="text-center">
                    <p className="text-gray-600">No feature is available for this node</p>
                  </div>
                </div>
              )}
            </div>
          );
        })()}
      </div>

    </div>
  );
};

export default CircuitVisualization;
