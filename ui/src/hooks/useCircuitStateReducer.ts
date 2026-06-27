/**
 * Custom hook for managing circuit visualization state using useReducer
 * Consolidates multiple useState calls into organized state slices
 */

import { useReducer, useCallback } from "react";

// State interfaces
export interface FileState {
  originalCircuitJson: any | null;
  originalFileName: string;
  multiOriginalJsons: { json: any; fileName: string }[];
  hasUnsavedChanges: boolean;
  saveHistory: string[];
}

export interface ActivationState {
  topActivations: any[];
  loadingTopActivations: boolean;
  tokenPredictions: any | null;
  loadingTokenPredictions: boolean;
  allPositionsActivationData: any | null;
  loadingAllPositions: boolean;
  multiGraphActivationData: Record<number, any | null>;
  loadingBackendZPattern: boolean;
  backendZPatternByNode: { nodeId: string; activations?: number[]; zPatternIndices?: number[][]; zPatternValues?: number[] } | null;
}

export interface DisplayState {
  showAllPositions: boolean;
  showSubgraph: boolean;
  subgraphData: any | null;
  subgraphRootNodeId: string | null;
  showDiffingLogs: boolean;
}

export interface FeatureDiffingState {
  perturbedFen: string;
  isComparingFens: boolean;
  inactiveNodes: Set<string>;
  diffingLogs: Array<{ timestamp: number; message: string }>;
}

export interface PositionMappingState {
  enablePositionMapping: boolean;
  positionMappingSelections: Record<number, number>;
  draftPositionMappingSelections: Record<number, number>;
  positionMappingApplyNonce: number;
}

export interface DenseState {
  denseNodes: Set<string>;
  denseThreshold: string;
  checkingDenseFeatures: boolean;
}

export interface SyncState {
  syncingToBackend: boolean;
  syncingFromBackend: boolean;
}

export interface ClerpState {
  editingClerp: string;
  isSaving: boolean;
  updateCounter: number;
}

export interface SteeringState {
  steeringScale: number;
  steeringScaleInput: string;
}

export interface PosFeatureState {
  posFeatureLayer: number;
  posFeaturePositions: string;
  posFeatureComponentType: "attn" | "mlp";
}

export interface CircuitState {
  file: FileState;
  activation: ActivationState;
  display: DisplayState;
  featureDiffing: FeatureDiffingState;
  positionMapping: PositionMappingState;
  dense: DenseState;
  sync: SyncState;
  clerp: ClerpState;
  steering: SteeringState;
  posFeature: PosFeatureState;
}

// Initial states
const initialFileState: FileState = {
  originalCircuitJson: null,
  originalFileName: '',
  multiOriginalJsons: [],
  hasUnsavedChanges: false,
  saveHistory: [],
};

const initialActivationState: ActivationState = {
  topActivations: [],
  loadingTopActivations: false,
  tokenPredictions: null,
  loadingTokenPredictions: false,
  allPositionsActivationData: null,
  loadingAllPositions: false,
  multiGraphActivationData: {},
  loadingBackendZPattern: false,
  backendZPatternByNode: null,
};

const initialDisplayState: DisplayState = {
  showAllPositions: false,
  showSubgraph: false,
  subgraphData: null,
  subgraphRootNodeId: null,
  showDiffingLogs: false,
};

const initialFeatureDiffingState: FeatureDiffingState = {
  perturbedFen: '',
  isComparingFens: false,
  inactiveNodes: new Set(),
  diffingLogs: [],
};

const initialPositionMappingState: PositionMappingState = {
  enablePositionMapping: false,
  positionMappingSelections: {},
  draftPositionMappingSelections: {},
  positionMappingApplyNonce: 0,
};

const initialDenseState: DenseState = {
  denseNodes: new Set(),
  denseThreshold: '',
  checkingDenseFeatures: false,
};

const initialSyncState: SyncState = {
  syncingToBackend: false,
  syncingFromBackend: false,
};

const initialClerpState: ClerpState = {
  editingClerp: '',
  isSaving: false,
  updateCounter: 0,
};

const initialSteeringState: SteeringState = {
  steeringScale: 0,
  steeringScaleInput: '0',
};

const initialPosFeatureState: PosFeatureState = {
  posFeatureLayer: 0,
  posFeaturePositions: "",
  posFeatureComponentType: "attn",
};

const initialState: CircuitState = {
  file: initialFileState,
  activation: initialActivationState,
  display: initialDisplayState,
  featureDiffing: initialFeatureDiffingState,
  positionMapping: initialPositionMappingState,
  dense: initialDenseState,
  sync: initialSyncState,
  clerp: initialClerpState,
  steering: initialSteeringState,
  posFeature: initialPosFeatureState,
};

// Action types (simplified - using string literals for now)
type CircuitAction =
  | { type: 'SET_ORIGINAL_JSON'; payload: any }
  | { type: 'SET_ORIGINAL_FILENAME'; payload: string }
  | { type: 'SET_MULTI_ORIGINAL_JSONS'; payload: { json: any; fileName: string }[] }
  | { type: 'SET_HAS_UNSAVED_CHANGES'; payload: boolean }
  | { type: 'ADD_SAVE_HISTORY'; payload: string }
  | { type: 'CLEAR_SAVE_HISTORY' }
  | { type: 'SET_TOP_ACTIVATIONS'; payload: any[] }
  | { type: 'SET_LOADING_TOP_ACTIVATIONS'; payload: boolean }
  | { type: 'SET_TOKEN_PREDICTIONS'; payload: any | null }
  | { type: 'SET_LOADING_TOKEN_PREDICTIONS'; payload: boolean }
  | { type: 'SET_ALL_POSITIONS_DATA'; payload: any | null }
  | { type: 'SET_LOADING_ALL_POSITIONS'; payload: boolean }
  | { type: 'SET_MULTI_GRAPH_DATA'; payload: Record<number, any | null> }
  | { type: 'SET_LOADING_BACKEND_Z_PATTERN'; payload: boolean }
  | { type: 'SET_BACKEND_Z_PATTERN_BY_NODE'; payload: { nodeId: string; activations?: number[]; zPatternIndices?: number[][]; zPatternValues?: number[] } | null }
  | { type: 'SET_SHOW_ALL_POSITIONS'; payload: boolean }
  | { type: 'SET_SHOW_SUBGRAPH'; payload: boolean }
  | { type: 'SET_SUBGRAPH_DATA'; payload: any | null }
  | { type: 'SET_SUBGRAPH_ROOT_NODE_ID'; payload: string | null }
  | { type: 'SET_SHOW_DIFFING_LOGS'; payload: boolean }
  | { type: 'SET_PERTURBED_FEN'; payload: string }
  | { type: 'SET_IS_COMPARING_FENS'; payload: boolean }
  | { type: 'SET_INACTIVE_NODES'; payload: Set<string> }
  | { type: 'ADD_DIFFING_LOG'; payload: string }
  | { type: 'CLEAR_DIFFING_LOGS' }
  | { type: 'SET_ENABLE_POSITION_MAPPING'; payload: boolean }
  | { type: 'SET_POSITION_MAPPING_SELECTIONS'; payload: Record<number, number> }
  | { type: 'UPDATE_POSITION_MAPPING_SELECTION'; payload: { index: number; value: number } }
  | { type: 'SET_DRAFT_POSITION_MAPPING_SELECTIONS'; payload: Record<number, number> }
  | { type: 'UPDATE_DRAFT_POSITION_MAPPING_SELECTION'; payload: { index: number; value: number } }
  | { type: 'INCREMENT_POSITION_MAPPING_APPLY_NONCE' }
  | { type: 'SET_POSITION_MAPPING_APPLY_NONCE'; payload: number }
  | { type: 'SET_DENSE_NODES'; payload: Set<string> }
  | { type: 'SET_DENSE_THRESHOLD'; payload: string }
  | { type: 'SET_CHECKING_DENSE_FEATURES'; payload: boolean }
  | { type: 'SET_SYNCING_TO_BACKEND'; payload: boolean }
  | { type: 'SET_SYNCING_FROM_BACKEND'; payload: boolean }
  | { type: 'SET_EDITING_CLERP'; payload: string }
  | { type: 'SET_IS_SAVING'; payload: boolean }
  | { type: 'INCREMENT_UPDATE_COUNTER' }
  | { type: 'SET_UPDATE_COUNTER'; payload: number }
  | { type: 'SET_STEERING_SCALE'; payload: number }
  | { type: 'SET_STEERING_SCALE_INPUT'; payload: string }
  | { type: 'SET_POS_FEATURE_LAYER'; payload: number }
  | { type: 'SET_POS_FEATURE_POSITIONS'; payload: string }
  | { type: 'SET_POS_FEATURE_COMPONENT_TYPE'; payload: "attn" | "mlp" }
  | { type: 'RESET_FILE_STATE' }
  | { type: 'RESET_ACTIVATION_STATE' }
  | { type: 'RESET_DISPLAY_STATE' }
  | { type: 'RESET_FEATURE_DIFFING_STATE' }
  | { type: 'RESET_POSITION_MAPPING_STATE' }
  | { type: 'RESET_DENSE_STATE' }
  | { type: 'RESET_CLERP_STATE' };

const circuitReducer = (state: CircuitState, action: CircuitAction): CircuitState => {
  switch (action.type) {
    // File actions
    case 'SET_ORIGINAL_JSON':
      return { ...state, file: { ...state.file, originalCircuitJson: action.payload } };
    case 'SET_ORIGINAL_FILENAME':
      return { ...state, file: { ...state.file, originalFileName: action.payload } };
    case 'SET_MULTI_ORIGINAL_JSONS':
      return { ...state, file: { ...state.file, multiOriginalJsons: action.payload } };
    case 'SET_HAS_UNSAVED_CHANGES':
      return { ...state, file: { ...state.file, hasUnsavedChanges: action.payload } };
    case 'ADD_SAVE_HISTORY':
      return { ...state, file: { ...state.file, saveHistory: [...state.file.saveHistory, action.payload] } };
    case 'CLEAR_SAVE_HISTORY':
      return { ...state, file: { ...state.file, saveHistory: [] } };
    case 'RESET_FILE_STATE':
      return { ...state, file: initialFileState };

    // Activation actions
    case 'SET_TOP_ACTIVATIONS':
      return { ...state, activation: { ...state.activation, topActivations: action.payload } };
    case 'SET_LOADING_TOP_ACTIVATIONS':
      return { ...state, activation: { ...state.activation, loadingTopActivations: action.payload } };
    case 'SET_TOKEN_PREDICTIONS':
      return { ...state, activation: { ...state.activation, tokenPredictions: action.payload } };
    case 'SET_LOADING_TOKEN_PREDICTIONS':
      return { ...state, activation: { ...state.activation, loadingTokenPredictions: action.payload } };
    case 'SET_ALL_POSITIONS_DATA':
      return { ...state, activation: { ...state.activation, allPositionsActivationData: action.payload } };
    case 'SET_LOADING_ALL_POSITIONS':
      return { ...state, activation: { ...state.activation, loadingAllPositions: action.payload } };
    case 'SET_MULTI_GRAPH_DATA':
      return { ...state, activation: { ...state.activation, multiGraphActivationData: action.payload } };
    case 'SET_LOADING_BACKEND_Z_PATTERN':
      return { ...state, activation: { ...state.activation, loadingBackendZPattern: action.payload } };
    case 'SET_BACKEND_Z_PATTERN_BY_NODE':
      return { ...state, activation: { ...state.activation, backendZPatternByNode: action.payload } };
    case 'RESET_ACTIVATION_STATE':
      return { ...state, activation: initialActivationState };

    // Display actions
    case 'SET_SHOW_ALL_POSITIONS':
      return { ...state, display: { ...state.display, showAllPositions: action.payload } };
    case 'SET_SHOW_SUBGRAPH':
      return { ...state, display: { ...state.display, showSubgraph: action.payload } };
    case 'SET_SUBGRAPH_DATA':
      return { ...state, display: { ...state.display, subgraphData: action.payload } };
    case 'SET_SUBGRAPH_ROOT_NODE_ID':
      return { ...state, display: { ...state.display, subgraphRootNodeId: action.payload } };
    case 'SET_SHOW_DIFFING_LOGS':
      return { ...state, display: { ...state.display, showDiffingLogs: action.payload } };
    case 'RESET_DISPLAY_STATE':
      return { ...state, display: initialDisplayState };

    // Feature diffing actions
    case 'SET_PERTURBED_FEN':
      return { ...state, featureDiffing: { ...state.featureDiffing, perturbedFen: action.payload } };
    case 'SET_IS_COMPARING_FENS':
      return { ...state, featureDiffing: { ...state.featureDiffing, isComparingFens: action.payload } };
    case 'SET_INACTIVE_NODES':
      return { ...state, featureDiffing: { ...state.featureDiffing, inactiveNodes: action.payload } };
    case 'ADD_DIFFING_LOG':
      return {
        ...state,
        featureDiffing: {
          ...state.featureDiffing,
          diffingLogs: [
            ...state.featureDiffing.diffingLogs,
            {
              timestamp: Date.now(),
              message: `[${new Date().toLocaleTimeString()}] ${action.payload}`
            }
          ]
        }
      };
    case 'CLEAR_DIFFING_LOGS':
      return { ...state, featureDiffing: { ...state.featureDiffing, diffingLogs: [] } };
    case 'RESET_FEATURE_DIFFING_STATE':
      return { ...state, featureDiffing: initialFeatureDiffingState };

    // Position mapping actions
    case 'SET_ENABLE_POSITION_MAPPING':
      return { ...state, positionMapping: { ...state.positionMapping, enablePositionMapping: action.payload } };
    case 'SET_POSITION_MAPPING_SELECTIONS':
      return { ...state, positionMapping: { ...state.positionMapping, positionMappingSelections: action.payload } };
    case 'UPDATE_POSITION_MAPPING_SELECTION':
      return {
        ...state,
        positionMapping: {
          ...state.positionMapping,
          positionMappingSelections: {
            ...state.positionMapping.positionMappingSelections,
            [action.payload.index]: action.payload.value
          }
        }
      };
    case 'SET_DRAFT_POSITION_MAPPING_SELECTIONS':
      return { ...state, positionMapping: { ...state.positionMapping, draftPositionMappingSelections: action.payload } };
    case 'UPDATE_DRAFT_POSITION_MAPPING_SELECTION':
      return {
        ...state,
        positionMapping: {
          ...state.positionMapping,
          draftPositionMappingSelections: {
            ...state.positionMapping.draftPositionMappingSelections,
            [action.payload.index]: action.payload.value
          }
        }
      };
    case 'INCREMENT_POSITION_MAPPING_APPLY_NONCE':
      return {
        ...state,
        positionMapping: {
          ...state.positionMapping,
          positionMappingApplyNonce: state.positionMapping.positionMappingApplyNonce + 1
        }
      };
    case 'SET_POSITION_MAPPING_APPLY_NONCE':
      return {
        ...state,
        positionMapping: {
          ...state.positionMapping,
          positionMappingApplyNonce: action.payload
        }
      };
    case 'RESET_POSITION_MAPPING_STATE':
      return { ...state, positionMapping: initialPositionMappingState };

    // Dense actions
    case 'SET_DENSE_NODES':
      return { ...state, dense: { ...state.dense, denseNodes: action.payload } };
    case 'SET_DENSE_THRESHOLD':
      return { ...state, dense: { ...state.dense, denseThreshold: action.payload } };
    case 'SET_CHECKING_DENSE_FEATURES':
      return { ...state, dense: { ...state.dense, checkingDenseFeatures: action.payload } };
    case 'RESET_DENSE_STATE':
      return { ...state, dense: initialDenseState };

    // Sync actions
    case 'SET_SYNCING_TO_BACKEND':
      return { ...state, sync: { ...state.sync, syncingToBackend: action.payload } };
    case 'SET_SYNCING_FROM_BACKEND':
      return { ...state, sync: { ...state.sync, syncingFromBackend: action.payload } };

    // Clerp actions
    case 'SET_EDITING_CLERP':
      return { ...state, clerp: { ...state.clerp, editingClerp: action.payload } };
    case 'SET_IS_SAVING':
      return { ...state, clerp: { ...state.clerp, isSaving: action.payload } };
    case 'INCREMENT_UPDATE_COUNTER':
      return { ...state, clerp: { ...state.clerp, updateCounter: state.clerp.updateCounter + 1 } };
    case 'SET_UPDATE_COUNTER':
      return { ...state, clerp: { ...state.clerp, updateCounter: action.payload } };
    case 'RESET_CLERP_STATE':
      return { ...state, clerp: initialClerpState };

    // Steering actions
    case 'SET_STEERING_SCALE':
      return { ...state, steering: { ...state.steering, steeringScale: action.payload } };
    case 'SET_STEERING_SCALE_INPUT':
      return { ...state, steering: { ...state.steering, steeringScaleInput: action.payload } };

    // PosFeature actions
    case 'SET_POS_FEATURE_LAYER':
      return { ...state, posFeature: { ...state.posFeature, posFeatureLayer: action.payload } };
    case 'SET_POS_FEATURE_POSITIONS':
      return { ...state, posFeature: { ...state.posFeature, posFeaturePositions: action.payload } };
    case 'SET_POS_FEATURE_COMPONENT_TYPE':
      return { ...state, posFeature: { ...state.posFeature, posFeatureComponentType: action.payload } };

    default:
      return state;
  }
};

/**
 * Custom hook for circuit state management
 */
export const useCircuitStateReducer = () => {
  const [state, dispatch] = useReducer(circuitReducer, initialState);

  // Action creators
  const actions = {
    file: {
      setOriginalJson: useCallback((json: any) => {
        dispatch({ type: 'SET_ORIGINAL_JSON', payload: json });
      }, []),
      setOriginalFileName: useCallback((fileName: string) => {
        dispatch({ type: 'SET_ORIGINAL_FILENAME', payload: fileName });
      }, []),
      setMultiOriginalJsons: useCallback((jsons: { json: any; fileName: string }[]) => {
        dispatch({ type: 'SET_MULTI_ORIGINAL_JSONS', payload: jsons });
      }, []),
      setHasUnsavedChanges: useCallback((hasChanges: boolean) => {
        dispatch({ type: 'SET_HAS_UNSAVED_CHANGES', payload: hasChanges });
      }, []),
      addSaveHistory: useCallback((entry: string) => {
        dispatch({ type: 'ADD_SAVE_HISTORY', payload: entry });
      }, []),
      clearSaveHistory: useCallback(() => {
        dispatch({ type: 'CLEAR_SAVE_HISTORY' });
      }, []),
      resetFileState: useCallback(() => {
        dispatch({ type: 'RESET_FILE_STATE' });
      }, []),
    },
    activation: {
      setTopActivations: useCallback((activations: any[]) => {
        dispatch({ type: 'SET_TOP_ACTIVATIONS', payload: activations });
      }, []),
      setLoadingTopActivations: useCallback((loading: boolean) => {
        dispatch({ type: 'SET_LOADING_TOP_ACTIVATIONS', payload: loading });
      }, []),
      setTokenPredictions: useCallback((predictions: any | null) => {
        dispatch({ type: 'SET_TOKEN_PREDICTIONS', payload: predictions });
      }, []),
      setLoadingTokenPredictions: useCallback((loading: boolean) => {
        dispatch({ type: 'SET_LOADING_TOKEN_PREDICTIONS', payload: loading });
      }, []),
      setAllPositionsData: useCallback((data: any | null) => {
        dispatch({ type: 'SET_ALL_POSITIONS_DATA', payload: data });
      }, []),
      setLoadingAllPositions: useCallback((loading: boolean) => {
        dispatch({ type: 'SET_LOADING_ALL_POSITIONS', payload: loading });
      }, []),
      setMultiGraphData: useCallback((data: Record<number, any | null>) => {
        dispatch({ type: 'SET_MULTI_GRAPH_DATA', payload: data });
      }, []),
      setLoadingBackendZPattern: useCallback((loading: boolean) => {
        dispatch({ type: 'SET_LOADING_BACKEND_Z_PATTERN', payload: loading });
      }, []),
      setBackendZPatternByNode: useCallback((data: { nodeId: string; activations?: number[]; zPatternIndices?: number[][]; zPatternValues?: number[] } | null) => {
        dispatch({ type: 'SET_BACKEND_Z_PATTERN_BY_NODE', payload: data });
      }, []),
      resetActivationState: useCallback(() => {
        dispatch({ type: 'RESET_ACTIVATION_STATE' });
      }, []),
    },
    display: {
      setShowAllPositions: useCallback((show: boolean) => {
        dispatch({ type: 'SET_SHOW_ALL_POSITIONS', payload: show });
      }, []),
      setShowSubgraph: useCallback((show: boolean) => {
        dispatch({ type: 'SET_SHOW_SUBGRAPH', payload: show });
      }, []),
      setSubgraphData: useCallback((data: any | null) => {
        dispatch({ type: 'SET_SUBGRAPH_DATA', payload: data });
      }, []),
      setSubgraphRootNodeId: useCallback((nodeId: string | null) => {
        dispatch({ type: 'SET_SUBGRAPH_ROOT_NODE_ID', payload: nodeId });
      }, []),
      setShowDiffingLogs: useCallback((show: boolean) => {
        dispatch({ type: 'SET_SHOW_DIFFING_LOGS', payload: show });
      }, []),
      resetDisplayState: useCallback(() => {
        dispatch({ type: 'RESET_DISPLAY_STATE' });
      }, []),
    },
    featureDiffing: {
      setPerturbedFen: useCallback((fen: string) => {
        dispatch({ type: 'SET_PERTURBED_FEN', payload: fen });
      }, []),
      setIsComparingFens: useCallback((comparing: boolean) => {
        dispatch({ type: 'SET_IS_COMPARING_FENS', payload: comparing });
      }, []),
      setInactiveNodes: useCallback((nodes: Set<string>) => {
        dispatch({ type: 'SET_INACTIVE_NODES', payload: nodes });
      }, []),
      addDiffingLog: useCallback((message: string) => {
        dispatch({ type: 'ADD_DIFFING_LOG', payload: message });
      }, []),
      clearDiffingLogs: useCallback(() => {
        dispatch({ type: 'CLEAR_DIFFING_LOGS' });
      }, []),
      resetFeatureDiffingState: useCallback(() => {
        dispatch({ type: 'RESET_FEATURE_DIFFING_STATE' });
      }, []),
    },
    positionMapping: {
      setEnablePositionMapping: useCallback((enable: boolean) => {
        dispatch({ type: 'SET_ENABLE_POSITION_MAPPING', payload: enable });
      }, []),
      setPositionMappingSelections: useCallback((selections: Record<number, number>) => {
        dispatch({ type: 'SET_POSITION_MAPPING_SELECTIONS', payload: selections });
      }, []),
      updatePositionMappingSelection: useCallback((index: number, value: number) => {
        dispatch({ type: 'UPDATE_POSITION_MAPPING_SELECTION', payload: { index, value } });
      }, []),
      setDraftPositionMappingSelections: useCallback((selections: Record<number, number>) => {
        dispatch({ type: 'SET_DRAFT_POSITION_MAPPING_SELECTIONS', payload: selections });
      }, []),
      updateDraftPositionMappingSelection: useCallback((index: number, value: number) => {
        dispatch({ type: 'UPDATE_DRAFT_POSITION_MAPPING_SELECTION', payload: { index, value } });
      }, []),
      incrementPositionMappingApplyNonce: useCallback(() => {
        dispatch({ type: 'INCREMENT_POSITION_MAPPING_APPLY_NONCE' });
      }, []),
      setPositionMappingApplyNonce: useCallback((nonce: number) => {
        dispatch({ type: 'SET_POSITION_MAPPING_APPLY_NONCE', payload: nonce });
      }, []),
      resetPositionMappingState: useCallback(() => {
        dispatch({ type: 'RESET_POSITION_MAPPING_STATE' });
      }, []),
    },
    dense: {
      setDenseNodes: useCallback((nodes: Set<string>) => {
        dispatch({ type: 'SET_DENSE_NODES', payload: nodes });
      }, []),
      setDenseThreshold: useCallback((threshold: string) => {
        dispatch({ type: 'SET_DENSE_THRESHOLD', payload: threshold });
      }, []),
      setCheckingDenseFeatures: useCallback((checking: boolean) => {
        dispatch({ type: 'SET_CHECKING_DENSE_FEATURES', payload: checking });
      }, []),
      resetDenseState: useCallback(() => {
        dispatch({ type: 'RESET_DENSE_STATE' });
      }, []),
    },
    sync: {
      setSyncingToBackend: useCallback((syncing: boolean) => {
        dispatch({ type: 'SET_SYNCING_TO_BACKEND', payload: syncing });
      }, []),
      setSyncingFromBackend: useCallback((syncing: boolean) => {
        dispatch({ type: 'SET_SYNCING_FROM_BACKEND', payload: syncing });
      }, []),
    },
    clerp: {
      setEditingClerp: useCallback((clerp: string) => {
        dispatch({ type: 'SET_EDITING_CLERP', payload: clerp });
      }, []),
      setIsSaving: useCallback((saving: boolean) => {
        dispatch({ type: 'SET_IS_SAVING', payload: saving });
      }, []),
      incrementUpdateCounter: useCallback(() => {
        dispatch({ type: 'INCREMENT_UPDATE_COUNTER' });
      }, []),
      setUpdateCounter: useCallback((counter: number) => {
        dispatch({ type: 'SET_UPDATE_COUNTER', payload: counter });
      }, []),
      resetClerpState: useCallback(() => {
        dispatch({ type: 'RESET_CLERP_STATE' });
      }, []),
    },
    steering: {
      setSteeringScale: useCallback((scale: number) => {
        dispatch({ type: 'SET_STEERING_SCALE', payload: scale });
      }, []),
      setSteeringScaleInput: useCallback((input: string) => {
        dispatch({ type: 'SET_STEERING_SCALE_INPUT', payload: input });
      }, []),
    },
    posFeature: {
      setPosFeatureLayer: useCallback((layer: number) => {
        dispatch({ type: 'SET_POS_FEATURE_LAYER', payload: layer });
      }, []),
      setPosFeaturePositions: useCallback((positions: string) => {
        dispatch({ type: 'SET_POS_FEATURE_POSITIONS', payload: positions });
      }, []),
      setPosFeatureComponentType: useCallback((type: "attn" | "mlp") => {
        dispatch({ type: 'SET_POS_FEATURE_COMPONENT_TYPE', payload: type });
      }, []),
    },
  };

  return {
    state,
    actions,
  };
};
