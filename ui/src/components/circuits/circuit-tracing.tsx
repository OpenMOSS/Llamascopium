import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2, Settings, ExternalLink } from 'lucide-react';
import { LinkGraphContainer } from './link-graph-container';
import { NodeConnections } from './node-connections';
import { FeatureCard } from '@/components/feature/feature-card';
import { ChessBoard } from '@/components/chess/chess-board';
import { Feature } from '@/types/feature';
import { transformCircuitData } from './link-graph/utils';
import { CIRCUIT_TRACE_DEFAULTS } from '@/config/circuit-trace-defaults';

interface CircuitTracingProps {
  gameFen: string; // FEN before move
  previousFen?: string | null; // Previous FEN state
  currentFen?: string; // Current FEN state
  gameHistory: string[];
  lastMove?: string | null; // Last move
  onCircuitTraceStart?: () => void;
  onCircuitTraceEnd?: () => void;
  isTracing?: boolean;
}

export const CircuitTracing: React.FC<CircuitTracingProps> = ({
  gameFen,
  previousFen: _previousFen,
  currentFen,
  gameHistory,
  lastMove,
  onCircuitTraceStart,
  onCircuitTraceEnd,
  isTracing = false,
}) => {
  const navigate = useNavigate();
  const [circuitTraceResult, setCircuitTraceResult] = useState<any>(null);
  const [circuitVisualizationData, setCircuitVisualizationData] = useState<any>(null);
  const [clickedNodeId, setClickedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [pinnedNodeIds, setPinnedNodeIds] = useState<string[]>([]);
  const [hiddenNodeIds, setHiddenNodeIds] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [connectedFeatures, setConnectedFeatures] = useState<Feature[]>([]);
  const [, setIsLoadingConnectedFeatures] = useState(false);

  const [showParamsDialog, setShowParamsDialog] = useState(false);
  const [circuitParams, setCircuitParams] = useState({
    max_feature_nodes: CIRCUIT_TRACE_DEFAULTS.max_feature_nodes,
    node_threshold: CIRCUIT_TRACE_DEFAULTS.node_threshold,
    edge_threshold: CIRCUIT_TRACE_DEFAULTS.edge_threshold,
    max_act_times: null as number | null,
  });

  const [positiveMove, setPositiveMove] = useState<string>('');
  const [negativeMove, setNegativeMove] = useState<string>('');
  const [moveError, setMoveError] = useState<string>('');
  const [traceSide, setTraceSide] = useState<'q' | 'k' | 'both'>('both');
  
  const traceModel = 'BT4';

  const [topActivations, setTopActivations] = useState<any[]>([]);
  const [loadingTopActivations, setLoadingTopActivations] = useState(false);

  const [traceLogs, setTraceLogs] = useState<Array<{timestamp: number; message: string}>>([]);
  const MAX_VISIBLE_LOGS = 100;
  const effectiveGameFen = gameFen;

  const safeDecodeFen = useCallback((fen: string): string => {
    try {
      return decodeURIComponent(fen);
    } catch {
      return fen;
    }
  }, []);

  const MOVE_CACHE_KEY = 'circuit_move_by_fen_v1';
  const POSITIVE_MOVE_CACHE_KEY = 'circuit_positive_move_by_fen_v1';
  const NEGATIVE_MOVE_CACHE_KEY = 'circuit_negative_move_by_fen_v1';
  
  const loadCachedMove = useCallback((fen: string): string => {
    try {
      const raw = localStorage.getItem(MOVE_CACHE_KEY);
      if (!raw) return '';
      const obj = JSON.parse(raw) as Record<string, string>;
      return obj[fen] || '';
    } catch {
      return '';
    }
  }, []);

  const loadCachedPositiveMove = useCallback((fen: string): string => {
    try {
      const raw = localStorage.getItem(POSITIVE_MOVE_CACHE_KEY);
      if (!raw) return '';
      const obj = JSON.parse(raw) as Record<string, string>;
      return obj[fen] || '';
    } catch {
      return '';
    }
  }, []);
  
  const loadCachedNegativeMove = useCallback((fen: string): string => {
    try {
      const raw = localStorage.getItem(NEGATIVE_MOVE_CACHE_KEY);
      if (!raw) return '';
      const obj = JSON.parse(raw) as Record<string, string>;
      return obj[fen] || '';
    } catch {
      return '';
    }
  }, []);
  
  const saveCachedPositiveMove = useCallback((fen: string, move: string) => {
    try {
      const raw = localStorage.getItem(POSITIVE_MOVE_CACHE_KEY);
      const obj = raw ? (JSON.parse(raw) as Record<string, string>) : {};
      obj[fen] = move;
      localStorage.setItem(POSITIVE_MOVE_CACHE_KEY, JSON.stringify(obj));
    } catch {
      /* no-op */
    }
  }, []);
  
  const saveCachedNegativeMove = useCallback((fen: string, move: string) => {
    try {
      const raw = localStorage.getItem(NEGATIVE_MOVE_CACHE_KEY);
      const obj = raw ? (JSON.parse(raw) as Record<string, string>) : {};
      obj[fen] = move;
      localStorage.setItem(NEGATIVE_MOVE_CACHE_KEY, JSON.stringify(obj));
    } catch {
      /* no-op */
    }
  }, []);

  // Node activation data interface
  interface NodeActivationData {
    activations?: number[];
    zPatternIndices?: any;
    zPatternValues?: number[];
    nodeType?: string;
    clerp?: string;
  }

  const handleCircuitTraceResult = useCallback((result: any) => {
    if (result && result.nodes) {
      try {
        const transformedData = transformCircuitData(result);
        setCircuitVisualizationData(transformedData);
        setCircuitTraceResult(result);
      } catch (error) {
        console.error('Circuit data conversion failed:', error);
        alert('Circuit data conversion failed: ' + (error instanceof Error ? error.message : 'Unknown error'));
      }
    }
  }, []);

  // New: handle node click - fix parameter passing
  const handleNodeClick = useCallback((node: any, isMetaKey: boolean) => {
    const nodeId = node.nodeId || node.id;
    console.log('🔍 Node clicked:', { nodeId, isMetaKey, currentClickedId: clickedNodeId, node });
    
    if (isMetaKey) {
      // Toggle pinned state
      const newPinnedIds = pinnedNodeIds.includes(nodeId)
        ? pinnedNodeIds.filter(id => id !== nodeId)
        : [...pinnedNodeIds, nodeId];
      setPinnedNodeIds(newPinnedIds);
      console.log('Toggle pinned state:', newPinnedIds);
    } else {
      // Set clicked node
      const newClickedId = nodeId === clickedNodeId ? null : nodeId;
      setClickedNodeId(newClickedId);
      console.log('Set clicked node:', newClickedId);
      
      // Clear previous feature selection
      if (newClickedId === null) {
        setSelectedFeature(null);
        setConnectedFeatures([]);
      }
    }
  }, [clickedNodeId, pinnedNodeIds]);

  // New: handle node hover - fix parameter passing
  const handleNodeHover = useCallback((nodeId: string | null) => {
    if (nodeId !== hoveredNodeId) {
      setHoveredNodeId(nodeId);
    }
  }, [hoveredNodeId]);

  // New: handle feature select
  const handleFeatureSelect = useCallback((feature: Feature | null) => {
    setSelectedFeature(feature);
  }, []);

  // New: handle connected features select
  const handleConnectedFeaturesSelect = useCallback((features: Feature[]) => {
    setConnectedFeatures(features);
    setIsLoadingConnectedFeatures(false);
  }, []);

  // New: handle connected features loading
  const handleConnectedFeaturesLoading = useCallback((loading: boolean) => {
    setIsLoadingConnectedFeatures(loading);
  }, []);

  const validateMove = useCallback((move: string, _fen: string): boolean => {
    try {
      if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move)) {
        setMoveError('Move format is incorrect, should be UCI format (e.g. e2e4)');
        return false;
      }
      setMoveError('');
      return true;
    } catch (_error) {
      setMoveError('Move validation failed');
      return false;
    }
  }, []);

  // Start one trace request and use its response as the result.
  const handleCircuitTrace = useCallback(async (orderMode: 'positive' | 'negative' | 'both' = 'positive') => {
    let moveUci: string | null = null;
    const decodedFen = safeDecodeFen(effectiveGameFen);
    const lastMoveStr: string | null = lastMove ? lastMove : null;
    
    // Debug log: record all relevant state
    console.log('🔍 [DEBUG] handleCircuitTrace called with:', {
      orderMode,
      positiveMove: positiveMove,
      negativeMove: negativeMove,
      lastMove: lastMove,
      gameFen: gameFen,
      cachedPositive: loadCachedPositiveMove(gameFen),
      cachedNegative: loadCachedNegativeMove(gameFen),
      cachedMove: loadCachedMove(gameFen),
    });
    
    if (orderMode === 'both') {
      // Both Trace: requires both a positive move and a negative move
      const posMove = positiveMove.trim() || loadCachedPositiveMove(decodedFen);
      const negMove = negativeMove.trim() || loadCachedNegativeMove(decodedFen);
      
      if (!posMove) {
        alert('Both Trace requires a Positive Move.');
        return;
      }
      if (!negMove) {
        alert('Both Trace requires a Negative Move.');
        return;
      }
      
      // Validate both moves
      if (!validateMove(posMove, gameFen)) {
        setMoveError('Positive Move format is incorrect.');
        return;
      }
      if (!validateMove(negMove, gameFen)) {
        setMoveError('Negative Move format is incorrect.');
        return;
      }
      
      // For "both" trace, use the positive move as the primary move; the negative move is passed via order_mode
      moveUci = posMove;
      
      console.log('🔍 Both Circuit Trace parameters:', {
        fen: gameFen,
        positive_move: posMove,
        negative_move: negMove,
        side: 'both',
        order_mode: 'both',
        trace_model: traceModel
      });
    } else {
      // Positive/Negative Trace: use the corresponding move
      if (orderMode === 'positive') {
        const trimmedPositive = positiveMove.trim();
        const cachedPos = loadCachedPositiveMove(decodedFen);
        const cached = loadCachedMove(decodedFen);
        
        // Prefer user input; fall back to cache or lastMove only when user input is empty
        moveUci = trimmedPositive || cachedPos || cached || lastMoveStr;
        
        // Debug log: record selection process
        console.log('🔍 [DEBUG] Positive Trace move selection:', {
          'User input (positiveMove.trim())': trimmedPositive || '(empty)',
          'Cached Positive Move': cachedPos || '(none)',
          'Cached Move': cached || '(none)',
          'Last move': lastMoveStr || '(none)',
          'Final choice': moveUci,
        });
      } else {
        const trimmedNegative = negativeMove.trim();
        const cachedNeg = loadCachedNegativeMove(decodedFen);
        const cached = loadCachedMove(decodedFen);
        
        moveUci = trimmedNegative || cachedNeg || cached || lastMoveStr;
        
        // Debug log: record selection process
        console.log('🔍 [DEBUG] Negative Trace move selection:', {
          'User input (negativeMove.trim())': trimmedNegative || '(empty)',
          'Cached Negative Move': cachedNeg || '(none)',
          'Cached Move': cached || '(none)',
          'Last move': lastMoveStr || '(none)',
          'Final choice': moveUci,
        });
      }
      
      if (!moveUci) {
        alert(`Please enter a ${orderMode === 'positive' ? 'Positive' : 'Negative'} Move or play a move first.`);
        return;
      }
      
      // Validate move format
      if (!validateMove(moveUci, gameFen)) {
        return;
      }
      
      console.log('🔍 Circuit Trace parameters:', {
        fen: decodedFen,
        move_uci: moveUci,
        order_mode: orderMode,
        side: traceSide,
        trace_model: 'BT4'  // Always use the BT4 model
      });
    }
    
    onCircuitTraceStart?.();
    setTraceLogs([]);

    let logPollTimer: number | undefined;
    let pollLogs: (() => Promise<void>) | undefined;
    try {
      // Get the currently selected SAE combo ID (read from localStorage, consistent with SaeComboLoader)
      const LOCAL_STORAGE_KEY = "bt4_sae_combo_id";
      const currentSaeComboId = window.localStorage.getItem(LOCAL_STORAGE_KEY) || null;
      
      // Build request payload
      const requestBody: any = { 
        fen: decodedFen,
        move_uci: moveUci,
        side: orderMode === 'both' ? 'both' : traceSide,
        order_mode: orderMode,
        max_feature_nodes: circuitParams.max_feature_nodes,
        node_threshold: circuitParams.node_threshold,
        edge_threshold: circuitParams.edge_threshold,
        max_act_times: circuitParams.max_act_times,
        save_activation_info: CIRCUIT_TRACE_DEFAULTS.save_activation_info
      };
      
      // If the frontend has a selected SAE combo ID, pass it to the backend
      if (currentSaeComboId) {
        requestBody.sae_combo_id = currentSaeComboId;
      }
      
      // For "both" trace, also send the negative move
      if (orderMode === 'both') {
        const negMove = negativeMove.trim() || loadCachedNegativeMove(decodedFen);
        if (negMove) {
          requestBody.negative_move_uci = negMove;
        }
      }
      
      // Debug log: record the actual request payload
      console.log('🔍 [DEBUG] Sending Circuit Trace request:', {
        requestBody,
        'move_uci used': requestBody.move_uci,
        'user positiveMove input': positiveMove,
        'user negativeMove input': negativeMove,
        'current SAE combo ID': currentSaeComboId,
      });

      const logParams = new URLSearchParams({
        model_name: 'lc0/BT4-1024x15x32h',
        fen: decodedFen,
        move_uci: requestBody.move_uci,
      });
      if (currentSaeComboId) logParams.set('sae_combo_id', currentSaeComboId);
      pollLogs = async () => {
        try {
          const response = await fetch(
            `${import.meta.env.VITE_BACKEND_URL}/circuit_trace/logs?${logParams.toString()}`,
          );
          if (!response.ok) return;
          const data = await response.json();
          setTraceLogs((data.logs ?? []).slice(-MAX_VISIBLE_LOGS));
        } catch (error) {
          console.error('Failed to fetch circuit trace logs:', error);
        }
      };

      // Start the trace first. Polling is scoped to this exact request and ends
      // as soon as the synchronous trace response completes.
      const traceRequest = fetch(`${import.meta.env.VITE_BACKEND_URL}/circuit_trace`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      logPollTimer = window.setInterval(() => void pollLogs?.(), 1000);
      const response = await traceRequest;
      await pollLogs();

      if (response.ok) {
        const data = await response.json();
        // Cache moves on success
        if (orderMode === 'both' || orderMode === 'positive') {
          const posMove = positiveMove.trim() || loadCachedPositiveMove(decodedFen);
          if (posMove) {
            saveCachedPositiveMove(decodedFen, posMove);
          }
        }
        if (orderMode === 'both' || orderMode === 'negative') {
          const negMove = negativeMove.trim() || loadCachedNegativeMove(decodedFen);
          if (negMove) {
            saveCachedNegativeMove(decodedFen, negMove);
          }
        }
        
        // Backend has already set correct metadata based on sae_combo_id from constants.py,
        // so we use the metadata as returned without overriding it
        if (data.metadata) {
          console.log('🔍 Metadata returned by backend:', {
            lorsa_analysis_name: data.metadata.lorsa_analysis_name,
            tc_analysis_name: data.metadata.tc_analysis_name,
            sae_combo_id: currentSaeComboId,
          });
        }
        
        handleCircuitTraceResult(data);
      } else {
        const errorText = await response.text();
        console.error('Circuit trace API call failed:', response.status, response.statusText, errorText);
        alert('Circuit trace failed: ' + errorText);
      }
    } catch (error) {
      console.error('Circuit trace error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert('Circuit trace failed: ' + errorMessage);
    } finally {
      if (logPollTimer !== undefined) window.clearInterval(logPollTimer);
      onCircuitTraceEnd?.();
    }
  }, [effectiveGameFen, gameFen, lastMove, positiveMove, negativeMove, validateMove, onCircuitTraceStart, onCircuitTraceEnd, handleCircuitTraceResult, circuitParams, traceSide, loadCachedMove, loadCachedPositiveMove, loadCachedNegativeMove, saveCachedPositiveMove, saveCachedNegativeMove, safeDecodeFen]);

  // Save raw graph JSON (same structure as backend create_graph_files)
  const handleSaveGraphJson = useCallback(() => {
    try {
      const raw = circuitTraceResult || circuitVisualizationData;
      if (!raw) {
        alert('No graph data available to save');
        return;
      }
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const slug = raw?.metadata?.slug || 'circuit_trace';
      // Parse fullmove number (6th FEN field); if parsing fails, fall back to an estimate based on move history length
      const fenParts = effectiveGameFen.split(' ');
      const fullmove = fenParts.length >= 6 && !Number.isNaN(parseInt(fenParts[5]))
        ? parseInt(fenParts[5])
        : Math.max(1, Math.ceil(gameHistory.length / 2));
      const fileName = `${slug}_m${fullmove}_${timestamp}.json`;
      const jsonString = JSON.stringify(raw, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to save JSON:', error);
      alert('Failed to save JSON');
    }
  }, [circuitTraceResult, circuitVisualizationData, effectiveGameFen, gameHistory]);

  // Handle parameter changes
  const handleParamsChange = useCallback((key: keyof typeof circuitParams, value: string) => {
    setCircuitParams(prev => ({
      ...prev,
      [key]: key === 'max_feature_nodes' ? parseInt(value) || 1024 : 
              key === 'max_act_times' ? (() => {
                if (value === '') return null;
                const num = parseInt(value);
                if (isNaN(num)) return null;
                // Clamp to the 10M–100M range and snap to 10M steps
                const clamped = Math.max(10000000, Math.min(100000000, num));
                // Round to the nearest 10M
                return Math.round(clamped / 10000000) * 10000000;
              })() :
              parseFloat(value) || prev[key]
    }));
  }, []);

  const handleSaveParams = useCallback(() => {
    setShowParamsDialog(false);
  }, []);

  // Fetch Top Activation data for a node
  const fetchTopActivations = useCallback(async (nodeId: string) => {
    if (!nodeId) return;
    
    setLoadingTopActivations(true);
    try {
      // Parse feature information from nodeId
      const parts = nodeId.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureIndex = Number(parts[1]) || 0;
      const layerIdx = Math.floor(rawLayer / 2);
      
      // Determine node type and corresponding dictionary name
      const currentNode = circuitVisualizationData?.nodes.find((n: any) => n.nodeId === nodeId);
      const isLorsa = currentNode?.feature_type?.toLowerCase() === 'lorsa';
      
      // Use metadata to determine dictionary name
      let dictionary: string;
      if (isLorsa) {
        const lorsaAnalysisName = circuitVisualizationData?.metadata?.lorsa_analysis_name;
        if (lorsaAnalysisName && lorsaAnalysisName.includes('BT4')) {
          // BT4 format: BT4_lorsa_L{layer}A
          dictionary = `BT4_lorsa_L${layerIdx}A`;
        } else {
          dictionary = lorsaAnalysisName ? lorsaAnalysisName.replace("{}", layerIdx.toString()) : `lc0-lorsa-L${layerIdx}`;
        }
      } else {
        const tcAnalysisName = (circuitVisualizationData?.metadata as any)?.tc_analysis_name || circuitVisualizationData?.metadata?.clt_analysis_name;
        if (tcAnalysisName && tcAnalysisName.includes('BT4')) {
          // BT4 format: BT4_tc_L{layer}M
          dictionary = `BT4_tc_L${layerIdx}M`;
        } else {
          dictionary = tcAnalysisName ? tcAnalysisName.replace("{}", layerIdx.toString()) : `lc0_L${layerIdx}M_16x_k30_lr2e-03_auxk_sparseadam`;
        }
      }
      
      console.log('🔍 Fetching Top Activation data:', {
        nodeId,
        layerIdx,
        featureIndex,
        dictionary,
        isLorsa
      });
      
      // Call backend API to get feature data
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
      
      // Parse data
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
      
      // Find samples containing FEN and extract activation values
      const chessSamples: any[] = [];
      
      for (const sample of allSamples) {
        if (sample.text) {
          const lines = sample.text.split('\n');
          
          for (const line of lines) {
            const trimmed = line.trim();
            
            // Check if it contains FEN format
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
                    // Process sparse activation data - correctly mapped to 64-square chessboard
                    let activationsArray: number[] | undefined = undefined;
                    let maxActivation = 0; // Use maximum activation value instead of total sum
                    
                    if (sample.featureActsIndices && sample.featureActsValues && 
                        Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
                      
                      // Create activation array for 64 squares
                      activationsArray = new Array(64).fill(0);
                      
                      // Map sparse activation values to the correct chessboard positions and find the maximum activation value
                      for (let i = 0; i < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); i++) {
                        const index = sample.featureActsIndices[i];
                        const value = sample.featureActsValues[i];
                        
                        // Ensure index is within valid range
                        if (index >= 0 && index < 64) {
                          activationsArray[index] = value;
                          // Use maximum activation value (consistent with feature page logic)
                          if (Math.abs(value) > Math.abs(maxActivation)) {
                            maxActivation = value;
                          }
                        }
                      }
                      
                      console.log('🔍 Processing activation data:', {
                        indicesLength: sample.featureActsIndices.length,
                        valuesLength: sample.featureActsValues.length,
                        nonZeroCount: activationsArray.filter(v => v !== 0).length,
                        maxActivation
                      });
                    }
                    
                    chessSamples.push({
                      fen: trimmed,
                      activationStrength: maxActivation,
                      activations: activationsArray,
                      zPatternIndices: sample.zPatternIndices,
                      zPatternValues: sample.zPatternValues,
                      contextId: sample.contextIdx || sample.context_idx,
                      sampleIndex: sample.sampleIndex || 0
                    });
                    
                    break; // Found a valid FEN, move to next sample
                  }
                }
              }
            }
          }
        }
      }
      
      // Sort by maximum activation value and take top 8 (consistent with feature page logic)
      const topSamples = chessSamples
        .sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength))
        .slice(0, 8);
      
      console.log('✅ Got Top Activation data:', {
        totalChessSamples: chessSamples.length,
        topSamplesCount: topSamples.length
      });
      
      setTopActivations(topSamples);
      
    } catch (error) {
      console.error('❌ Failed to get Top Activation data:', error);
      setTopActivations([]);
    } finally {
      setLoadingTopActivations(false);
    }
  }, [circuitVisualizationData]);

  // When clicking a node, get Top Activation data
  useEffect(() => {
    if (clickedNodeId) {
      fetchTopActivations(clickedNodeId);
    } else {
      setTopActivations([]);
    }
  }, [clickedNodeId, fetchTopActivations]);

  // Extract FEN string from circuit trace result
  const extractFenFromCircuitTrace = useCallback(() => {
    if (!circuitTraceResult?.metadata?.prompt_tokens) return null;
    
    const promptText = circuitTraceResult.metadata.prompt_tokens.join(' ');
    console.log('🔍 Searching FEN string:', promptText);
    
    // More lenient FEN format detection
    const lines = promptText.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      // Check if it contains FEN format - contains slash and has enough characters
      if (trimmed.includes('/')) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 6) {
          const [boardPart, activeColor] = parts;
          const boardRows = boardPart.split('/');
          
          if (boardRows.length === 8 && /^[wb]$/.test(activeColor)) {
            console.log('✅ Found FEN string:', trimmed);
            return trimmed;
          }
        }
      }
    }
    
    const simpleMatch = promptText.match(/[rnbqkpRNBQKP1-8\/]{15,}\s+[wb]\s+[KQkqA-Za-z-]+\s+[a-h][36-]?\s*\d*\s*\d*/);
    if (simpleMatch) {
      console.log('✅ Found simple FEN match:', simpleMatch[0]);
      return simpleMatch[0];
    }
    
    console.log('No FEN string found');
    return null;
  }, [circuitTraceResult]);


  // Get node activation data
  const getNodeActivationData = useCallback((nodeId: string | null): NodeActivationData => {
    if (!nodeId || !circuitTraceResult) {
      console.log('Missing required parameters:', { nodeId, hasCircuitTraceResult: !!circuitTraceResult });
      return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
    }
    
    console.log(`🔍 Looking up activation data for node ${nodeId}...`);
    console.log('📋 Circuit trace result structure:', {
      hasActivationInfo: !!circuitTraceResult.activation_info,
      activationInfoKeys: circuitTraceResult.activation_info ? Object.keys(circuitTraceResult.activation_info) : [],
      hasNodes: !!circuitTraceResult.nodes,
      nodesLength: circuitTraceResult.nodes?.length || 0
    });
    
    // Parse node_id -> rawLayer, featureOrHead, ctx(position)
    const parseFromNodeId = (id: string) => {
      const parts = id.split('_');
      const rawLayer = Number(parts[0]) || 0;
      const featureOrHead = Number(parts[1]) || 0;
      const ctxIdx = Number(parts[2]) || 0;
      // Divide raw layer index by 2 to get actual layer index
      const layerForActivation = Math.floor(rawLayer / 2);
      return { rawLayer, layerForActivation, featureOrHead, ctxIdx };
    };
    const parsed = parseFromNodeId(nodeId);

    // First determine node type
    let featureTypeForNode: string | undefined = undefined;
    if (circuitTraceResult.nodes && Array.isArray(circuitTraceResult.nodes)) {
      const nodeMeta = circuitTraceResult.nodes.find((n: any) => n?.node_id === nodeId);
      featureTypeForNode = nodeMeta?.feature_type;
    }

    console.log('🔍 Node parsing information:', {
      nodeId,
      parsed,
      featureTypeForNode
    });

    // 1) Priority: look up activation data from activation_info
    if (circuitTraceResult.activation_info) {
      console.log('🔍 Looking up activation data from activation_info...');
      console.log('📋 activation_info structure:', {
        hasActivationInfo: !!circuitTraceResult.activation_info,
        activationInfoKeys: Object.keys(circuitTraceResult.activation_info),
        traceSide,
        hasDirectFeatures: !!circuitTraceResult.activation_info.features,
        sideActivationInfo: circuitTraceResult.activation_info[traceSide]
      });
      
      // Check if it is merged activation information (directly contains features)
      let featuresToSearch = null;
      if (circuitTraceResult.activation_info.features && Array.isArray(circuitTraceResult.activation_info.features)) {
        // This is merged activation information, directly use
        featuresToSearch = circuitTraceResult.activation_info.features;
        console.log(`🔍 Using merged activation information, found ${featuresToSearch.length} features`);
      } else {
        // This is the original q/k branch structure, select based on traceSide   
        const sideActivationInfo = circuitTraceResult.activation_info[traceSide];
        if (sideActivationInfo && sideActivationInfo.features && Array.isArray(sideActivationInfo.features)) {
          featuresToSearch = sideActivationInfo.features;
          console.log(`Found activation information for ${traceSide} side with ${featuresToSearch.length} features`);
        }
      }
      
      if (featuresToSearch) {
        // Look for matching features in the features array
        for (const featureInfo of featuresToSearch) {
          const matchesLayer = featureInfo.layer === parsed.layerForActivation;
          const matchesPosition = featureInfo.position === parsed.ctxIdx;
          
          let matchesIndex = false;
          if (featureTypeForNode) {
            const t = featureTypeForNode.toLowerCase();
            if (t === 'lorsa') {
              matchesIndex = featureInfo.head_idx === parsed.featureOrHead;
            } else if (t === 'cross layer transcoder' || t.includes('transcoder')) {
              matchesIndex = featureInfo.feature_idx === parsed.featureOrHead;
            }
          } else {
            // Fallback: try matching any index
            matchesIndex = (featureInfo.head_idx === parsed.featureOrHead) || 
                          (featureInfo.feature_idx === parsed.featureOrHead);
          }
          
          if (matchesLayer && matchesPosition && matchesIndex) {
            console.log('✅ Found matching feature in activation_info:', {
              featureId: featureInfo.featureId,
              type: featureInfo.type,
              layer: featureInfo.layer,
              position: featureInfo.position,
              head_idx: featureInfo.head_idx,
              feature_idx: featureInfo.feature_idx,
              hasActivations: !!featureInfo.activations,
              hasZPattern: !!(featureInfo.zPatternIndices && featureInfo.zPatternValues)
            });
            
            return {
              activations: featureInfo.activations,
              zPatternIndices: featureInfo.zPatternIndices,
              zPatternValues: featureInfo.zPatternValues,
              nodeType: featureInfo.type,
              clerp: undefined // activation_info does not contain clerp information
            };
          }
        }
        
        console.log('No matching feature found in activation_info');
      } else {
        console.log(`No activation_info or features array for ${traceSide} side`);
      }
    }

    // 2) Fallback: check original node inline fields
    let nodesToSearch: any[] = [];
    if (circuitTraceResult.nodes && Array.isArray(circuitTraceResult.nodes)) {
      nodesToSearch = circuitTraceResult.nodes;
    } else if (Array.isArray(circuitTraceResult)) {
      nodesToSearch = circuitTraceResult;
    }

    if (nodesToSearch.length > 0) {
      const exactMatch = nodesToSearch.find(node => node?.node_id === nodeId);
      if (exactMatch) {
        const inlineActs = exactMatch.activations;
        const inlineZIdx = exactMatch.zPatternIndices;
        const inlineZVal = exactMatch.zPatternValues;
        console.log('✅ Node inline fields check:', {
          hasInlineActivations: !!inlineActs,
          hasInlineZIdx: !!inlineZIdx,
          hasInlineZVal: !!inlineZVal,
        });
        if (inlineActs || (inlineZIdx && inlineZVal)) {
          return {
            activations: inlineActs,
            zPatternIndices: inlineZIdx,
            zPatternValues: inlineZVal,
            nodeType: exactMatch.feature_type,
            clerp: exactMatch.clerp,
          };
        } 
      }
    }

    // 3) Deep scan activation record collection
    const candidateRecords: any[] = [];
    const pushCandidateArrays = (obj: any) => {
      if (!obj) return;
      if (Array.isArray(obj)) {
        for (const item of obj) {
          if (item && typeof item === 'object') {
            const hasActivationShape = ('layer' in item) && ('position' in item) && ('activations' in item);
            const hasZShape = ('zPatternIndices' in item) && ('zPatternValues' in item);
            const hasIndexKey = ('head_idx' in item) || ('feature_idx' in item);
            if (hasActivationShape || hasZShape || hasIndexKey) {
              candidateRecords.push(item);
            }
          }
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) pushCandidateArrays(v);
      }
    };
    pushCandidateArrays(circuitTraceResult);

    console.log('🧭 Candidate record count:', candidateRecords.length);

    // Define matching function
    const tryMatchRecord = (rec: any, featureType?: string) => {
      const recLayer = Number(rec?.layer);
      const recPos = Number(rec?.position);
      const recHead = rec?.head_idx;
      const recFeatIdx = rec?.feature_idx;

      const layerOk = !Number.isNaN(recLayer) && recLayer === parsed.layerForActivation;
      const posOk = !Number.isNaN(recPos) && recPos === parsed.ctxIdx;

      let indexOk = false;
      if (featureType) {
        const t = featureType.toLowerCase();
        if (t === 'lorsa') indexOk = recHead === parsed.featureOrHead;
        else if (t === 'cross layer transcoder' || t.includes('transcoder')) indexOk = recFeatIdx === parsed.featureOrHead;
        else indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      } else {
        indexOk = (recHead === parsed.featureOrHead) || (recFeatIdx === parsed.featureOrHead);
      }

      return layerOk && posOk && indexOk;
    };

    const matched = candidateRecords.find(rec => tryMatchRecord(rec, featureTypeForNode));
    if (matched) {
      console.log('Matched activation record via parsing:', {
        nodeId,
        layerForActivation: parsed.layerForActivation,
        ctxIdx: parsed.ctxIdx,
        featureOrHead: parsed.featureOrHead,
        featureTypeForNode,
      });
      return {
        activations: matched.activations,
        zPatternIndices: matched.zPatternIndices,
        zPatternValues: matched.zPatternValues,
        nodeType: featureTypeForNode,
        clerp: (nodesToSearch.find(n => n?.node_id === nodeId) || {}).clerp,
      };
    }

    // 4) Final fuzzy matching
    if (nodesToSearch.length > 0) {
      const fuzzyMatches = nodesToSearch.filter(node => node?.node_id && node.node_id.includes(nodeId.split('_')[0]));
      if (fuzzyMatches.length > 0) {
        const firstMatch = fuzzyMatches[0];
        console.log('Using fuzzy matched node:', {
          node_id: firstMatch.node_id,
          hasActivations: !!firstMatch.activations,
        });
        return {
          activations: firstMatch.activations,
          zPatternIndices: firstMatch.zPatternIndices,
          zPatternValues: firstMatch.zPatternValues,
          nodeType: firstMatch.feature_type,
          clerp: firstMatch.clerp,
        };
      }
    }

    console.log('No matching node/record found');
    return { activations: undefined, zPatternIndices: undefined, zPatternValues: undefined };
  }, [circuitTraceResult, traceSide]);

  // When lastMove changes, update positiveMove (if empty)
  useEffect(() => {
    if (lastMove && !positiveMove) {
      setPositiveMove(lastMove);
    }
  }, [lastMove, positiveMove]);

  // When effective analysis FEN changes: only clear pending analysis move when user has not input, avoid overwriting user input
  const prevEffectiveGameFenRef = useRef<string>(effectiveGameFen);
  useEffect(() => {
    if (prevEffectiveGameFenRef.current !== effectiveGameFen) {
      if (!positiveMove.trim() && !negativeMove.trim()) {
        setPositiveMove('');
        setNegativeMove('');
        setMoveError('');
      }
      prevEffectiveGameFenRef.current = effectiveGameFen;
    }
  }, [effectiveGameFen, positiveMove, negativeMove]);

  return (
    <div className="space-y-6">
      {/* Circuit Trace control panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Circuit Trace Analysis</span>
            <div className="flex gap-2">
              <Button
                onClick={() => setShowParamsDialog(true)}
                variant="outline"
                size="sm"
              >
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
              <Button
                onClick={() => handleCircuitTrace('positive')}
                disabled={isTracing}
                variant={isTracing ? 'destructive' : 'default'}
                size="sm"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Tracing...
                  </>
                ) : (
                  'Positive Trace'
                )}
              </Button>
              <Button
                onClick={() => handleCircuitTrace('negative')}
                disabled={isTracing}
                variant={isTracing ? 'destructive' : 'outline'}
                size="sm"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Tracing...
                  </>
                ) : (
                  'Negative Trace'
                )}
              </Button>
              <Button
                onClick={() => handleCircuitTrace('both')}
                disabled={isTracing}
                variant={isTracing ? 'destructive' : 'outline'}
                size="sm"
                className="bg-purple-500 hover:bg-purple-600 text-white"
              >
                {isTracing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Tracing...
                  </>
                ) : (
                  'Both Trace'
                )}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Circuit Trace log display */}
          {(isTracing || traceLogs.length > 0) && (
            <div className="mb-4 rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
              <div className="mb-2 font-semibold flex items-center justify-between">
                <span>
                  {isTracing ? '🔍 Circuit Tracing Logs' : '📋 Circuit Tracing Logs (completed)'}
                </span>
              </div>
              <div className="max-h-40 overflow-y-auto rounded bg-blue-100 p-2 text-xs font-mono leading-relaxed">
                {traceLogs.length === 0 ? (
                  <div className="text-blue-700 opacity-80">
                    {isTracing ? 'Waiting for logs...' : 'No logs yet'}
                  </div>
                ) : (
                  traceLogs.map((log, idx) => (
                    <div key={`${log.timestamp}-${idx}`} className="mb-1">
                      {new Date(log.timestamp * 1000).toLocaleTimeString()} - {log.message}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
          <div className="space-y-4">
            {/* Side selector */}
            <div className="space-y-2">
              <Label htmlFor="side-select" className="text-sm font-medium text-gray-700">
                Attention side
              </Label>
              <Select value={traceSide} onValueChange={(v: 'q' | 'k' | 'both') => setTraceSide(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="q">Q side (Query)</SelectItem>
                  <SelectItem value="k">K side (Key)</SelectItem>
                  <SelectItem value="both">Q+K side (merged)</SelectItem>
                </SelectContent>
              </Select>
              <div className="text-xs text-gray-500">
                Choose which attention side to analyze
              </div>
            </div>
            
            {/* Positive Move input */}
            <div className="space-y-2">
              <Label htmlFor="positive-move-input" className="text-sm font-medium text-gray-700">
                Positive Move (UCI, e.g. e2e4)
              </Label>
              <div className="flex gap-2">
                <Input
                  id="positive-move-input"
                  type="text"
                  placeholder="Enter the UCI move you want to promote"
                  value={positiveMove}
                  onChange={(e) => {
                    setPositiveMove(e.target.value);
                    setMoveError('');
                    saveCachedPositiveMove(safeDecodeFen(effectiveGameFen), e.target.value);
                  }}
                  className={`font-mono ${moveError && moveError.includes('Positive') ? 'border-red-500' : ''}`}
                />
                <Button
                  onClick={() => {
                    const move = lastMove || '';
                    setPositiveMove(move);
                    if (move) {
                      saveCachedPositiveMove(safeDecodeFen(effectiveGameFen), move);
                    }
                  }}
                  variant="outline"
                  size="sm"
                  disabled={!lastMove}
                >
                  Use last move
                </Button>
              </div>
              <div className="text-xs text-gray-500">
                Used for Positive Trace and Both Trace (promote this move)
              </div>
            </div>
            
            {/* Negative Move input */}
            <div className="space-y-2">
              <Label htmlFor="negative-move-input" className="text-sm font-medium text-gray-700">
                Negative Move (UCI, e.g. e2e4)
              </Label>
              <div className="flex gap-2">
                <Input
                  id="negative-move-input"
                  type="text"
                  placeholder="Enter the UCI move you want to suppress"
                  value={negativeMove}
                  onChange={(e) => {
                    setNegativeMove(e.target.value);
                    setMoveError('');
                    saveCachedNegativeMove(safeDecodeFen(effectiveGameFen), e.target.value);
                  }}
                  className={`font-mono ${moveError && moveError.includes('Negative') ? 'border-red-500' : ''}`}
                />
              </div>
              <div className="text-xs text-gray-500">
                Used for Negative Trace and Both Trace (suppress this move)
              </div>
              {moveError && (
                <p className="text-sm text-red-600">{moveError}</p>
              )}
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Analysis FEN (before move):</span>
                <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-1 break-all border border-blue-200">
                  {gameFen}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Current FEN (after move):</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 break-all border border-green-200">
                  {currentFen || effectiveGameFen}
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Positive Move:</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 border border-green-200">
                  {positiveMove || lastMove || 'No move yet'}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Negative Move:</span>
                <div className="font-mono text-xs bg-red-50 p-2 rounded mt-1 border border-red-200">
                  {negativeMove || 'No move yet'}
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Move to analyze:</span>
                <div className="font-mono text-xs bg-yellow-50 p-2 rounded mt-1 border border-yellow-200">
                  {positiveMove || negativeMove || lastMove || 'No move yet'}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Move history:</span>
                <div className="font-mono text-xs bg-gray-100 p-2 rounded mt-1">
                  {gameHistory.length > 0 ? gameHistory.join(' ') : 'No move yet'}
                </div>
              </div>
            </div>
            
            {/* Current parameter display */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Max feature nodes:</span>
                <div className="font-mono text-xs bg-blue-50 p-2 rounded mt-1 border border-blue-200">
                  {circuitParams.max_feature_nodes}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Node threshold:</span>
                <div className="font-mono text-xs bg-green-50 p-2 rounded mt-1 border border-green-200">
                  {circuitParams.node_threshold}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Edge threshold:</span>
                <div className="font-mono text-xs bg-purple-50 p-2 rounded mt-1 border border-purple-200">
                  {circuitParams.edge_threshold}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-700">Max activation count:</span>
                <div className="font-mono text-xs bg-orange-50 p-2 rounded mt-1 border border-orange-200">
                  {circuitParams.max_act_times === null ? 'Unlimited' : 
                   circuitParams.max_act_times >= 1000000 ? 
                   `${(circuitParams.max_act_times / 1000000).toFixed(0)}M` : 
                   circuitParams.max_act_times.toLocaleString()}
                </div>
              </div>
            </div>
            
            {!positiveMove && !negativeMove && !lastMove && (
              <div className="text-center py-4 text-gray-500 bg-yellow-50 rounded-lg border border-yellow-200">
                <p>Please enter a Positive or Negative Move (UCI format), or play a move first.</p>
                <p className="text-sm mt-1">For example: e2e4, Nf3, O-O (castling king side uses e1g1), O-O-O (castling queen side uses e1c1).</p>
                <p className="text-sm mt-1 text-purple-600">Both Trace requires both a Positive Move and a Negative Move.</p>
              </div>
            )}

          </div>
        </CardContent>
      </Card>

      {/* Circuit visualization area */}
      {circuitVisualizationData && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Circuit Trace Visualization</span>
              <div className="flex gap-2">
                <Button
                  onClick={handleSaveGraphJson}
                  variant="outline"
                  size="sm"
                >
                  Save JSON
                </Button>
                <Button
                  onClick={() => {
                    setCircuitVisualizationData(null);
                    setCircuitTraceResult(null);
                    setClickedNodeId(null);
                    setHoveredNodeId(null);
                    setPinnedNodeIds([]);
                    setHiddenNodeIds([]);
                    setSelectedFeature(null);
                    setConnectedFeatures([]);
                  }}
                  variant="outline"
                  size="sm"
                >
                  Clear visualization
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Circuit Visualization Layout */}
            <div className="space-y-6 w-full max-w-full overflow-hidden">
              {/* Top Row: Link Graph and Node Connections side by side */}
              <div className="flex gap-6 h-[900px] w-full max-w-full overflow-hidden">
                {/* Link Graph Component - Left Side */}
                <div className="flex-1 min-w-0 max-w-full border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                  <div className="w-full h-full overflow-hidden relative">
                    <LinkGraphContainer 
                      data={circuitVisualizationData} 
                      onNodeClick={handleNodeClick}
                      onNodeHover={handleNodeHover}
                      onFeatureSelect={handleFeatureSelect}
                      onConnectedFeaturesSelect={handleConnectedFeaturesSelect}
                      onConnectedFeaturesLoading={handleConnectedFeaturesLoading}
                      clickedId={clickedNodeId}
                      hoveredId={hoveredNodeId}
                      pinnedIds={pinnedNodeIds}
                    />
                  </div>
                </div>

                {/* Node Connections Component - Right Side */}
                <div className="w-96 flex-shrink-0 border rounded-lg p-4 bg-white shadow-sm overflow-hidden">
                  <NodeConnections
                    data={circuitVisualizationData}
                    clickedId={clickedNodeId}
                    hoveredId={hoveredNodeId}
                    pinnedIds={pinnedNodeIds}
                    hiddenIds={hiddenNodeIds}
                    onFeatureClick={handleNodeClick}
                    onFeatureSelect={handleFeatureSelect}
                    onFeatureHover={handleNodeHover}
                  />
                </div>
              </div>

              {/* Chess Board Display */}
              {(() => {
                const fen = extractFenFromCircuitTrace();
                const nodeActivationData = getNodeActivationData(clickedNodeId);
                
                if (!fen) return null;
                
                return (
                  <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
                    <h3 className="text-lg font-semibold mb-4 text-center">
                      Circuit Trace Board State
                      {clickedNodeId && nodeActivationData && (
                        <span className="text-sm font-normal text-blue-600 ml-2">
                          (Node: {clickedNodeId}{nodeActivationData.nodeType ? ` - ${nodeActivationData.nodeType.toUpperCase()}` : ''})
                        </span>
                      )}
                    </h3>
                    {clickedNodeId && nodeActivationData && nodeActivationData.activations && (
                      <div className="text-center mb-2 text-sm text-purple-600">
                        Activations: {nodeActivationData.activations.filter((v: number) => v !== 0).length} non-zero cells
                        {nodeActivationData.zPatternIndices && nodeActivationData.zPatternValues && 
                          `, ${nodeActivationData.zPatternValues.length} Z-pattern connections`
                        }
                      </div>
                    )}
                    <div className="flex justify-center">
                      <ChessBoard
                        fen={fen}
                        size="medium"
                        showCoordinates={true}
                        activations={nodeActivationData?.activations}
                        zPatternIndices={nodeActivationData?.zPatternIndices}
                        zPatternValues={nodeActivationData?.zPatternValues}
                        flip_activation={Boolean(fen && fen.split(' ')[1] === 'b')}
                        sampleIndex={clickedNodeId ? parseInt(clickedNodeId.split('_')[1]) : undefined}
                        analysisName={`${nodeActivationData?.nodeType || 'Circuit Node'} (${traceSide.toUpperCase()} side)`}
                      />
                    </div>
                  </div>
                );
              })()}

              {/* Bottom Row: Feature Card - only shown when there is no Top Activation data */}
              {clickedNodeId && topActivations.length === 0 && (() => {
                const currentNode = circuitVisualizationData.nodes.find((node: any) => node.nodeId === clickedNodeId);
                
                if (!currentNode) {
                  console.log('❌ Node not found:', clickedNodeId);
                  return null;
                }
                
                console.log('✅ Found node:', currentNode);
                
                // Parse the real feature ID from node_id (format: layer_featureId_ctxIdx)
                const parseNodeId = (nodeId: string) => {
                  const parts = nodeId.split('_');
                  if (parts.length >= 2) {
                    const rawLayer = parseInt(parts[0]) || 0;
                    return {
                      layerIdx: Math.floor(rawLayer / 2), // divide by 2 to get actual model layer index
                      featureIndex: parseInt(parts[1]) || 0
                    };
                  }
                  return { layerIdx: 0, featureIndex: 0 };
                };
                
                const { layerIdx, featureIndex } = parseNodeId(currentNode.nodeId);
                const isLorsa = currentNode.feature_type?.toLowerCase() === 'lorsa';
                
                // Build the correct dictionary name based on node type
                let dictionary: string;
                if (isLorsa) {
                  const lorsaAnalysisName = circuitVisualizationData?.metadata?.lorsa_analysis_name;
                  if (lorsaAnalysisName && lorsaAnalysisName.includes('BT4')) {
                    // BT4 format: BT4_lorsa_L{layer}A
                    dictionary = `BT4_lorsa_L${layerIdx}A`;
                  } else {
                    dictionary = lorsaAnalysisName ? lorsaAnalysisName.replace("{}", layerIdx.toString()) : `lc0-lorsa-L${layerIdx}`;
                  }
                } else {
                  const tcAnalysisName = (circuitVisualizationData?.metadata as any)?.tc_analysis_name || circuitVisualizationData?.metadata?.clt_analysis_name;
                  if (tcAnalysisName && tcAnalysisName.includes('BT4')) {
                    // BT4 format: BT4_tc_L{layer}M
                    dictionary = `BT4_tc_L${layerIdx}M`;
                  } else {
                    dictionary = tcAnalysisName ? tcAnalysisName.replace("{}", layerIdx.toString()) : `lc0_L${layerIdx}M_16x_k30_lr2e-03_auxk_sparseadam`;
                  }
                }
                
                const nodeTypeDisplay = isLorsa ? 'LORSA' : 'SAE';
                
                // Function to navigate to the global-weight page
                const handleViewGlobalWeight = () => {
                  const featureType = isLorsa ? 'lorsa' : 'tc';
                  const saeComboId = circuitVisualizationData?.metadata?.sae_combo_id;
                  const params = new URLSearchParams({
                    feature_type: featureType,
                    layer_idx: layerIdx.toString(),
                    feature_idx: featureIndex.toString(),
                  });
                  if (saeComboId) {
                    params.append('sae_combo_id', saeComboId);
                  }
                  navigate(`/global-weight?${params.toString()}`);
                };
                
                return (
                  <div className="w-full border rounded-lg p-4 bg-white shadow-sm">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold">Selected Feature Details</h3>
                      <div className="flex items-center space-x-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleViewGlobalWeight}
                          className="flex items-center gap-2"
                        >
                          <ExternalLink className="w-4 h-4" />
                          View global weight
                        </Button>
                        {connectedFeatures.length > 0 && (
                          <div className="flex items-center space-x-2">
                            <span className="text-sm text-gray-600">Connected features:</span>
                            <span className="px-2 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
                              {connectedFeatures.length}
                            </span>
                          </div>
                        )}
                        {/* Link to the Feature page */}
                        {currentNode && featureIndex !== undefined && (
                          <Link
                            to={`/features?dictionary=${encodeURIComponent(dictionary)}&featureIndex=${featureIndex}`}
                            className="inline-flex items-center px-3 py-2 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 transition-colors"
                            title={`Open L${layerIdx} ${nodeTypeDisplay} Feature #${featureIndex}`}
                          >
                            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                            View L{layerIdx} {nodeTypeDisplay} #{featureIndex}
                          </Link>
                        )}
                      </div>
                    </div>
                    
                    {/* Basic node information */}
                    <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium text-gray-700">Node ID:</span>
                          <span className="ml-2 font-mono text-blue-600">{currentNode.nodeId}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Feature type:</span>
                          <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                            {currentNode.feature_type || 'Unknown'}
                          </span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Layer index:</span>
                          <span className="ml-2">{layerIdx}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Feature index:</span>
                          <span className="ml-2">{featureIndex}</span>
                        </div>
                        {currentNode.sourceLinks && (
                          <div>
                            <span className="font-medium text-gray-700">Outgoing edges:</span>
                            <span className="ml-2">{currentNode.sourceLinks.length}</span>
                          </div>
                        )}
                        {currentNode.targetLinks && (
                          <div>
                            <span className="font-medium text-gray-700">Incoming edges:</span>
                            <span className="ml-2">{currentNode.targetLinks.length}</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {selectedFeature ? (
                      <FeatureCard feature={selectedFeature} />
                    ) : (
                      <div className="flex items-center justify-center p-8 bg-gray-50 border rounded-lg">
                        <div className="text-center">
                          <p className="text-gray-600 mb-2">No feature is available for this node</p>
                          <p className="text-sm text-gray-500">
                            Click the "View L{layerIdx} {nodeTypeDisplay} #{featureIndex}" link above to see more details.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Keep original simple circuitTraceResult display but remove navigation button */}
      {circuitTraceResult && !circuitVisualizationData && (
        <Card>
          <CardHeader>
            <CardTitle>Circuit Trace Result</CardTitle>
          </CardHeader>
          <CardContent>
            {circuitTraceResult.nodes ? (
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Analysis summary</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-blue-700">Node count:</span>
                      <span className="ml-2 font-mono">{circuitTraceResult.nodes.length}</span>
                    </div>
                    <div>
                      <span className="text-blue-700">Link count:</span>
                      <span className="ml-2 font-mono">{circuitTraceResult.links?.length || 0}</span>
                    </div>
                    {circuitTraceResult.metadata?.target_move && (
                      <div>
                        <span className="text-blue-700">Target move:</span>
                        <span className="ml-2 font-mono text-green-600">{circuitTraceResult.metadata.target_move}</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium">Key nodes (top 10)</h4>
                  {circuitTraceResult.nodes.slice(0, 10).map((node: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-sm">{node.node_id}</span>
                        <Badge variant="outline" className="text-xs">
                          {node.feature_type}
                        </Badge>
                      </div>
                      {node.node_id && (
                        <Link
                          to={`/features?nodeId=${encodeURIComponent(node.node_id)}`}
                          className="text-blue-600 underline text-sm hover:text-blue-800"
                        >
                          View feature
                        </Link>
                      )}
                    </div>
                  ))}
                  {circuitTraceResult.nodes.length > 10 && (
                    <div className="text-center text-sm text-gray-500">
                      {circuitTraceResult.nodes.length - 10} more nodes
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                No node data
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Top Activation section */}
      {clickedNodeId && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Top Activation Boards</span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Node: {clickedNodeId}</span>
                {loadingTopActivations && (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-500">Loading...</span>
                  </div>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loadingTopActivations ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-gray-600">Fetching Top Activation data...</p>
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
                      Max activation: {sample.activationStrength.toFixed(3)}
                      </div>
                    </div>
                    <ChessBoard
                      fen={sample.fen}
                      size="small"
                      showCoordinates={false}
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
                <p>No activation samples containing a chess board were found.</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Parameter settings dialog */}
      <Dialog open={showParamsDialog} onOpenChange={setShowParamsDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Circuit Trace Parameter Settings
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="max_feature_nodes">Max feature nodes</Label>
                <Input
                  id="max_feature_nodes"
                  type="number"
                  min="1"
                  max="10000"
                  step="1"
                  value={circuitParams.max_feature_nodes}
                  onChange={(e) => handleParamsChange('max_feature_nodes', e.target.value)}
                  className="font-mono"
                />
                <p className="text-xs text-gray-500">
                  Controls the maximum number of feature nodes considered in circuit trace. Default: {CIRCUIT_TRACE_DEFAULTS.max_feature_nodes}.
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="node_threshold">Node threshold</Label>
                <Input
                  id="node_threshold"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={circuitParams.node_threshold}
                  onChange={(e) => handleParamsChange('node_threshold', e.target.value)}
                  className="font-mono"
                />
                <p className="text-xs text-gray-500">
                  Node importance threshold used to filter out unimportant nodes. Default: {CIRCUIT_TRACE_DEFAULTS.node_threshold}.
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="edge_threshold">Edge threshold</Label>
                <Input
                  id="edge_threshold"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={circuitParams.edge_threshold}
                  onChange={(e) => handleParamsChange('edge_threshold', e.target.value)}
                  className="font-mono"
                />
                <p className="text-xs text-gray-500">
                  Edge importance threshold used to filter out unimportant connections. Default: {CIRCUIT_TRACE_DEFAULTS.edge_threshold}.
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="max_act_times">Max activation count</Label>
                <Input
                  id="max_act_times"
                  type="number"
                  min="10000000"
                  max="100000000"
                  step="10000000"
                  value={circuitParams.max_act_times || ''}
                  onChange={(e) => handleParamsChange('max_act_times', e.target.value)}
                  className="font-mono"
                  placeholder="Leave empty for no limit"
                />
                <p className="text-xs text-gray-500">
                  Filters dense features. Range: 10M–100M; leave empty for no limit.
                </p>
              </div>
            </div>
            
            {/* Current parameter preview */}
            <div className="bg-gray-50 p-4 rounded-lg space-y-2">
              <h4 className="font-medium text-sm text-gray-700">Current parameter preview:</h4>
              <div className="grid grid-cols-1 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Max feature nodes:</span>
                  <span className="font-mono text-blue-600">{circuitParams.max_feature_nodes}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Node threshold:</span>
                  <span className="font-mono text-green-600">{circuitParams.node_threshold}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Edge threshold:</span>
                  <span className="font-mono text-purple-600">{circuitParams.edge_threshold}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Max activation count:</span>
                  <span className="font-mono text-orange-600">
                    {circuitParams.max_act_times === null ? 'Unlimited' : 
                     circuitParams.max_act_times >= 1000000 ? 
                     `${(circuitParams.max_act_times / 1000000).toFixed(0)}M` : 
                     circuitParams.max_act_times.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          <DialogFooter className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => setShowParamsDialog(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={() => {
                // Reset to default values
                setCircuitParams({
                  max_feature_nodes: CIRCUIT_TRACE_DEFAULTS.max_feature_nodes,
                  node_threshold: CIRCUIT_TRACE_DEFAULTS.node_threshold,
                  edge_threshold: CIRCUIT_TRACE_DEFAULTS.edge_threshold,
                  max_act_times: null,
                });
              }}
              variant="outline"
            >
              Reset defaults
            </Button>
            <Button
              onClick={handleSaveParams}
            >
              Save settings
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
