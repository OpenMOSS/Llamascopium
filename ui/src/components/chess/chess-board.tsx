import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ChessBoardProps {
  fen: string;
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  sampleIndex?: number;
  analysisName?: string;
  contextId?: number;
  size?: 'small' | 'medium' | 'large';
  showCoordinates?: boolean;
  move?: string; // UCI move string, e.g. "a2a4"
  orientation?: 'white' | 'black' | 'auto';
  flip_activation?: boolean; // whether to flip activation index by row
  onMove?: (move: string) => void;
  onSquareClick?: (square: string) => void;
  isInteractive?: boolean;
  autoFlipWhenBlack?: boolean; // auto flip board when black to move
  moveColor?: string; // arrow color (match node color)
  showSelfPlay?: boolean;
  disableAutoAnalyze?: boolean; // disable auto board analysis to avoid repeated model load
  /** When set with disableAutoAnalyze, schedule loadWdl after this index * delay (e.g. 1.5s) so WDL loads one-by-one after all boards render */
  wdlLoadDelayIndex?: number;
}

// Piece SVG filename map (aligned with exp/chess-board-visualizer, load from pieces dir)
const PIECE_SVG_NAMES: { [key: string]: string } = {
  K: 'wK',
  Q: 'wQ',
  R: 'wR',
  B: 'wB',
  N: 'wN',
  P: 'wP',
  k: 'bK',
  q: 'bQ',
  r: 'bR',
  b: 'bB',
  n: 'bN',
  p: 'bP',
};

/** Return SVG URL for a FEN piece character (for img src) */
const getPieceSrc = (piece: string): string => {
  const name = PIECE_SVG_NAMES[piece] || piece;
  return new URL(`./pieces/${name}.svg`, import.meta.url).href;
};

// FEN string parser
const parseFEN = (fen: string) => {
  const parts = fen.trim().split(' ');
  if (parts.length < 4) {
    throw new Error('Invalid FEN string');
  }

  const [boardPart, activeColor, castling, enPassant] = parts;
  const rows = boardPart.split('/');

  if (rows.length !== 8) {
    throw new Error('Invalid board configuration in FEN');
  }

  // Convert FEN to 8x8 board array
  const board: (string | null)[][] = [];
  
  for (let i = 0; i < 8; i++) {
    const row: (string | null)[] = [];
    const rowStr = rows[i];
    
    for (const char of rowStr) {
      if (/\d/.test(char)) {
        // Digit = number of empty squares
        const emptySquares = parseInt(char);
        for (let j = 0; j < emptySquares; j++) {
          row.push(null);
        }
      } else {
        // Piece character
        row.push(char);
      }
    }
    
    board.push(row);
  }

  return {
    board,
    activeColor,
    castling,
    enPassant,
    isWhiteToMove: activeColor === 'w'
  };
};

// Row/col to linear index (0-63)
const getSquareIndex = (row: number, col: number): number => {
  return row * 8 + col;
};

// Color for activation strength, scaled relative to the board's activation range.
const getActivationColor = (activation: number, maxActivation: number): string => {
  if (activation === 0) return 'transparent';

  const safeMaxActivation = Math.max(maxActivation, Number.EPSILON);
  const normalized = Math.min(Math.abs(activation) / safeMaxActivation, 1);

  // Use a gentle curve so mid-range activations separate more clearly.
  const easedIntensity = Math.pow(normalized, 0.65);
  const opacity = 0.08 + easedIntensity * 0.84;

  return `rgba(239, 68, 68, ${opacity})`;
};

/**
 * Z-pattern: which squares "contribute" to the hovered square (source).
 *
 * Logic in 3 steps:
 *
 * 1) Parse input into list of (source, target, strength).
 *    Supports two shapes (same as feature page):
 *    - Pair list: indices[i] = [source_i, target_i], values[i] = strength_i
 *    - Parallel:  indices[0] = [s1,s2,...], indices[1] = [t1,t2,...], values[i] = strength_i
 *
 * 2) Keep only entries where source === sourceSquare (hovered square).
 *    Result: list of (target, strength) for that source.
 *
 * 3) Sort by |strength| descending, take at most 8.
 *    Formula: result = sort(targets, key = |strength|, descending)[0:8]
 *
 * Why fewer than 8? The backend (lorsa) stores z_pattern sparsely: only (q,k) with
 * |z| >= 0.01 * activation(q) are saved. So for a given source we may have < 8
 * (or < 64) entries in the data; there is no extra filter here.
 */
const getZPatternTargets = (sourceSquare: number, zPatternIndices?: number[][], zPatternValues?: number[]) => {
  if (!zPatternIndices || !zPatternValues) return [];

  const targets: { square: number; strength: number }[] = [];

  const looksLikePairList = Array.isArray(zPatternIndices[0]) && (zPatternIndices[0] as number[]).length === 2;

  if (looksLikePairList) {
    for (let i = 0; i < zPatternIndices.length; i++) {
      const pair = zPatternIndices[i] as number[];
      const value = zPatternValues[i] || 0;
      const [source, target] = pair;
      if (source === sourceSquare) targets.push({ square: target, strength: value });
    }
  } else {
    if (zPatternIndices.length >= 2) {
      const sources = zPatternIndices[0] as number[];
      const targets_array = zPatternIndices[1] as number[];
      for (let i = 0; i < Math.min(sources.length, targets_array.length, zPatternValues.length); i++) {
        const source = sources[i];
        const target = targets_array[i];
        const value = zPatternValues[i] || 0;
        if (source === sourceSquare) targets.push({ square: target, strength: value });
      }
    }
  }

  return targets
    .sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))
    .slice(0, 8);
};

const getZPatternSourceSquares = (zPatternIndices?: number[][], zPatternValues?: number[]) => {
  const sources = new Set<number>();
  if (!zPatternIndices || !zPatternValues) return sources;

  const looksLikePairList = Array.isArray(zPatternIndices[0]) && (zPatternIndices[0] as number[]).length === 2;
  if (looksLikePairList) {
    for (const pair of zPatternIndices) {
      if (Array.isArray(pair) && pair.length >= 2) sources.add(Number(pair[0]));
    }
  } else if (zPatternIndices.length >= 2) {
    const sourceList = zPatternIndices[0] as number[];
    for (const source of sourceList) sources.add(Number(source));
  }
  return sources;
};

// Parse chess move string
const parseMove = (move: string) => {
  if (!move || move.length < 4) return null;
  
  const fromSquare = move.substring(0, 2);
  const toSquare = move.substring(2, 4);
  
  const parseSquare = (square: string) => {
    if (square.length !== 2) return null;
    const col = square.charCodeAt(0) - 97; // a=0, b=1, etc.
    const row = 8 - parseInt(square[1]); // 8=0, 7=1, etc.
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    return { row, col, index: row * 8 + col };
  };
  
  const from = parseSquare(fromSquare);
  const to = parseSquare(toSquare);
  
  if (!from || !to) return null;
  
  return {
    from,
    to,
    moveString: `${fromSquare}${toSquare}`
  };
};

export const ChessBoard: React.FC<ChessBoardProps> = ({
  fen,
  activations,
  zPatternIndices,
  zPatternValues,
  sampleIndex,
  analysisName,
  size = 'medium',
  showCoordinates = true,
  move,
  flip_activation = true,
  onMove,
  onSquareClick,
  isInteractive = false,
  autoFlipWhenBlack = false,
  moveColor,
  showSelfPlay = false,
  disableAutoAnalyze = false,
  wdlLoadDelayIndex,
}) => {

  // Hover state
  const [hoveredSquare, setHoveredSquare] = useState<number | null>(null);
  
  // Click selection state
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [boardEvaluation, setBoardEvaluation] = useState<number[] | null>(null);
  const [boardAnalysisLoading, setBoardAnalysisLoading] = useState(false);
  const [boardAnalysisError, setBoardAnalysisError] = useState<string | null>(null);

  // Self-play state
  const [selfPlayData, setSelfPlayData] = useState<any>(null);
  const [selfPlayLoading, setSelfPlayLoading] = useState(false);
  const [selfPlayError, setSelfPlayError] = useState<string | null>(null);
  const [currentSelfPlayStep, setCurrentSelfPlayStep] = useState(0);
  const [showSelfPlayPanel, setShowSelfPlayPanel] = useState(false);

  // Forward inference state
  const [forwardInferenceData, setForwardInferenceData] = useState<any>(null);
  const [forwardInferenceLoading, setForwardInferenceLoading] = useState(false);
  const [forwardInferenceError, setForwardInferenceError] = useState<string | null>(null);

  // WDL only: one forward pass via /analyze/board (do NOT use logit_lens here)
  const loadWdl = useCallback(async () => {
    setBoardAnalysisLoading(true);
    setBoardAnalysisError(null);
    try {
      const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/analyze/board`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen })
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setBoardEvaluation(data.evaluation ?? null);
    } catch (error) {
      const msg = error instanceof Error ? error.message : "WDL request failed";
      setBoardAnalysisError(msg);
      console.error(`[CB#${sampleIndex ?? 'NA'}] Position WDL failed:`, error);
    } finally {
      setBoardAnalysisLoading(false);
    }
  }, [fen, sampleIndex]);

  const handleAnalyze = useCallback(() => {
    if (disableAutoAnalyze) return;
    loadWdl();
  }, [disableAutoAnalyze, loadWdl]);

  // Defer auto WDL to next tick so the board renders first
  useEffect(() => {
    if (disableAutoAnalyze) return;
    const t = setTimeout(() => handleAnalyze(), 0);
    return () => clearTimeout(t);
  }, [handleAnalyze, disableAutoAnalyze]);

  // When disableAutoAnalyze + wdlLoadDelayIndex: after all boards render, load WDL one-by-one (index * 1.5s)
  const [wdlAutoScheduled, setWdlAutoScheduled] = useState(false);
  const WDL_STAGGER_MS = 1500;
  useEffect(() => {
    if (!disableAutoAnalyze || wdlLoadDelayIndex === undefined) return;
    setWdlAutoScheduled(true);
    const delay = wdlLoadDelayIndex * WDL_STAGGER_MS;
    const t = setTimeout(() => loadWdl(), delay);
    return () => {
      clearTimeout(t);
      setWdlAutoScheduled(false);
    };
  }, [disableAutoAnalyze, wdlLoadDelayIndex, loadWdl]);

  // Forward inference
  const handleForwardInference = useCallback(async () => {
    setForwardInferenceLoading(true);
    setForwardInferenceError(null);

    try {
      console.log(`[CB#${sampleIndex ?? 'NA'}] Starting forward inference: ${fen.substring(0, 50)}...`);
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/logit_lens/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          fen: fen,
          topk_vocab: 2000,
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      console.log(`[CB#${sampleIndex ?? 'NA'}] Forward inference done:`, result);
      const preds = Array.isArray(result?.final_layer_predictions) ? result.final_layer_predictions : [];
      const moves = preds.slice(0, 5).map((p: any) => ({
        move: p?.uci,
        logit: typeof p?.score === 'number' ? p.score : p?.logit,
        probability: typeof p?.prob === 'number' ? p.prob : p?.probability,
      }));
      setForwardInferenceData({ moves });

    } catch (err) {
      console.error(`[CB#${sampleIndex ?? 'NA'}] Forward inference failed:`, err);
      setForwardInferenceError(err instanceof Error ? err.message : 'Forward inference failed');
    } finally {
      setForwardInferenceLoading(false);
    }
  }, [fen, sampleIndex]);

  // Self-play handler
  const handleSelfPlay = useCallback(async () => {
    setSelfPlayLoading(true);
    setSelfPlayError(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/self_play`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          initial_fen: fen,
          max_moves: 5, // fixed 5 moves
          temperature: 1.0
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const result = await response.json();
      setSelfPlayData(result);
      setCurrentSelfPlayStep(0);
      setShowSelfPlayPanel(true);

    } catch (err) {
      console.error('Self-play failed:', err);
      setSelfPlayError(err instanceof Error ? err.message : 'Self-play failed');
    } finally {
      setSelfPlayLoading(false);
    }
  }, [fen]);

  // Chart data for WDL
  const chartData = useMemo(() => {
    if (!selfPlayData || !selfPlayData.wdl_history.length) return [];
    
    return selfPlayData.wdl_history.map((wdl: any, index: number) => {
      // Determine side to move and convert to fixed white/black win rates
      const currentFen = selfPlayData.positions[index];
      const isWhiteToMove = currentFen.includes(' w ');
      
      let whiteWinRate, blackWinRate;
      if (isWhiteToMove) {
        // White to move: win=white win rate, loss=black win rate
        whiteWinRate = wdl.win;
        blackWinRate = wdl.loss;
      } else {
        // Black to move: win=black win rate, loss=white win rate
        whiteWinRate = wdl.loss;
        blackWinRate = wdl.win;
      }
      
      return {
        move: index + 1,
        whiteWin: (whiteWinRate * 100).toFixed(1),
        draw: (wdl.draw * 100).toFixed(1),
        blackWin: (blackWinRate * 100).toFixed(1),
        whiteWin_num: whiteWinRate,
        draw_num: wdl.draw,
        blackWin_num: blackWinRate
      };
    });
  }, [selfPlayData]);

  const parsedBoard = useMemo(() => {
    try {
      return parseFEN(fen);
    } catch (error) {
      console.error('Failed to parse FEN:', error);
      return null;
    }
  }, [fen]);

  if (!parsedBoard) {
    return (
      <div className="p-4 border border-red-200 rounded-lg bg-red-50">
        <p className="text-red-700 text-sm">Invalid FEN format: {fen}</p>
      </div>
    );
  }

  const { board, isWhiteToMove } = parsedBoard;

  // Flip board vertically when Black to move (if autoFlipWhenBlack enabled)
  const flip = autoFlipWhenBlack ? !isWhiteToMove : false;

  // Display board: 1) flip vertically if flip; 2) mirror rows only in black view
  const displayBoard = useMemo(() => {
    const baseBoard = flip ? [...board].reverse() : board;
    return flip ? baseBoard.map((row) => [...row].reverse()) : baseBoard;
  }, [board, flip]);

  // Parse move
  const parsedMove = useMemo(() => {
    return move ? parseMove(move) : null;
  }, [move]);

  const maxAbsActivation = useMemo(() => {
    if (!activations?.length) return 0;
    return activations.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
  }, [activations]);

  // Display position -> actual square index (flip + mirror)
  const getActualSquareIndex = (displayRow: number, col: number): number => {
    const actualRow = flip ? (7 - displayRow) : displayRow;
    const actualCol = flip ? (7 - col) : col;
    return getSquareIndex(actualRow, actualCol);
  };

  // Activation index: array in standard board order a1=0..h8=63
  const getActivationIndex = (displayRow: number, col: number): number => {
    const originalRow = flip ? (7 - displayRow) : displayRow;
    const actualCol = flip ? (7 - col) : col;

    if (flip_activation) {
      return originalRow * 8 + actualCol;
    } else {
      const standardRow = 7 - originalRow;
      return standardRow * 8 + actualCol;
    }
  };

  // Actual index -> display position (flip + mirror)
  const getBoardDisplayPosition = (actualSquareIndex: number) => {
    const actualRow = Math.floor(actualSquareIndex / 8);
    const actualCol = actualSquareIndex % 8;
    const displayRow = flip ? (7 - actualRow) : actualRow;
    const displayCol = flip ? (7 - actualCol) : actualCol;
    return { row: displayRow, col: displayCol };
  };

  // Display row index -> rank (1-8)
  const getDisplayRowNumber = (displayRowIndex: number) => {
    if (flip) return displayRowIndex + 1;
    return 8 - displayRowIndex;
  };

  // Display col index -> file (a-h)
  const getDisplayColLetter = (colIndex: number) => {
    if (flip) return String.fromCharCode(104 - colIndex);
    return String.fromCharCode(97 + colIndex);
  };

  // Activation index -> standard square name (e.g. a1)
  const getSquareNameFromActivationIndex = (activationIndex: number) => {
    const row = Math.floor(activationIndex / 8);
    const col = activationIndex % 8;
    if (flip_activation) {
      return `${String.fromCharCode(97 + col)}${8 - row}`;
    } else {
      return `${String.fromCharCode(97 + col)}${row + 1}`;
    }
  };

  // Handle square click (select from / to, or move)
  const handleSquareClick = (activationIndex: number) => {
    if (!isInteractive) return;
    
    const squareName = getSquareNameFromActivationIndex(activationIndex);
    console.log('Square clicked:', squareName, 'activation index:', activationIndex);
    
    if (selectedSquare === null) {
      setSelectedSquare(activationIndex);
      onSquareClick?.(squareName);
      console.log('From square selected:', squareName);
    } else if (selectedSquare === activationIndex) {
      setSelectedSquare(null);
      console.log('Selection cleared');
    } else {
      const fromSquare = getSquareNameFromActivationIndex(selectedSquare);
      const moveString = `${fromSquare}${squareName}`;
      
      console.log('Attempting move:', moveString);
      
      onMove?.(moveString);
      
      setSelectedSquare(null);
    }
  };

  // Size-dependent styles
  const sizeClasses = {
    small: 'w-64 h-64',
    medium: 'w-80 h-80',
    large: 'w-96 h-96'
  };

  const squareSize = {
    small: 'w-8 h-8',
    medium: 'w-10 h-10',
    large: 'w-12 h-12'
  };

  const squareSizePxMap = {
    small: 32,   // 256px / 8
    medium: 40,  // 320px / 8
    large: 48,   // 384px / 8
  } as const;
  const boardPx = squareSizePxMap[size] * 8;

  // Z-pattern targets for hovered square
  const zPatternTargets = hoveredSquare !== null ? getZPatternTargets(hoveredSquare, zPatternIndices, zPatternValues) : [];
  const zPatternSourceSquares = getZPatternSourceSquares(zPatternIndices, zPatternValues);

  // Forward inference moves for arrows
  const forwardMoves = useMemo(() => {
    if (!forwardInferenceData || !forwardInferenceData.moves) return [];

    return forwardInferenceData.moves.map((moveData: any) => {
      const parsedMove = parseMove(moveData.move);
      if (!parsedMove) return null;

      return {
        ...parsedMove,
        logit: moveData.logit,
        probability: moveData.probability
      };
    }).filter(Boolean);
  }, [forwardInferenceData]);

  return (
    <div className="flex flex-col items-center space-y-2">
      {/* Board info */}
      <div className="text-sm text-gray-600 text-center">
        {sampleIndex !== undefined && (
          <div>Sample #{sampleIndex}</div>
        )}
        {analysisName && (
          <div>Analysis: {analysisName}</div>
        )}
        <div className="mt-1">
          <span className={`inline-block w-3 h-3 rounded-full mr-1 ${
            isWhiteToMove ? 'bg-white border-2 border-gray-800' : 'bg-gray-800'
          }`}></span>
          {isWhiteToMove ? 'White to move' : 'Black to move'}
        </div>
        {isInteractive && selectedSquare !== null && (
          <div className="mt-1 text-blue-600 font-medium">
            Selected: {getSquareNameFromActivationIndex(selectedSquare)}
          </div>
        )}
      </div>

      {/* Board container */}
      <div className={`relative ${sizeClasses[size]} border-4 border-gray-800 rounded-lg overflow-hidden shadow-lg`}>
        {/* Board grid */}
        <div className="w-full h-full grid grid-cols-8 grid-rows-8">
          {displayBoard.map((row, displayRowIndex) =>
            row.map((_, colIndex) => {
              const piece = row[colIndex];
              // Square color: a1=dark, h1=light on standard board
              const normalIsLight = (displayRowIndex + colIndex) % 2 === 0;
              const isLight = normalIsLight;
              const squareIndex = getActualSquareIndex(displayRowIndex, colIndex);
              const activationIndex = getActivationIndex(displayRowIndex, colIndex);
              const activation = activations?.[activationIndex] || 0;
              const activationColor = getActivationColor(activation, maxAbsActivation);
              const hasZPatternSource = zPatternSourceSquares.has(activationIndex);
              
              const isZPatternTarget = zPatternTargets.some(target => target.square === activationIndex);
              const targetStrength = zPatternTargets.find(target => target.square === activationIndex)?.strength || 0;
              
              const isMoveFromSquare = parsedMove && parsedMove.from.index === squareIndex;
              const isMoveToSquare = parsedMove && parsedMove.to.index === squareIndex;
              
              const isSelectedSquare = selectedSquare === activationIndex;
              
              const baseColor = isLight ? 'bg-[#F0F0F0]' : 'bg-[#D1D1D1]';
              
              let finalBackgroundColor;
              const isSourceSquare = hoveredSquare === activationIndex;
              
              if (isSelectedSquare) {
                finalBackgroundColor = 'rgba(59, 130, 246, 0.8)';
              } else if (isMoveFromSquare) {
                finalBackgroundColor = (moveColor || 'rgba(34, 197, 94, 0.7)');
              } else if (isMoveToSquare) {
                finalBackgroundColor = (moveColor || 'rgba(22, 163, 74, 0.8)');
              } else if (isZPatternTarget) {
                const absStrength = Math.abs(targetStrength);
                const normalizedStrength = Math.min(absStrength / 0.01, 1);
                const opacity = Math.max(0.3, normalizedStrength * 0.7 + 0.3);
                
                if (targetStrength > 0) {
                  finalBackgroundColor = `rgba(59, 130, 246, ${opacity})`;
                } else {
                  finalBackgroundColor = `rgba(249, 115, 22, ${opacity})`;
                }
              } else if (activationColor !== 'transparent' && !isSourceSquare && hoveredSquare === null) {
                finalBackgroundColor = activationColor;
              }
              
              return (
                <div
                  key={`${displayRowIndex}-${colIndex}`}
                  className={`
                    ${squareSize[size]} 
                    relative flex items-center justify-center
                    ${baseColor}
                    transition-all duration-200 hover:brightness-110
                    ${activation !== 0 || hasZPatternSource ? 'cursor-pointer' : ''}
                    ${isInteractive ? 'cursor-pointer' : ''}
                    ${isSelectedSquare ? 'ring-2 ring-blue-500 ring-opacity-75' : ''}
                  `}
                  style={{
                    backgroundColor: finalBackgroundColor,
                  }}
                  onMouseEnter={() => {
                    if (activation !== 0 || hasZPatternSource) {
                      setHoveredSquare(activationIndex);
                    }
                  }}
                  onMouseLeave={() => {
                    setHoveredSquare(null);
                  }}
                  onClick={() => handleSquareClick(activationIndex)}
                  title={`${getDisplayColLetter(colIndex)}${getDisplayRowNumber(displayRowIndex)}${
                    activation !== 0 ? ` (${activation.toFixed(3)})` : ''
                  }${
                    isZPatternTarget ? ` [target: ${targetStrength.toFixed(3)}]` : ''
                  }${
                    isMoveFromSquare ? ' [from]' : ''
                  }${
                    isMoveToSquare ? ' [to]' : ''
                  }${
                    isSelectedSquare ? ' [selected]' : ''
                  }`}
                >
                  {/* Piece: ~93% of square size */}
                  {piece && (
                    <img
                      src={getPieceSrc(piece)}
                      alt={piece}
                      className="piece w-[93%] h-[93%] object-contain select-none pointer-events-none"
                      draggable={false}
                    />
                  )}

                  {/* Activation value (hidden on source square when hovered) */}
                  {activation !== 0 && !isSourceSquare && hoveredSquare === null && (
                    <div className="absolute top-0 right-0 bg-blue-600 text-white text-xs rounded px-1 leading-3" style={{ fontSize: '10px' }}>
                      {Math.abs(activation).toFixed(2)}
                    </div>
                  )}

                  {/* Z-pattern value (top-left) */}
                  {isZPatternTarget && (
                    <div className={`absolute top-0 left-0 text-white text-xs rounded px-1 leading-3 ${
                      targetStrength > 0 ? 'bg-blue-700' : 'bg-orange-700'
                    }`} style={{ fontSize: '10px' }}>
                      {targetStrength.toFixed(3)}
                    </div>
                  )}

                  {/* Coordinates: rank right (1-8), file bottom (a-h) */}
                  {showCoordinates && (
                    <>
                      {colIndex === 7 ? (
                        <div className={`absolute right-1 top-1 text-xs font-bold ${isLight ? 'text-gray-700' : 'text-gray-200'}`}>
                          {getDisplayRowNumber(displayRowIndex)}
                        </div>
                      ) : null}
                      {displayRowIndex === 7 ? (
                        <div className={`absolute bottom-1 left-1 text-xs font-bold ${isLight ? 'text-gray-700' : 'text-gray-200'}`}>
                          {getDisplayColLetter(colIndex)}
                        </div>
                      ) : null}
                    </>
                  )}
                </div>
              );
            })
          )}
        
        {/* Move arrow overlay */}
        {parsedMove && (
          <svg
            className="absolute inset-0 pointer-events-none"
            width={boardPx}
            height={boardPx}
            viewBox={`0 0 ${boardPx} ${boardPx}`}
            style={{ zIndex: 10 }}
          >
          <defs>
            <marker id="arrow-head" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L6,3 z" fill={moveColor || '#4b5563'} />
            </marker>
            <marker id="forward-arrow-head" markerWidth="6" markerHeight="6" refX="5" refY="2.5" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,5 L5,2.5 z" fill="#10b981" />
            </marker>
          </defs>
            {(() => {
              const fromPos = getBoardDisplayPosition(parsedMove.from.index);
              const toPos = getBoardDisplayPosition(parsedMove.to.index);
              const sq = squareSizePxMap[size];
              const fromX = fromPos.col * sq + sq / 2;
              const fromY = fromPos.row * sq + sq / 2;
              const toX = toPos.col * sq + sq / 2;
              const toY = toPos.row * sq + sq / 2;
              return (
                <line
                  x1={fromX}
                  y1={fromY}
                  x2={toX}
                  y2={toY}
                  stroke={moveColor || '#4b5563'}
                  strokeWidth={4}
                  strokeOpacity={0.9}
                  markerEnd="url(#arrow-head)"
                />
              );
            })()}

          {/* Forward inference move arrows */}
          {forwardMoves.map((moveData: any, index: number) => {
            const fromPos = getBoardDisplayPosition(moveData.from.index);
            const toPos = getBoardDisplayPosition(moveData.to.index);
            const sq = squareSizePxMap[size];
            const fromX = fromPos.col * sq + sq / 2;
            const fromY = fromPos.row * sq + sq / 2;
            const toX = toPos.col * sq + sq / 2;
            const toY = toPos.row * sq + sq / 2;

            // Arrow thickness by logit (higher logit = thicker)
            const minLogit = Math.min(...forwardMoves.map((m: any) => m.logit));
            const maxLogit = Math.max(...forwardMoves.map((m: any) => m.logit));
            const normalizedLogit = (moveData.logit - minLogit) / (maxLogit - minLogit);
            const strokeWidth = 2 + normalizedLogit * 4;

            return (
              <line
                key={`forward-${index}`}
                x1={fromX}
                y1={fromY}
                x2={toX}
                y2={toY}
                stroke="#10b981"
                strokeWidth={strokeWidth}
                strokeOpacity={0.8}
                markerEnd="url(#forward-arrow-head)"
              />
            );
          })}
        </svg>
        )}
      </div>

              {/* FEN and move info */}
      <div className="absolute -bottom-12 left-0 right-0 text-xs text-gray-500 text-center space-y-1">
        {parsedMove && (
          <div className="text-green-600 font-medium">
            Move: {parsedMove.moveString}
          </div>
        )}
        <div className="truncate">
          {fen}
        </div>
      </div>
      </div>

      {/* FEN string */}
      <div className="w-full max-w-lg text-xs text-gray-600 bg-gray-50 rounded p-2 border">
        <div className="font-medium text-gray-800 mb-1">FEN string:</div>
        <div className="font-mono text-xs break-all select-all">
          {fen}
        </div>
      </div>
      <div className="mt-2">
        {boardEvaluation ? (
          <div className="mt-1 text-sm text-gray-700">
            w:{boardEvaluation[0].toFixed(2)}, d:{boardEvaluation[1].toFixed(2)}, l:{boardEvaluation[2].toFixed(2)}
          </div>
        ) : boardAnalysisLoading ? (
          <div className="mt-1 text-sm text-gray-500">Loading WDL...</div>
        ) : boardAnalysisError ? (
          <div className="mt-1 text-sm text-red-600">{boardAnalysisError}</div>
        ) : disableAutoAnalyze && wdlLoadDelayIndex !== undefined && wdlAutoScheduled ? (
          <div className="mt-1 text-sm text-gray-500">Loading WDL...</div>
        ) : disableAutoAnalyze ? (
          <button
            type="button"
            onClick={() => loadWdl()}
            className="mt-1 text-sm text-blue-600 hover:underline"
          >
            Get WDL
          </button>
        ) : (
          <div className="mt-1 text-sm text-gray-700">Analyzing position...</div>
        )}
      </div>

      {/* Activation stats */}
      {activations && activations.some(a => a !== 0) && (
        <div className="text-xs text-gray-600 space-y-1">
          <div className="flex space-x-4">
            <span>Activated squares: {activations.filter(a => a !== 0).length}</span>
            <span>Max: {Math.max(...activations.map(Math.abs)).toFixed(3)}</span>
          </div>
        </div>
      )}

      {/* Hover: Z-pattern connections */}
      {hoveredSquare !== null && (() => {
        const squareName = getSquareNameFromActivationIndex(hoveredSquare);
        const activation = activations?.[hoveredSquare] || 0;
        
        return (
          <div className="text-xs bg-blue-50 border border-blue-200 rounded p-2 space-y-1">
            <div className="font-medium text-blue-800">
              Square {squareName} ({activation.toFixed(3)})
            </div>
            {zPatternTargets.length > 0 ? (
              <>
                <div className="text-blue-700">({zPatternTargets.length} targets):</div>
                <div className="grid grid-cols-2 gap-1">
                  {zPatternTargets.slice(0, 6).map((target, idx) => {
                    const targetName = getSquareNameFromActivationIndex(target.square);
                    const isPositive = target.strength > 0;
                    return (
                      <div key={idx} className={isPositive ? "text-blue-600" : "text-orange-600"}>
                        → {targetName} ({target.strength.toFixed(3)})
                      </div>
                    );
                  })}
                  {zPatternTargets.length > 6 && (
                    <div className="text-blue-500 col-span-2">... and {zPatternTargets.length - 6} more</div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-blue-600">No connections</div>
            )}
          </div>
        );
      })()}

      {/* Forward inference and self-play */}
      <div className="mt-6 space-y-4">
        <div className="flex justify-center space-x-4">
          <button
            onClick={handleForwardInference}
            disabled={forwardInferenceLoading}
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {forwardInferenceLoading ? 'Running...' : 'Run forward'}
          </button>
          {showSelfPlay && (
            <>
              <button
                onClick={handleSelfPlay}
                disabled={selfPlayLoading}
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {selfPlayLoading ? 'Running...' : 'Start self-play (5 moves)'}
              </button>
              {selfPlayData && (
                <button
                  onClick={() => setShowSelfPlayPanel(!showSelfPlayPanel)}
                  className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
                >
                  {showSelfPlayPanel ? 'Hide results' : 'Show results'}
                </button>
              )}
            </>
          )}
        </div>

        {forwardInferenceError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-700 text-sm">{forwardInferenceError}</p>
          </div>
        )}

        {forwardInferenceData && forwardInferenceData.moves && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-green-800 mb-2">Forward inference (Top 5 moves)</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-2">
              {forwardInferenceData.moves.map((moveData: any, index: number) => (
                <div key={index} className="bg-white rounded p-2 text-center border">
                  <div className="font-mono text-sm font-medium">{moveData.move}</div>
                  <div className="text-xs text-gray-600">
                    {(moveData.probability * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    logit: {moveData.logit?.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {showSelfPlay && (
          <>
            {selfPlayError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-700 text-sm">{selfPlayError}</p>
            </div>
            )}

            {selfPlayData && showSelfPlayPanel && (
            <div className="space-y-4">
              <div className="bg-white rounded-lg border p-4">
                <h3 className="text-lg font-semibold mb-3">Self-play step navigation</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Step: {currentSelfPlayStep + 1} / {selfPlayData.positions.length}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max={selfPlayData.positions.length - 1}
                      value={currentSelfPlayStep}
                      onChange={(e) => setCurrentSelfPlayStep(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                  
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <div className="text-sm font-medium text-blue-800 mb-2">Step {currentSelfPlayStep + 1} info:</div>
                    <div className="space-y-1 text-xs text-blue-700">
                      {currentSelfPlayStep > 0 && (
                        <div>
                          <span className="font-medium">Move played:</span> 
                          <span className="font-mono ml-1 px-1 bg-blue-100 rounded">
                            {selfPlayData.moves[currentSelfPlayStep - 1]}
                          </span>
                        </div>
                      )}
                      <div>
                        <span className="font-medium">Position FEN:</span>
                        <div className="font-mono text-xs mt-1 p-2 bg-white rounded border break-all">
                          {selfPlayData.positions[currentSelfPlayStep]}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex justify-center space-x-2">
                    <button
                      onClick={() => setCurrentSelfPlayStep(0)}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                    >
                      Start
                    </button>
                    <button
                      onClick={() => setCurrentSelfPlayStep(Math.max(0, currentSelfPlayStep - 1))}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                      disabled={currentSelfPlayStep === 0}
                    >
                      Previous
                    </button>
                    <button
                      onClick={() => setCurrentSelfPlayStep(Math.min(selfPlayData.positions.length - 1, currentSelfPlayStep + 1))}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                      disabled={currentSelfPlayStep === selfPlayData.positions.length - 1}
                    >
                      Next
                    </button>
                    <button
                      onClick={() => setCurrentSelfPlayStep(selfPlayData.positions.length - 1)}
                      className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors text-sm"
                    >
                      End
                    </button>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg border p-4">
                <h3 className="text-lg font-semibold mb-3">
                  Self-play position {currentSelfPlayStep + 1}
                  {currentSelfPlayStep < selfPlayData.moves.length && (
                    <span className="text-green-600 ml-2">
                      (Move: {selfPlayData.moves[currentSelfPlayStep]})
                    </span>
                  )}
                </h3>
                
                <div className="mb-4 p-3 bg-gray-50 rounded-lg border">
                  <div className="text-sm font-medium text-gray-700 mb-2">Current FEN:</div>
                  <div className="font-mono text-xs text-gray-800 break-all select-all bg-white p-2 rounded border">
                    {selfPlayData.positions[currentSelfPlayStep]}
                  </div>
                </div>
                <div className="flex justify-center">
                  <div className="relative w-80 h-80 border-4 border-gray-800 rounded-lg overflow-hidden shadow-lg">
                    <div className="w-full h-full grid grid-cols-8 grid-rows-8">
                      {(() => {
                        try {
                          const parsedBoard = parseFEN(selfPlayData.positions[currentSelfPlayStep]);
                          const currentMove = currentSelfPlayStep > 0 ? selfPlayData.moves[currentSelfPlayStep - 1] : undefined;
                          const parsedMove = currentMove ? parseMove(currentMove) : null;
                          
                          return parsedBoard.board.map((row, displayRowIndex) =>
                            row.map((piece, colIndex) => {
                              const isLight = (displayRowIndex + colIndex) % 2 === 0;
                              const squareIndex = displayRowIndex * 8 + colIndex;
                              const isMoveFromSquare = parsedMove && parsedMove.from.index === squareIndex;
                              const isMoveToSquare = parsedMove && parsedMove.to.index === squareIndex;
                              
                              return (
                                <div
                                  key={`${displayRowIndex}-${colIndex}`}
                                  className={`
                                    w-10 h-10 relative flex items-center justify-center
                                    ${isLight ? 'bg-[#F0F0F0]' : 'bg-[#D1D1D1]'}
                                    ${isMoveFromSquare ? 'bg-green-400' : ''}
                                    ${isMoveToSquare ? 'bg-green-600' : ''}
                                  `}
                                >
                                  {piece && (
                                    <img
                                      src={getPieceSrc(piece)}
                                      alt={piece}
                                      className="piece w-[93%] h-[93%] object-contain select-none pointer-events-none"
                                      draggable={false}
                                    />
                                  )}
                                </div>
                              );
                            })
                          );
                        } catch (error) {
                          return <div className="col-span-8 row-span-8 flex items-center justify-center text-red-500">Invalid FEN</div>;
                        }
                      })()}
                    </div>
                    
                    {/* Move arrow */}
                    {(() => {
                      const currentMove = currentSelfPlayStep > 0 ? selfPlayData.moves[currentSelfPlayStep - 1] : undefined;
                      const parsedMove = currentMove ? parseMove(currentMove) : null;
                      if (!parsedMove) return null;
                      
                      const fromX = parsedMove.from.col * 40 + 20;
                      const fromY = parsedMove.from.row * 40 + 20;
                      const toX = parsedMove.to.col * 40 + 20;
                      const toY = parsedMove.to.row * 40 + 20;
                      
                      return (
                        <svg className="absolute inset-0 pointer-events-none" width="320" height="320">
                          <defs>
                            <marker id="arrow-head-selfplay" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth">
                              <path d="M0,0 L0,6 L6,3 z" fill="#3b82f6" />
                            </marker>
                          </defs>
                          <line
                            x1={fromX}
                            y1={fromY}
                            x2={toX}
                            y2={toY}
                            stroke="#3b82f6"
                            strokeWidth={3}
                            markerEnd="url(#arrow-head-selfplay)"
                          />
                        </svg>
                      );
                    })()}
                  </div>
                </div>
              </div>

              {/* WDL curve chart */}
              {chartData.length > 0 && (
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-lg font-semibold mb-3">WDL Curve</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="move" />
                        <YAxis />
                        <Tooltip 
                          formatter={(value: any, name: string) => [
                            `${parseFloat(value).toFixed(1)}%`, 
                            name === 'whiteWin' ? 'White win rate' : name === 'draw' ? 'Draw rate' : name === 'blackWin' ? 'Black win rate' : name
                          ]}
                        />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="whiteWin" 
                          stroke="#10b981" 
                          strokeWidth={2}
                          name="White win rate"
                          dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="draw" 
                          stroke="#f59e0b" 
                          strokeWidth={2}
                          name="Draw rate"
                          dot={{ fill: '#f59e0b', strokeWidth: 2, r: 3 }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="blackWin" 
                          stroke="#ef4444" 
                          strokeWidth={2}
                          name="Black win rate"
                          dot={{ fill: '#ef4444', strokeWidth: 2, r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Current step WDL evaluation */}
              {currentSelfPlayStep < selfPlayData.wdl_history.length && (
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-lg font-semibold mb-3">Step {currentSelfPlayStep + 1} WDL Evaluation</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {(() => {
                      const currentFen = selfPlayData.positions[currentSelfPlayStep];
                      const currentIsWhiteToMove = currentFen.includes(' w ');
                      
                      if (currentIsWhiteToMove) {
                        return (
                          <>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-green-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].win * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">White win rate</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-yellow-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].draw * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">Draw rate</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-red-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].loss * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">Black win rate</div>
                            </div>
                          </>
                        );
                      } else {
                        return (
                          <>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-green-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].win * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">Black win rate</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-yellow-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].draw * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">Draw rate</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-red-600">
                                {(selfPlayData.wdl_history[currentSelfPlayStep].loss * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm text-gray-600">White win rate</div>
                            </div>
                          </>
                        );
                      }
                    })()}
                  </div>
                </div>
              )}

              {/* Move probabilities */}
              {currentSelfPlayStep < selfPlayData.move_probabilities.length && 
               Object.keys(selfPlayData.move_probabilities[currentSelfPlayStep]).length > 0 && (
                <div className="bg-white rounded-lg border p-4">
                  <h3 className="text-lg font-semibold mb-3">Step {currentSelfPlayStep + 1} Move Probabilities</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                    {Object.entries(selfPlayData.move_probabilities[currentSelfPlayStep])
                      .sort(([,a], [,b]) => (b as number) - (a as number))
                      .slice(0, 10)
                      .map(([move, prob]) => (
                        <div key={move} className="bg-gray-50 rounded p-2 text-center">
                          <div className="font-mono text-sm font-medium">{move}</div>
                          <div className="text-xs text-gray-600">
                            {((prob as number) * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
