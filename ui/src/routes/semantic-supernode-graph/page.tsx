import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Edge as ReactFlowEdge,
  Handle,
  MarkerType,
  Node as ReactFlowNode,
  NodeProps,
  Position,
  ReactFlow,
  reconnectEdge,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import { AppNavbar } from "@/components/app/navbar";
import { ChessBoard } from "@/components/chess/chess-board";
import { AppPagination } from "@/components/ui/pagination";
import {
  fetchFeatureByDictionaryName,
  fetchSemanticSupernodeGraphExample,
  fetchSemanticSupernodeGraphFromPath,
  normalizeSemanticSupernodeGraph,
  parseSemanticSupernodeGraphFile,
} from "@/utils/api";
import { Feature } from "@/types/feature";
import { normalizeZPattern } from "@/utils/activationUtils";
import { extractFenFromText, validateFen } from "@/utils/fenUtils";
import {
  SemanticChildFeature,
  SemanticSupernode,
  SemanticSupernodeGraph,
} from "@/types/semantic-supernode-graph";

type SemanticNodeData = {
  supernode: SemanticSupernode;
  childCount: number;
  selected: boolean;
};

const DEFAULT_SEMANTIC_GRAPH_PATH = "scripts/Attribution_Graph/semantic_supernode_examples/lookingahead_k30_e16.semantic_supernodes.json";
const LOCAL_LOOKING_AHEAD_GRAPH_ASSET = "semantic_supernode_graphs/lookingahead_k30_e16.semantic_supernodes.json";

const getLocalLookingAheadGraphUrl = () => {
  const appBasePath = window.location.pathname.replace(/\/semantic-supernode-graph.*$/, "");
  return `${appBasePath}/${LOCAL_LOOKING_AHEAD_GRAPH_ASSET}`;
};

const fetchLocalLookingAheadGraph = async (): Promise<SemanticSupernodeGraph> => {
  const response = await fetch(getLocalLookingAheadGraphUrl());
  if (!response.ok) {
    throw new Error(`Failed to load bundled semantic graph: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

type ChessTopActivationSample = {
  fen: string;
  activationStrength: number;
  activations?: number[];
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  sampleIndex: number;
};

type CurrentBoardActivationState = {
  single?: {
    activations?: number[];
    zPatternIndices?: number[][];
    zPatternValues?: number[];
    position: number;
  };
  all?: {
    activations?: number[];
    zPatternIndices?: number[][];
    zPatternValues?: number[];
  };
};

type BoardInputToken = {
  square?: string;
  piece?: string;
  position?: number;
  label?: string;
};

const TYPE_COLORS: Record<string, { border: string; bg: string; text: string }> = {
  Det: { border: "#94a3b8", bg: "#f8fafc", text: "#334155" },
  Src: { border: "#64748b", bg: "#f1f5f9", text: "#334155" },
  Tgt: { border: "#f59e0b", bg: "#fff7ed", text: "#9a3412" },
  Mov: { border: "#22c55e", bg: "#f0fdf4", text: "#166534" },
  Tac: { border: "#a855f7", bg: "#faf5ff", text: "#6b21a8" },
  Pro: { border: "#06b6d4", bg: "#ecfeff", text: "#155e75" },
  Cap: { border: "#ef4444", bg: "#fef2f2", text: "#991b1b" },
  Val: { border: "#0ea5e9", bg: "#f0f9ff", text: "#075985" },
  Spa: { border: "#84cc16", bg: "#f7fee7", text: "#3f6212" },
  Reg: { border: "#78716c", bg: "#fafaf9", text: "#44403c" },
  Semantic: { border: "#3b82f6", bg: "#eff6ff", text: "#1d4ed8" },
};

const getTypeColor = (semanticType?: string | null) => {
  const key = semanticType || "Semantic";
  return TYPE_COLORS[key] || TYPE_COLORS.Semantic;
};

const SemanticSupernodeCard = ({ data }: NodeProps<ReactFlowNode<SemanticNodeData>>) => {
  const { supernode, childCount, selected } = data;
  const color = getTypeColor(supernode.semantic_type);
  const square =
    supernode.position?.board_square ||
    supernode.position?.board_squares?.[0] ||
    supernode.position?.label;

  return (
    <div
      className="min-w-[180px] max-w-[230px] rounded-xl border-2 px-3 py-2 shadow-sm"
      style={{
        borderColor: selected ? "#111827" : color.border,
        background: color.bg,
        color: color.text,
      }}
    >
      <Handle type="target" position={Position.Left} className="!h-2 !w-2 !bg-slate-500" />
      <Handle type="source" position={Position.Right} className="!h-2 !w-2 !bg-slate-500" />
      <div className="flex items-center justify-between gap-2">
        <span className="rounded-full bg-white/70 px-2 py-0.5 text-xs font-semibold">
          {supernode.role_id || supernode.semantic_type || "Semantic"}
        </span>
        {square ? <span className="text-xs font-medium">@ {square}</span> : null}
      </div>
      <div className="mt-1 text-sm font-semibold leading-snug">{supernode.label}</div>
      {supernode.interpretation ? (
        <div className="mt-1 line-clamp-2 text-xs opacity-80">{supernode.interpretation}</div>
      ) : null}
      <div className="mt-2 text-[11px] opacity-70">
        {childCount} feature{childCount === 1 ? "" : "s"}
        {supernode.position?.token_position != null
          ? ` · token ${supernode.position.token_position}`
          : ""}
      </div>
    </div>
  );
};

const nodeTypes = {
  semanticSupernode: SemanticSupernodeCard,
};

const boardFiles = ["a", "b", "c", "d", "e", "f", "g", "h"];

const squareToActivationIndex = (square: string): number | null => {
  if (!/^[a-h][1-8]$/.test(square)) return null;
  const fileIndex = boardFiles.indexOf(square[0]);
  const rank = Number(square[1]);
  return (8 - rank) * 8 + fileIndex;
};

const getFeaturePreferredPosition = (feature: SemanticChildFeature): number | null => {
  if (feature.token_position != null && Number.isFinite(feature.token_position)) {
    return feature.token_position;
  }
  if (feature.board_square) {
    return squareToActivationIndex(feature.board_square);
  }
  return null;
};

const extractChessTopActivationSamples = (feature: Feature | null): ChessTopActivationSample[] => {
  if (!feature?.sampleGroups?.length) return [];
  const samples: ChessTopActivationSample[] = [];

  for (const sampleGroup of feature.sampleGroups) {
    for (const [sampleIndex, sample] of sampleGroup.samples.entries()) {
      const fen = extractFenFromText(sample.text ?? "");
      if (!fen || !validateFen(fen)) continue;

      let activations: number[] | undefined;
      let activationStrength = 0;
      if (Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
        activations = Array(64).fill(0);
        for (let idx = 0; idx < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length); idx += 1) {
          const boardIndex = sample.featureActsIndices[idx];
          const value = sample.featureActsValues[idx];
          if (boardIndex >= 0 && boardIndex < 64) {
            activations[boardIndex] = value;
            if (Math.abs(value) > Math.abs(activationStrength)) {
              activationStrength = value;
            }
          }
        }
      }

      samples.push({
        fen,
        activationStrength,
        activations,
        ...normalizeZPattern(sample.zPatternIndices, sample.zPatternValues),
        sampleIndex,
      });
    }
  }

  return samples.sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength));
};

const ChessTopActivationBoards = ({ feature }: { feature: Feature }) => {
  const [page, setPage] = useState(1);
  const chessSamples = useMemo(() => extractChessTopActivationSamples(feature), [feature]);
  const pageSize = 4;
  const maxPage = Math.max(1, Math.ceil(chessSamples.length / pageSize));
  const currentSamples = useMemo(
    () => chessSamples.slice((page - 1) * pageSize, page * pageSize),
    [chessSamples, page],
  );

  useEffect(() => {
    setPage(1);
  }, [feature]);

  if (chessSamples.length === 0) {
    return (
      <div className="rounded border bg-slate-50 p-3 text-sm text-slate-500">
        No chessboard top activation samples were found for this feature.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="text-sm text-slate-600">
        Showing {chessSamples.length} top activation samples as chess boards.
      </div>
      <div className="grid grid-cols-1 gap-3">
        {currentSamples.map((sample, index) => (
          <div key={`${sample.sampleIndex}-${sample.fen}`} className="rounded-lg border bg-background p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs text-slate-500">Sample #{(page - 1) * pageSize + index + 1}</span>
              <span className="text-xs text-slate-500">Max act: {sample.activationStrength.toFixed(3)}</span>
            </div>
            <ChessBoard
              fen={sample.fen}
              size="small"
              showCoordinates
              activations={sample.activations}
              zPatternIndices={sample.zPatternIndices}
              zPatternValues={sample.zPatternValues}
              sampleIndex={sample.sampleIndex}
              analysisName="Top Activation"
              flip_activation={sample.fen.includes(" b ")}
              autoFlipWhenBlack
              disableAutoAnalyze
            />
          </div>
        ))}
      </div>
      {maxPage > 1 ? <AppPagination page={page} setPage={setPage} maxPage={maxPage} /> : null}
    </div>
  );
};

const ChessboardSummary = ({
  graph,
  selectedSupernode,
}: {
  graph: SemanticSupernodeGraph | null;
  selectedSupernode: SemanticSupernode | null;
}) => {
  const selectedHighlightSquares = [
    selectedSupernode?.position?.board_squares?.length
      ? selectedSupernode.position.board_squares[0]
      : selectedSupernode?.position?.board_square
        ? selectedSupernode.position.board_square
        : null,
  ].filter((square): square is string => Boolean(square && /^[a-h][1-8]$/.test(square)));
  const fen = graph?.board?.fen;

  return (
    <div className="rounded-lg border bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-semibold">Chessboard positions</div>
        {graph?.action?.san ? (
          <div className="text-sm font-semibold text-green-700">
            {graph.action.san}
            {graph.action.prob != null ? ` (${(graph.action.prob * 100).toFixed(1)}%)` : ""}
          </div>
        ) : null}
      </div>
      {fen ? (
        <ChessBoard
          fen={fen}
          size="small"
          showCoordinates
          highlightSquares={selectedHighlightSquares}
          highlightColor="rgba(250, 204, 21, 0.78)"
          move={graph?.action?.uci}
          analysisName="Semantic Supernode"
          disableAutoAnalyze
        />
      ) : (
        <div className="rounded border bg-slate-50 p-3 text-sm text-slate-500">No FEN provided.</div>
      )}
      {fen ? (
        <div className="mt-2 break-all text-xs text-slate-500">FEN: {fen}</div>
      ) : null}
    </div>
  );
};

const PIECE_SVG_NAMES: Record<string, string> = {
  K: "wK",
  Q: "wQ",
  R: "wR",
  B: "wB",
  N: "wN",
  P: "wP",
  k: "bK",
  q: "bQ",
  r: "bR",
  b: "bB",
  n: "bN",
  p: "bP",
};

const getPieceSvgSrc = (piece: string): string => {
  const name = PIECE_SVG_NAMES[piece] || piece;
  return new URL(`../../components/chess/pieces/${name}.svg`, import.meta.url).href;
};

const BoardSquareTile = ({
  square,
  token,
  active,
}: {
  square: string;
  token?: BoardInputToken;
  active?: boolean;
}) => {
  const piece = token?.piece;
  return (
    <div
      className={`relative flex h-14 w-14 shrink-0 items-center justify-center rounded border text-center shadow-sm ${
        active ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-white"
      }`}
    >
      <div className="absolute right-1 top-0.5 text-[10px] font-semibold text-slate-400">{square}</div>
      {piece ? (
        <img className="h-10 w-10 object-contain" src={getPieceSvgSrc(piece)} alt={`${piece} on ${square}`} />
      ) : (
        <div className="text-xs font-semibold text-slate-400">{square}</div>
      )}
      {token?.position != null ? (
        <div className="absolute bottom-0.5 left-1 text-[9px] text-slate-400">pos {token.position}</div>
      ) : null}
    </div>
  );
};

const TaxonomySupernodePanel = ({
  graph,
  selectedId,
  selectedTaxonomy,
  onSelectSupernode,
  onSelectTaxonomy,
}: {
  graph: SemanticSupernodeGraph | null;
  selectedId: string | null;
  selectedTaxonomy: string;
  onSelectSupernode: (id: string) => void;
  onSelectTaxonomy: (taxonomy: string) => void;
}) => {
  const tokens = ((graph?.board?.input_tokens as BoardInputToken[] | undefined) || []).filter((token) => token.square);
  const tokenBySquare = new Map(tokens.map((token) => [token.square, token]));
  const taxonomyOptions = Array.from(new Set((graph?.semantic_supernodes || []).map((node) => node.semantic_type || "Semantic")))
    .sort((a, b) => {
      const order = ["Det", "Mov", "Tac", "Src", "Tgt", "Reg", "Val", "Cap", "Pro", "Spa", "Semantic"];
      return (order.indexOf(a) === -1 ? 999 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 999 : order.indexOf(b));
    });
  const filteredNodes = (graph?.semantic_supernodes || [])
    .filter((node) => (node.semantic_type || "Semantic") === selectedTaxonomy)
    .sort((a, b) => {
      const aRow = typeof a.visual?.row === "number" ? a.visual.row : 999;
      const bRow = typeof b.visual?.row === "number" ? b.visual.row : 999;
      if (aRow !== bRow) return aRow - bRow;
      const aColumn = typeof a.visual?.column === "number" ? a.visual.column : 999;
      const bColumn = typeof b.visual?.column === "number" ? b.visual.column : 999;
      return aColumn - bColumn;
    });

  if (!graph?.semantic_supernodes.length) {
    return (
      <div className="rounded-lg border bg-white p-3 text-sm text-slate-500">
        No semantic supernodes are present in this graph.
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-white p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold">Supernodes with board positions</div>
          <div className="text-xs text-slate-500">Choose a taxonomy, then inspect position-specific supernodes.</div>
        </div>
        <select
          className="rounded border bg-white px-2 py-1 text-xs"
          value={selectedTaxonomy}
          onChange={(event) => onSelectTaxonomy(event.target.value)}
        >
          {taxonomyOptions.map((taxonomy) => (
            <option key={taxonomy} value={taxonomy}>
              {taxonomy}
            </option>
          ))}
        </select>
      </div>
      <div className="space-y-2">
        {filteredNodes.map((node) => {
          const primarySquare = node.position?.board_square || node.position?.board_squares?.[0] || "?";
          return (
            <button
              key={node.supernode_id}
              className={`flex w-full gap-3 rounded-lg border p-2 text-left transition hover:bg-slate-50 ${
                selectedId === node.supernode_id ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-white"
              }`}
              onClick={() => onSelectSupernode(node.supernode_id)}
            >
              <div className="shrink-0">
                <BoardSquareTile
                  square={primarySquare}
                  token={tokenBySquare.get(primarySquare)}
                  active={selectedId === node.supernode_id}
                />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600">
                    {node.role_id || "Det"}
                  </span>
                  <span className="text-xs text-slate-500">{node.member_feature_ids.length} features</span>
                </div>
                <div className="mt-1 text-sm font-semibold text-slate-800">{node.label}</div>
                {node.interpretation ? (
                  <div className="mt-1 line-clamp-2 text-xs text-slate-600">{node.interpretation}</div>
                ) : null}
                {node.position?.board_squares && node.position.board_squares.length > 1 ? (
                  <div className="mt-1 text-[11px] text-slate-400">
                    evidence squares: {node.position.board_squares.slice(1, 5).join(", ")}
                    {node.position.board_squares.length > 5 ? ` +${node.position.board_squares.length - 5} more` : ""}
                  </div>
                ) : null}
              </div>
            </button>
          );
        })}
        {!filteredNodes.length ? (
          <div className="rounded border bg-slate-50 p-3 text-sm text-slate-500">
            No {selectedTaxonomy} supernodes are present in this graph.
          </div>
        ) : null}
      </div>
    </div>
  );
};

const formatUnknown = (value: unknown) => {
  if (value == null || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  if (typeof value === "string") return value;
  return JSON.stringify(value);
};

const FeatureTable = ({
  features,
  selectedFeatureId,
  onSelectFeature,
}: {
  features: SemanticChildFeature[];
  selectedFeatureId: string | null;
  onSelectFeature: (feature: SemanticChildFeature) => void;
}) => (
  <div className="overflow-auto rounded-lg border">
    <table className="w-full text-left text-xs">
      <thead className="bg-slate-50 text-slate-600">
        <tr>
          <th className="px-2 py-2">Feature ID</th>
          <th className="px-2 py-2">Layer</th>
          <th className="px-2 py-2">Type</th>
          <th className="px-2 py-2">Position</th>
          <th className="px-2 py-2">Square</th>
          <th className="px-2 py-2">Activation</th>
          <th className="px-2 py-2">Influence</th>
          <th className="px-2 py-2">Interpretation</th>
        </tr>
      </thead>
      <tbody>
        {features.map((feature) => (
          <tr
            key={feature.feature_id}
            className={`border-t ${selectedFeatureId === feature.feature_id ? "bg-blue-50" : ""}`}
          >
            <td className="max-w-[180px] px-2 py-2">
              <button
                className="break-all text-left font-mono text-blue-700 underline-offset-2 hover:underline"
                onClick={() => onSelectFeature(feature)}
              >
                {feature.feature_id}
              </button>
            </td>
            <td className="px-2 py-2">{formatUnknown(feature.layer)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.feature_type)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.token_position)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.board_square)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.activation)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.metadata?.influence)}</td>
            <td className="min-w-[220px] px-2 py-2">{feature.interpretation || "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const buildFlowGraph = (
  graph: SemanticSupernodeGraph | null,
  selectedId: string | null,
): {
  nodes: ReactFlowNode<SemanticNodeData>[];
  edges: ReactFlowEdge[];
} => {
  if (!graph) return { nodes: [], edges: [] };

  const typeOrder = ["Det", "Src", "Mov", "Tac", "Tgt", "Val", "Cap", "Pro", "Spa", "Reg", "Semantic"];
  const laneByType = new Map(typeOrder.map((type, index) => [type, index]));
  const sortedNodes = [...graph.semantic_supernodes].sort((a, b) => {
    const aColumn = typeof a.visual?.column === "number" ? a.visual.column : null;
    const bColumn = typeof b.visual?.column === "number" ? b.visual.column : null;
    if (aColumn !== null || bColumn !== null) return (aColumn ?? 999) - (bColumn ?? 999);
    const aToken = a.position?.token_position ?? 999;
    const bToken = b.position?.token_position ?? 999;
    if (aToken !== bToken) return aToken - bToken;
    return (laneByType.get(a.semantic_type || "Semantic") ?? 10) - (laneByType.get(b.semantic_type || "Semantic") ?? 10);
  });

  const nodes = sortedNodes.map((supernode, index) => {
    const column = typeof supernode.visual?.column === "number"
      ? supernode.visual.column
      : Math.min(index, 5);
    const row = typeof supernode.visual?.row === "number"
      ? supernode.visual.row
      : laneByType.get(supernode.semantic_type || "Semantic") ?? (index % 5);
    const subcolumn = typeof supernode.visual?.subcolumn === "number" ? supernode.visual.subcolumn : 0;
    return {
      id: supernode.supernode_id,
      type: "semanticSupernode",
      position: {
        x: 90 + column * 410 + subcolumn * 245,
        y: 70 + row * 230,
      },
      data: {
        supernode,
        childCount: supernode.member_feature_ids.length,
        selected: selectedId === supernode.supernode_id,
      },
    };
  });

  const edges = graph.semantic_edges.map((edge) => {
    const weight = edge.weight ?? 0;
    return {
      id: edge.edge_id,
      source: edge.source,
      target: edge.target,
      label: edge.label || edge.relation_type || undefined,
      animated: Math.abs(weight) > 0.2,
      selectable: true,
      reconnectable: true,
      focusable: true,
      interactionWidth: 28,
      markerEnd: {
        type: MarkerType.ArrowClosed,
      },
      style: {
        stroke: weight < 0 ? "#dc2626" : "#16a34a",
        strokeWidth: Math.max(2, Math.min(6, Math.abs(weight) * 8 || 2.2)),
      },
      labelStyle: { fontSize: 13, fontWeight: 600, fill: "#334155" },
      labelBgStyle: { fill: "white", fillOpacity: 0.92 },
      labelBgPadding: [6, 3] as [number, number],
    };
  });

  return { nodes, edges };
};

export const SemanticSupernodeGraphPage = () => {
  const [graph, setGraph] = useState<SemanticSupernodeGraph | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [path, setPath] = useState(DEFAULT_SEMANTIC_GRAPH_PATH);
  const [jsonText, setJsonText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedChildFeatureId, setSelectedChildFeatureId] = useState<string | null>(null);
  const [loadedFeature, setLoadedFeature] = useState<Feature | null>(null);
  const [featureLoadError, setFeatureLoadError] = useState<string | null>(null);
  const [currentBoardActivationError, setCurrentBoardActivationError] = useState<string | null>(null);
  const [currentBoardActivation, setCurrentBoardActivation] = useState<CurrentBoardActivationState | null>(null);
  const [showAllPositions, setShowAllPositions] = useState(false);
  const [isFeatureLoading, setIsFeatureLoading] = useState(false);
  const [selectedTaxonomy, setSelectedTaxonomy] = useState("Det");
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  const selectedSupernode = useMemo(
    () => graph?.semantic_supernodes.find((node) => node.supernode_id === selectedId) || null,
    [graph, selectedId],
  );
  const featureById = useMemo(
    () => new Map((graph?.child_features || []).map((feature) => [feature.feature_id, feature])),
    [graph],
  );
  const selectedFeatures = useMemo(
    () =>
      selectedSupernode?.member_feature_ids
        .map((featureId) => featureById.get(featureId))
        .filter((feature): feature is SemanticChildFeature => Boolean(feature)) || [],
    [featureById, selectedSupernode],
  );
  const selectedFeatureEdges = useMemo(() => {
    if (!selectedSupernode || !graph?.child_feature_edges) return [];
    const memberIds = new Set(selectedSupernode.member_feature_ids);
    return graph.child_feature_edges.filter((edge) => memberIds.has(edge.source) && memberIds.has(edge.target));
  }, [graph, selectedSupernode]);
  const flowGraph = useMemo(() => buildFlowGraph(graph, selectedId), [graph, selectedId]);
  const [flowNodes, setFlowNodes, onFlowNodesChange] = useNodesState<ReactFlowNode<SemanticNodeData>>([]);
  const [flowEdges, setFlowEdges, onFlowEdgesChange] = useEdgesState<ReactFlowEdge>([]);
  const selectedSemanticEdge = useMemo(
    () => graph?.semantic_edges.find((edge) => edge.edge_id === selectedEdgeId) || null,
    [graph?.semantic_edges, selectedEdgeId],
  );
  const displayedFlowEdges = useMemo(
    () =>
      flowEdges.map((edge) =>
        edge.id === selectedEdgeId
          ? {
              ...edge,
              style: {
                ...(edge.style || {}),
                stroke: "#2563eb",
                strokeWidth: 6,
              },
              animated: true,
              zIndex: 20,
            }
          : edge,
      ),
    [flowEdges, selectedEdgeId],
  );

  useEffect(() => {
    setFlowNodes(flowGraph.nodes);
    setFlowEdges(flowGraph.edges);
  }, [flowGraph.edges, flowGraph.nodes, setFlowEdges, setFlowNodes]);

  const setLoadedGraph = useCallback((loaded: SemanticSupernodeGraph) => {
    setGraph(loaded);
    setSelectedId(loaded.semantic_supernodes[0]?.supernode_id || null);
    setSelectedTaxonomy(loaded.semantic_supernodes.find((node) => node.semantic_type === "Det")?.semantic_type || loaded.semantic_supernodes[0]?.semantic_type || "Semantic");
    setSelectedEdgeId(null);
    setSelectedChildFeatureId(null);
    setLoadedFeature(null);
    setFeatureLoadError(null);
    setCurrentBoardActivationError(null);
    setCurrentBoardActivation(null);
    setShowAllPositions(false);
    setError(null);
  }, []);

  const handleSelectFeature = useCallback(async (feature: SemanticChildFeature) => {
    setSelectedChildFeatureId(feature.feature_id);
    setLoadedFeature(null);
    setFeatureLoadError(null);
    setCurrentBoardActivationError(null);
    setCurrentBoardActivation(null);
    setShowAllPositions(false);

    if (!feature.dictionary_name || feature.feature_index == null) {
      setFeatureLoadError("This child feature does not include dictionary_name and feature_index, so top activation samples cannot be fetched.");
      return;
    }

    setIsFeatureLoading(true);
    try {
      const fetched = await fetchFeatureByDictionaryName(feature.dictionary_name, feature.feature_index);
      if (!fetched) {
        setFeatureLoadError("Backend returned no feature data.");
      } else {
        setLoadedFeature(fetched);
      }

      if (graph?.board?.fen && !feature.feature_type?.toLowerCase().includes("mlp")) {
        const response = await fetch(
          `${import.meta.env.VITE_BACKEND_URL || ""}/dictionaries/${encodeURIComponent(feature.dictionary_name)}/features/${feature.feature_index}/analyze_fen_all_positions`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ fen: graph.board.fen }),
          },
        );
        if (!response.ok) {
          throw new Error(`Current-board activation failed: ${await response.text()}`);
        }
        const data = await response.json();
        const preferredPosition = getFeaturePreferredPosition(feature);
        const positions = Array.isArray(data.positions) ? data.positions : [];
        const allActivations = Array(64).fill(0);
        let singleActivations: number[] | undefined;
        let singleZPatternIndices: number[][] | undefined;
        let singleZPatternValues: number[] | undefined;
        let singlePosition = preferredPosition ?? 0;

        for (const position of data.positions || []) {
          const pos = Number(position.position);
          const activations = Array.isArray(position.activations) ? position.activations : [];
          for (let idx = 0; idx < Math.min(64, activations.length); idx += 1) {
            const value = Number(activations[idx] || 0);
            if (Math.abs(value) > Math.abs(allActivations[idx])) {
              allActivations[idx] = value;
            }
          }

          if (preferredPosition != null && pos === preferredPosition) {
            singlePosition = pos;
            singleActivations = activations.slice(0, 64);
            const normalized = normalizeZPattern(position.z_pattern_indices, position.z_pattern_values);
            singleZPatternIndices = normalized.zPatternIndices;
            singleZPatternValues = normalized.zPatternValues;
          }
        }

        if (!singleActivations && positions.length > 0) {
          const fallback = positions.find((position: any) => {
            const activations = Array.isArray(position.activations) ? position.activations : [];
            return activations.some((value: number) => value !== 0);
          }) || positions[0];
          singlePosition = Number(fallback.position) || 0;
          singleActivations = Array.isArray(fallback.activations) ? fallback.activations.slice(0, 64) : undefined;
          const normalized = normalizeZPattern(fallback.z_pattern_indices, fallback.z_pattern_values);
          singleZPatternIndices = normalized.zPatternIndices;
          singleZPatternValues = normalized.zPatternValues;
        }

        if (feature.activation != null && preferredPosition != null && singleActivations && singleActivations.every((value) => value === 0)) {
          singleActivations[preferredPosition] = feature.activation;
          allActivations[preferredPosition] = feature.activation;
        }

        setCurrentBoardActivation({
          single: {
            activations: singleActivations,
            zPatternIndices: singleZPatternIndices,
            zPatternValues: singleZPatternValues,
            position: singlePosition,
          },
          all: {
            activations: allActivations,
          },
        });
      } else if (!graph?.board?.fen) {
        setCurrentBoardActivationError("No current-board FEN is provided in this semantic graph JSON.");
      } else {
        setCurrentBoardActivationError("Current-board activation endpoint currently supports LoRSA/Transcoder names, not MLP features.");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.startsWith("Current-board activation failed")) {
        setCurrentBoardActivationError(message);
      } else {
        setFeatureLoadError(message);
      }
    } finally {
      setIsFeatureLoading(false);
    }
  }, [graph?.board?.fen]);

  const loadExample = useCallback(async () => {
    setIsLoading(true);
    try {
      setLoadedGraph(await fetchSemanticSupernodeGraphExample());
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      try {
        setLoadedGraph(await fetchLocalLookingAheadGraph());
        setError(`Backend example fetch failed (${message}); loaded bundled looking-ahead graph instead.`);
      } catch (fallbackErr) {
        const fallbackMessage = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
        setError(`Backend example fetch failed (${message}); bundled fallback also failed (${fallbackMessage}).`);
      }
    } finally {
      setIsLoading(false);
    }
  }, [setLoadedGraph]);

  useEffect(() => {
    loadExample();
  }, [loadExample]);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setIsLoading(true);
    try {
      setLoadedGraph(await parseSemanticSupernodeGraphFile(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
      event.target.value = "";
    }
  };

  const handleLoadPath = async () => {
    if (!path.trim()) return;
    setIsLoading(true);
    try {
      setLoadedGraph(await fetchSemanticSupernodeGraphFromPath(path.trim()));
    } catch (err) {
      if (path.trim() === DEFAULT_SEMANTIC_GRAPH_PATH) {
        const message = err instanceof Error ? err.message : String(err);
        try {
          setLoadedGraph(await fetchLocalLookingAheadGraph());
          setError(`Backend path fetch failed (${message}); loaded bundled looking-ahead graph instead.`);
        } catch (fallbackErr) {
          const fallbackMessage = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
          setError(`Backend path fetch failed (${message}); bundled fallback also failed (${fallbackMessage}).`);
        }
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleNormalizeText = async () => {
    setIsLoading(true);
    try {
      const parsed = JSON.parse(jsonText);
      setLoadedGraph(await normalizeSemanticSupernodeGraph(parsed));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <AppNavbar />
      <div className="container mx-auto flex min-h-[calc(100vh-72px)] flex-col p-4">
      <div className="mb-3 flex flex-wrap items-center gap-3 rounded-lg border bg-white p-3 shadow-sm">
        <div>
          <h1 className="text-xl font-bold">Semantic Supernode Graph</h1>
          <p className="text-sm text-slate-500">
            Upload or load a JSON file, then click a semantic supernode to inspect its child features.
          </p>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <button
            className="rounded border px-3 py-1.5 text-sm hover:bg-slate-50 disabled:opacity-50"
            disabled={isLoading}
            onClick={loadExample}
          >
            Load example
          </button>
          <label className="cursor-pointer rounded border px-3 py-1.5 text-sm hover:bg-slate-50">
            Upload JSON
            <input className="hidden" type="file" accept=".json,application/json" onChange={handleUpload} />
          </label>
        </div>
      </div>

      <div className="mb-3 grid gap-3 lg:grid-cols-[1fr_1fr]">
        <div className="flex gap-2 rounded-lg border bg-white p-2">
          <input
            className="min-w-0 flex-1 rounded border px-2 py-1 text-sm"
            value={path}
            onChange={(event) => setPath(event.target.value)}
            placeholder="Path relative to repo root"
          />
          <button
            className="rounded bg-slate-900 px-3 py-1 text-sm text-white disabled:opacity-50"
            disabled={isLoading}
            onClick={handleLoadPath}
          >
            Load path
          </button>
        </div>
        <div className="flex gap-2 rounded-lg border bg-white p-2">
          <textarea
            className="h-9 min-w-0 flex-1 resize-none rounded border px-2 py-1 text-sm"
            value={jsonText}
            onChange={(event) => setJsonText(event.target.value)}
            placeholder="Paste raw semantic graph JSON here"
          />
          <button
            className="rounded bg-slate-900 px-3 py-1 text-sm text-white disabled:opacity-50"
            disabled={isLoading || !jsonText.trim()}
            onClick={handleNormalizeText}
          >
            Normalize
          </button>
        </div>
      </div>

      {error ? <div className="mb-3 rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700">{error}</div> : null}
      {graph?.warnings?.length ? (
        <div className="mb-3 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
          {graph.warnings.slice(0, 3).map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
          {graph.warnings.length > 3 ? <div>... {graph.warnings.length - 3} more warning(s)</div> : null}
        </div>
      ) : null}

      <div className="grid min-h-0 flex-1 gap-3 xl:grid-cols-[minmax(360px,430px)_minmax(0,1fr)]">
        <div className="min-h-0 space-y-3 overflow-auto">
          <ChessboardSummary graph={graph} selectedSupernode={selectedSupernode} />
          <TaxonomySupernodePanel
            graph={graph}
            selectedId={selectedId}
            selectedTaxonomy={selectedTaxonomy}
            onSelectSupernode={setSelectedId}
            onSelectTaxonomy={setSelectedTaxonomy}
          />
          <div className="rounded-lg border bg-white p-3 text-sm">
            <div className="font-semibold">Graph summary</div>
            <div className="mt-2 text-slate-600">
              <div>Supernodes: {graph?.semantic_supernodes.length || 0}</div>
              <div>Child features: {graph?.child_features.length || 0}</div>
              <div>Edges: {graph?.semantic_edges.length || 0}</div>
              {graph?.metadata?.title ? <div>Title: {String(graph.metadata.title)}</div> : null}
            </div>
          </div>
        </div>

        <div className="min-h-0 overflow-auto rounded-lg border bg-white p-3">
          {selectedSupernode ? (
            <>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    {selectedSupernode.role_id || selectedSupernode.semantic_type || "Semantic supernode"}
                  </div>
                  <h2 className="text-lg font-bold">{selectedSupernode.label}</h2>
                </div>
                <span
                  className="rounded-full px-2 py-1 text-xs font-semibold"
                  style={{
                    background: getTypeColor(selectedSupernode.semantic_type).bg,
                    color: getTypeColor(selectedSupernode.semantic_type).text,
                  }}
                >
                  {selectedSupernode.semantic_type || "Semantic"}
                </span>
              </div>

              {selectedSupernode.interpretation ? (
                <p className="mt-3 text-sm text-slate-700">{selectedSupernode.interpretation}</p>
              ) : null}

              <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                <div className="rounded bg-slate-50 p-2">
                  <div className="text-slate-500">Token position</div>
                  <div className="font-semibold">{formatUnknown(selectedSupernode.position?.token_position)}</div>
                </div>
                <div className="rounded bg-slate-50 p-2">
                  <div className="text-slate-500">Board square</div>
                  <div className="font-semibold">
                    {selectedSupernode.position?.board_squares?.join(", ") ||
                      formatUnknown(selectedSupernode.position?.board_square)}
                  </div>
                </div>
              </div>

              {selectedSemanticEdge ? (
                <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm">
                  <div className="mb-2 flex items-center justify-between">
                    <div className="font-semibold text-blue-900">Selected semantic edge</div>
                    <button className="text-xs text-blue-700 underline" onClick={() => setSelectedEdgeId(null)}>
                      clear
                    </button>
                  </div>
                  <div className="space-y-1 text-xs text-blue-950">
                    <div className="font-mono break-all">
                      {selectedSemanticEdge.source} → {selectedSemanticEdge.target}
                    </div>
                    <div>Label: {selectedSemanticEdge.label || "—"}</div>
                    <div>Relation: {selectedSemanticEdge.relation_type || "—"}</div>
                    <div>Weight: {formatUnknown(selectedSemanticEdge.weight)}</div>
                    {selectedSemanticEdge.member_edge_ids?.length ? (
                      <div className="break-all">
                        Member edges: {selectedSemanticEdge.member_edge_ids.slice(0, 8).join(", ")}
                        {selectedSemanticEdge.member_edge_ids.length > 8 ? ` +${selectedSemanticEdge.member_edge_ids.length - 8} more` : ""}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}

              <div className="mt-4">
                <div className="mb-2 text-sm font-semibold">Child features</div>
                <FeatureTable
                  features={selectedFeatures}
                  selectedFeatureId={selectedChildFeatureId}
                  onSelectFeature={handleSelectFeature}
                />
              </div>

              {selectedFeatureEdges.length > 0 ? (
                <div className="mt-4">
                  <div className="mb-2 text-sm font-semibold">Internal child-feature edges</div>
                  <div className="space-y-2">
                    {selectedFeatureEdges.map((edge) => (
                      <div key={edge.edge_id} className="rounded border bg-slate-50 p-2 text-xs">
                        <div className="font-mono">
                          {edge.source} → {edge.target}
                        </div>
                        <div className="mt-1 text-slate-600">
                          {edge.label || edge.relation_type || "feature edge"}
                          {edge.weight != null ? ` · weight ${edge.weight.toFixed(4)}` : ""}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {selectedSupernode.evidence && Object.keys(selectedSupernode.evidence).length > 0 ? (
                <div className="mt-4">
                  <div className="mb-2 text-sm font-semibold">Evidence</div>
                  <pre className="max-h-56 overflow-auto rounded bg-slate-950 p-3 text-xs text-slate-100">
                    {JSON.stringify(selectedSupernode.evidence, null, 2)}
                  </pre>
                </div>
              ) : null}

              <div className="mt-4 border-t pt-4">
                <div className="mb-2 text-sm font-semibold">Selected feature detail</div>
                {selectedChildFeatureId ? (
                  <div className="mb-3 rounded-lg border bg-white p-3">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <div className="text-sm font-semibold">Current-board activation</div>
                      {currentBoardActivation ? (
                        <button
                          onClick={() => setShowAllPositions((value) => !value)}
                          className={`rounded px-3 py-1 text-xs transition-colors ${
                            showAllPositions
                              ? "bg-blue-500 text-white hover:bg-blue-600"
                              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                          }`}
                          title={showAllPositions ? "Show activation of single selected position" : "Show merged activation of all positions"}
                        >
                          {showAllPositions ? "Single Position Mode" : "All Positions Mode"}
                        </button>
                      ) : null}
                    </div>
                    {currentBoardActivation ? (
                      <>
                        <div className="mb-2 text-center text-sm text-purple-600">
                          {showAllPositions ? (
                            <>
                              All positions merged activation:{" "}
                              {(currentBoardActivation.all?.activations || []).filter((value) => value !== 0).length} non-zero activations
                            </>
                          ) : (
                            <>
                              Single position {currentBoardActivation.single?.position ?? "—"} activation:{" "}
                              {(currentBoardActivation.single?.activations || []).filter((value) => value !== 0).length} non-zero activations
                              {currentBoardActivation.single?.zPatternValues?.length
                                ? `, ${currentBoardActivation.single.zPatternValues.length} Z pattern connections`
                                : ""}
                            </>
                          )}
                        </div>
                        <ChessBoard
                          fen={graph?.board?.fen || ""}
                          size="small"
                          showCoordinates
                          activations={
                            showAllPositions
                              ? currentBoardActivation.all?.activations
                              : currentBoardActivation.single?.activations
                          }
                          zPatternIndices={showAllPositions ? undefined : currentBoardActivation.single?.zPatternIndices}
                          zPatternValues={showAllPositions ? undefined : currentBoardActivation.single?.zPatternValues}
                          sampleIndex={showAllPositions ? undefined : currentBoardActivation.single?.position}
                          analysisName={showAllPositions ? "All Positions Feature Activation" : "Single Position Feature Activation"}
                          flip_activation={Boolean(graph?.board?.fen?.includes(" b "))}
                          autoFlipWhenBlack
                          disableAutoAnalyze
                        />
                      </>
                    ) : currentBoardActivationError ? (
                      <div className="rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                        {currentBoardActivationError}
                      </div>
                    ) : isFeatureLoading ? (
                      <div className="rounded border bg-slate-50 p-3 text-sm text-slate-600">
                        Loading current-board activation...
                      </div>
                    ) : (
                      <div className="rounded border bg-slate-50 p-3 text-sm text-slate-500">
                        Current-board activation is not loaded.
                      </div>
                    )}
                  </div>
                ) : null}
                {isFeatureLoading ? (
                  <div className="rounded border bg-slate-50 p-3 text-sm text-slate-600">
                    Loading feature top activation samples...
                  </div>
                ) : featureLoadError ? (
                  <div className="rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                    {featureLoadError}
                  </div>
                ) : loadedFeature ? (
                  <div className="max-h-[720px] overflow-auto rounded border bg-white p-3">
                    <div className="mb-2 text-sm font-semibold">
                      Top activation samples for {loadedFeature.dictionaryName} #{loadedFeature.featureIndex}
                    </div>
                    {loadedFeature.interpretation?.text ? (
                      <div className="mb-3 rounded border bg-slate-50 p-2 text-sm text-slate-700">
                        {loadedFeature.interpretation.text}
                      </div>
                    ) : null}
                    <ChessTopActivationBoards feature={loadedFeature} />
                  </div>
                ) : (
                  <div className="rounded border bg-slate-50 p-3 text-sm text-slate-500">
                    Click a child feature id above to load its interpretation and top activation samples.
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="text-sm text-slate-500">Select a semantic supernode to inspect features.</div>
          )}
        </div>
      </div>

      <div className="mt-3 h-[620px] min-h-[520px] overflow-hidden rounded-lg border bg-white shadow-sm">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <div>
            <div className="text-sm font-semibold">Semantic Supernode Graph</div>
            <div className="text-xs text-slate-500">
              Drag nodes directly; click edges to inspect labels; drag edge endpoints to adjust local routing.
            </div>
          </div>
          <div className="text-xs text-slate-500">
            {flowNodes.length} nodes · {flowEdges.length} edges
          </div>
        </div>
        <div className="h-[calc(100%-49px)] min-w-0">
          <ReactFlow
            nodes={flowNodes}
            edges={displayedFlowEdges}
            nodeTypes={nodeTypes}
            onNodesChange={onFlowNodesChange}
            onEdgesChange={onFlowEdgesChange}
            fitView
            fitViewOptions={{ padding: 0.18 }}
            onNodeClick={(_, node) => {
              setSelectedId(node.id);
              const semanticType = node.data?.supernode?.semantic_type || "Semantic";
              setSelectedTaxonomy(semanticType);
              setSelectedEdgeId(null);
            }}
            onEdgeClick={(_, edge) => setSelectedEdgeId(edge.id)}
            onReconnect={(oldEdge, newConnection) => {
              setFlowEdges((edges) => reconnectEdge(oldEdge, newConnection, edges));
            }}
            edgesReconnectable
            nodesDraggable
            panOnDrag
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>
      </div>
      </div>
    </div>
  );
};
