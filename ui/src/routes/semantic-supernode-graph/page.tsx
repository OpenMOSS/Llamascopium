import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Edge as ReactFlowEdge,
  MarkerType,
  Node as ReactFlowNode,
  NodeProps,
  ReactFlow,
} from "@xyflow/react";
import {
  fetchSemanticSupernodeGraphExample,
  fetchSemanticSupernodeGraphFromPath,
  normalizeSemanticSupernodeGraph,
  parseSemanticSupernodeGraphFile,
} from "@/utils/api";
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
const boardRanks = ["8", "7", "6", "5", "4", "3", "2", "1"];

const ChessboardSummary = ({
  graph,
  selectedSupernode,
}: {
  graph: SemanticSupernodeGraph | null;
  selectedSupernode: SemanticSupernode | null;
}) => {
  const selectedSquares = new Set(
    selectedSupernode?.position?.board_squares?.length
      ? selectedSupernode.position.board_squares
      : selectedSupernode?.position?.board_square
        ? [selectedSupernode.position.board_square]
        : [],
  );
  const allSquares = new Set(
    graph?.semantic_supernodes.flatMap((node) => node.position?.board_squares || []) || [],
  );

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
      <div className="grid aspect-square max-w-[260px] grid-cols-8 overflow-hidden rounded border">
        {boardRanks.flatMap((rank, rankIndex) =>
          boardFiles.map((file, fileIndex) => {
            const square = `${file}${rank}`;
            const isDark = (rankIndex + fileIndex) % 2 === 1;
            const isSelected = selectedSquares.has(square);
            const isUsed = allSquares.has(square);
            return (
              <div
                key={square}
                className="relative flex items-center justify-center text-[10px]"
                style={{
                  background: isSelected
                    ? "#86efac"
                    : isUsed
                      ? "#bfdbfe"
                      : isDark
                        ? "#cbd5e1"
                        : "#f8fafc",
                }}
              >
                <span className="absolute left-1 top-0.5 text-[9px] text-slate-500">{square}</span>
                {isSelected ? <span className="h-3 w-3 rounded-full bg-green-600" /> : null}
              </div>
            );
          }),
        )}
      </div>
      {graph?.board?.fen ? (
        <div className="mt-2 break-all text-xs text-slate-500">FEN: {graph.board.fen}</div>
      ) : null}
    </div>
  );
};

const formatUnknown = (value: unknown) => {
  if (value == null || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  if (typeof value === "string") return value;
  return JSON.stringify(value);
};

const FeatureTable = ({ features }: { features: SemanticChildFeature[] }) => (
  <div className="overflow-auto rounded-lg border">
    <table className="w-full text-left text-xs">
      <thead className="bg-slate-50 text-slate-600">
        <tr>
          <th className="px-2 py-2">Feature</th>
          <th className="px-2 py-2">Layer</th>
          <th className="px-2 py-2">Type</th>
          <th className="px-2 py-2">Token</th>
          <th className="px-2 py-2">Square</th>
          <th className="px-2 py-2">Interpretation</th>
        </tr>
      </thead>
      <tbody>
        {features.map((feature) => (
          <tr key={feature.feature_id} className="border-t">
            <td className="max-w-[180px] px-2 py-2 font-mono">{feature.feature_id}</td>
            <td className="px-2 py-2">{formatUnknown(feature.layer)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.feature_type)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.token_position)}</td>
            <td className="px-2 py-2">{formatUnknown(feature.board_square)}</td>
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
    const aToken = a.position?.token_position ?? 999;
    const bToken = b.position?.token_position ?? 999;
    if (aToken !== bToken) return aToken - bToken;
    return (laneByType.get(a.semantic_type || "Semantic") ?? 10) - (laneByType.get(b.semantic_type || "Semantic") ?? 10);
  });

  const nodes = sortedNodes.map((supernode, index) => {
    const token = supernode.position?.token_position ?? index;
    const lane = laneByType.get(supernode.semantic_type || "Semantic") ?? (index % 5);
    return {
      id: supernode.supernode_id,
      type: "semanticSupernode",
      position: {
        x: 80 + token * 130,
        y: 50 + lane * 125,
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
      markerEnd: {
        type: MarkerType.ArrowClosed,
      },
      style: {
        stroke: weight < 0 ? "#dc2626" : "#16a34a",
        strokeWidth: Math.max(1.5, Math.min(5, Math.abs(weight) * 8 || 1.8)),
      },
      labelStyle: { fontSize: 11, fill: "#334155" },
      labelBgStyle: { fill: "white", fillOpacity: 0.85 },
    };
  });

  return { nodes, edges };
};

export const SemanticSupernodeGraphPage = () => {
  const [graph, setGraph] = useState<SemanticSupernodeGraph | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [path, setPath] = useState("scripts/Attribution_Graph/semantic_supernode_examples/looking_ahead_semantic_graph.json");
  const [jsonText, setJsonText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

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

  const setLoadedGraph = useCallback((loaded: SemanticSupernodeGraph) => {
    setGraph(loaded);
    setSelectedId(loaded.semantic_supernodes[0]?.supernode_id || null);
    setError(null);
  }, []);

  const loadExample = useCallback(async () => {
    setIsLoading(true);
    try {
      setLoadedGraph(await fetchSemanticSupernodeGraphExample());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
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
      setError(err instanceof Error ? err.message : String(err));
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
    <div className="flex h-screen flex-col bg-slate-100 p-4">
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

      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[280px_minmax(0,1fr)_430px]">
        <div className="min-h-0 space-y-3 overflow-auto">
          <ChessboardSummary graph={graph} selectedSupernode={selectedSupernode} />
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

        <div className="min-h-0 overflow-hidden rounded-lg border bg-white">
          <ReactFlow
            nodes={flowGraph.nodes}
            edges={flowGraph.edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            onNodeClick={(_, node) => setSelectedId(node.id)}
          >
            <Background />
            <Controls />
          </ReactFlow>
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

              <div className="mt-4">
                <div className="mb-2 text-sm font-semibold">Child features</div>
                <FeatureTable features={selectedFeatures} />
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
            </>
          ) : (
            <div className="text-sm text-slate-500">Select a semantic supernode to inspect features.</div>
          )}
        </div>
      </div>
    </div>
  );
};
