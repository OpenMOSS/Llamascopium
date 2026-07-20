import { memo, useMemo } from 'react'
import { Link, useMatches } from '@tanstack/react-router'
import { ArrowLeftRight, ChevronRight, Send, Target } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { Card } from '../ui/card'
import { Info } from '../ui/info'
import { Button } from '../ui/button'
import type { Node, QKTracingResults } from '@/types/circuit'
import type { RawEdgeIndex, RawNodeIndex } from '@/utils/circuit-index'
import { cn } from '@/lib/utils'
import { formatFeatureId } from '@/utils/circuit'
import { getWeightStyle } from '@/utils/style'
import { getEdgesBySource, getEdgesByTarget } from '@/utils/circuit-index'

interface FeatureRowProps {
  node: Node
  weight: number
  isHidden: boolean
  isHovered: boolean
  isClicked: boolean
  onNodeClick: (nodeId: string, isMultiSelect: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

const FeatureRow = memo(
  ({
    node,
    weight,
    isHidden,
    isHovered,
    isClicked,
    onNodeClick,
    onNodeHover,
  }: FeatureRowProps) => {
    const weightStyle = getWeightStyle(weight)

    return (
      <div
        className={cn(
          'py-2 px-2 mx-1 border rounded cursor-pointer transition-colors bg-gray-50 border-gray-200',
          isHidden && 'opacity-50',
          isHovered && 'ring-2 ring-blue-300',
          isClicked && 'ring-2 ring-blue-500',
        )}
        style={weightStyle}
        onClick={(e) => onNodeClick(node.nodeId, e.metaKey || e.ctrlKey)}
        onMouseEnter={() => onNodeHover(node.nodeId)}
        onMouseLeave={() => onNodeHover(null)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-gray-600">
              {formatFeatureId(node, false)}
            </span>
            <span className="text-xs font-medium">
              {(node.featureType === 'cross layer transcoder' ||
                node.featureType === 'lorsa' ||
                node.featureType === 'matryoshka sae') &&
                node.feature.interpretation?.text}
            </span>
          </div>
          <div className="text-right flex flex-col items-end">
            <div className="text-xs font-mono" title="Edge Weight">
              {weight > 0 ? '+' : ''}
              {weight.toFixed(3)}
            </div>
            {'activation' in node && (
              <div
                className="text-[10px] text-orange-500 font-mono"
                title="Node Activation"
              >
                {node.activation.toFixed(2)}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  },
)

FeatureRow.displayName = 'FeatureRow'

const QKTracingSection = memo(
  ({
    results,
    nodeIndex,
    onNodeClick,
    onNodeHover,
    hoveredId,
    clickedId,
  }: {
    results: QKTracingResults
    nodeIndex: RawNodeIndex
    onNodeClick: (nodeId: string, isMultiSelect: boolean) => void
    onNodeHover: (nodeId: string | null) => void
    hoveredId: string | null
    clickedId: string
  }) => {
    const renderNodeLabel = (nodeId: string) => {
      const node = nodeIndex.byId.get(nodeId)
      if (!node) return <span className="text-xs font-mono">{nodeId}</span>

      return (
        <div className="flex items-center space-x-2 min-w-0 flex-1">
          <span className="text-xs font-mono text-gray-600 shrink-0">
            {formatFeatureId(node, false)}
          </span>
          <span className="text-xs font-medium truncate">
            {node.featureType === 'lorsa' ||
            node.featureType === 'cross layer transcoder' ||
            node.featureType === 'matryoshka sae'
              ? node.feature.interpretation?.text
              : 'token' in node
                ? node.token
                : ''}
          </span>
        </div>
      )
    }

    return (
      <>
        <div className="flex gap-3 min-h-0">
          <div className="flex flex-col w-1/2 gap-1.5 min-h-0">
            <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
              <span>K MARGINAL</span>
            </div>
            <div className="flex flex-col gap-1 overflow-y-auto no-scrollbar max-h-40">
              {results.topKMarginalContributors.map(([nodeId, score], idx) => (
                <div
                  key={`k-${nodeId}-${idx}`}
                  className={cn(
                    'py-2 px-2 mx-1 border rounded cursor-pointer transition-colors bg-gray-50 border-gray-200 flex justify-between items-center',
                    nodeId === hoveredId && 'ring-2 ring-blue-300',
                    nodeId === clickedId && 'ring-2 ring-blue-500',
                  )}
                  style={getWeightStyle(score)}
                  onClick={(e) => onNodeClick(nodeId, e.metaKey || e.ctrlKey)}
                  onMouseEnter={() => onNodeHover(nodeId)}
                  onMouseLeave={() => onNodeHover(null)}
                >
                  {renderNodeLabel(nodeId)}
                  <div className="text-right flex flex-col items-end">
                    <div className="text-xs font-mono">{score.toFixed(3)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col w-1/2 gap-1.5 min-h-0">
            <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
              <span>Q MARGINAL</span>
            </div>
            <div className="flex flex-col gap-1 overflow-y-auto no-scrollbar max-h-40">
              {results.topQMarginalContributors.map(([nodeId, score], idx) => (
                <div
                  key={`q-${nodeId}-${idx}`}
                  className={cn(
                    'py-2 px-2 mx-1 border rounded cursor-pointer transition-colors bg-gray-50 border-gray-200 flex justify-between items-center',
                    nodeId === hoveredId && 'ring-2 ring-blue-300',
                    nodeId === clickedId && 'ring-2 ring-blue-500',
                  )}
                  style={getWeightStyle(score)}
                  onClick={(e) => onNodeClick(nodeId, e.metaKey || e.ctrlKey)}
                  onMouseEnter={() => onNodeHover(nodeId)}
                  onMouseLeave={() => onNodeHover(null)}
                >
                  {renderNodeLabel(nodeId)}
                  <div className="text-right flex flex-col items-end">
                    <div className="text-xs font-mono">{score.toFixed(3)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-1.5 min-h-0">
          <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
            <span>TOP PAIRWISE CONTRIBUTORS</span>
          </div>
          <div className="flex flex-col gap-1 max-h-40 overflow-y-auto no-scrollbar pr-1">
            {results.pairWiseContributors.map(([qId, kId, score], idx) => (
              <div
                key={`pw-${idx}`}
                className="mx-1 border rounded transition-colors bg-gray-50 border-gray-200 flex items-stretch"
                style={getWeightStyle(score)}
              >
                <div
                  className={cn(
                    'flex-1 min-w-0 cursor-pointer px-2 py-1.5 transition-colors flex items-center gap-1.5',
                    kId === hoveredId ? 'bg-black/5' : 'hover:bg-black/5',
                    kId === clickedId &&
                      'bg-blue-100/50 shadow-[inset_0_0_0_1px_#3b82f6]',
                  )}
                  onClick={(e) => onNodeClick(kId, e.metaKey || e.ctrlKey)}
                  onMouseEnter={() => onNodeHover(kId)}
                  onMouseLeave={() => onNodeHover(null)}
                >
                  <span className="text-xs font-bold text-slate-400 shrink-0">
                    K:
                  </span>
                  {renderNodeLabel(kId)}
                </div>

                <div className="flex items-center justify-center shrink-0">
                  <ChevronRight className="w-3 h-3 text-slate-400/50" />
                </div>

                <div
                  className={cn(
                    'flex-1 min-w-0 cursor-pointer px-2 py-1.5 transition-colors flex items-center gap-1.5',
                    qId === hoveredId ? 'bg-black/5' : 'hover:bg-black/5',
                    qId === clickedId &&
                      'bg-blue-100/50 shadow-[inset_0_0_0_1px_#3b82f6]',
                  )}
                  onClick={(e) => onNodeClick(qId, e.metaKey || e.ctrlKey)}
                  onMouseEnter={() => onNodeHover(qId)}
                  onMouseLeave={() => onNodeHover(null)}
                >
                  <span className="text-xs font-bold text-slate-400 shrink-0">
                    Q:
                  </span>
                  {renderNodeLabel(qId)}
                </div>

                <div className="text-right flex flex-col justify-center items-end px-2 shrink-0 border-l border-black/5 bg-white/30">
                  <div className="text-xs font-mono">{score.toFixed(3)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </>
    )
  },
)

QKTracingSection.displayName = 'QKTracingSection'

interface NodeConnectionsProps {
  nodeIndex: RawNodeIndex
  edgeIndex: RawEdgeIndex
  clickedId: string
  hoveredId: string | null
  hiddenIds: string[]
  className?: string
  onNodeClick: (nodeId: string, isMultiSelect: boolean) => void
  onNodeHover: (nodeId: string | null) => void
}

export const NodeConnections = memo(
  ({
    nodeIndex,
    edgeIndex,
    clickedId,
    hoveredId,
    hiddenIds,
    onNodeClick,
    onNodeHover,
    className,
  }: NodeConnectionsProps) => {
    const matches = useMatches()
    const isEmbed = matches.some(
      (match) => (match.staticData as { embed?: boolean })?.embed,
    )

    const clickedNode = useMemo(
      () => nodeIndex.byId.get(clickedId)!,
      [nodeIndex, clickedId],
    )

    const inputNodes = useMemo(() => {
      const incomingEdges = getEdgesByTarget(edgeIndex, clickedId)
      return incomingEdges
        .map((edge) => {
          const node = nodeIndex.byId.get(edge.source)
          if (!node) return null
          return { node, weight: edge.weight }
        })
        .filter((item): item is { node: Node; weight: number } => item !== null)
        .sort((a, b) => b.weight - a.weight)
    }, [edgeIndex, nodeIndex, clickedId])

    const outputNodes = useMemo(() => {
      const outgoingEdges = getEdgesBySource(edgeIndex, clickedId)
      return outgoingEdges
        .map((edge) => {
          const node = nodeIndex.byId.get(edge.target)
          if (!node) return null
          return { node, weight: edge.weight }
        })
        .filter((item): item is { node: Node; weight: number } => item !== null)
        .sort((a, b) => b.weight - a.weight)
    }, [edgeIndex, nodeIndex, clickedId])

    const hiddenIdsSet = useMemo(() => new Set(hiddenIds), [hiddenIds])

    const hasQKResults =
      clickedNode.featureType === 'lorsa' && !!clickedNode.qkTracingResults

    return (
      <Card
        className={cn('flex flex-col basis-1/2 min-w-0 gap-4 p-4', className)}
      >
        <div className="flex items-center justify-between">
          <div className="flex justify-between items-center space-x-2 w-full">
            <div className="flex items-center space-x-3 overflow-hidden">
              {/* <span className="text-xs font-mono text-gray-500 shrink-0">
                {formatFeatureId(clickedNode, true)}
              </span> */}
              <span className="text-sm font-semibold truncate">
                {clickedNode.featureType === 'cross layer transcoder' ||
                clickedNode.featureType === 'lorsa' ||
                clickedNode.featureType === 'matryoshka sae'
                  ? clickedNode.feature.interpretation?.text
                  : 'token' in clickedNode
                    ? clickedNode.token
                    : ''}
              </span>
              {'activation' in clickedNode && (
                <span className="text-xs font-mono text-orange-500 shrink-0">
                  ({clickedNode.activation.toFixed(3)})
                </span>
              )}
            </div>
            {(clickedNode.featureType === 'cross layer transcoder' ||
              clickedNode.featureType === 'lorsa' ||
              clickedNode.featureType === 'matryoshka sae') && (
              <Link
                to="/dictionaries/$dictionaryName/features/$featureIndex"
                params={{
                  dictionaryName: clickedNode.saeName,
                  featureIndex: clickedNode.feature.featureIndex.toString(),
                }}
                {...(isEmbed && {
                  target: '_blank',
                  rel: 'noopener noreferrer',
                })}
              >
                <Button size="sm" className="h-8 px-4 text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-mono text-gray-600">
                      {formatFeatureId(clickedNode, true)}
                    </span>
                    <Send className="w-3.5 h-3.5 text-gray-400" />
                  </div>
                </Button>
              </Link>
            )}
          </div>
        </div>

        <Tabs defaultValue="ov" className="flex flex-col flex-1 min-h-0">
          <TabsList className="w-full justify-start rounded-none border-b bg-transparent p-0 h-auto gap-6 px-1">
            <TabsTrigger
              value="ov"
              className="rounded-none border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:bg-transparent data-[state=active]:shadow-none px-2 py-2 text-xs font-semibold transition-all hover:text-blue-600"
            >
              <ArrowLeftRight className="h-3.5 w-3.5 mr-2" />
              OV Inputs/Outputs
            </TabsTrigger>
            <TabsTrigger
              value="qk"
              disabled={!hasQKResults}
              className="rounded-none border-b-2 border-transparent data-[state=active]:border-purple-500 data-[state=active]:bg-transparent data-[state=active]:shadow-none px-2 py-2 text-xs font-semibold transition-all hover:text-purple-600 disabled:opacity-30"
            >
              <Target className="h-3.5 w-3.5 mr-2" />
              QK Tracing
            </TabsTrigger>
          </TabsList>
          <TabsContent value="ov" className="flex-1 min-h-0 mt-4">
            <div className="flex gap-4 h-full">
              <div className="flex flex-col w-1/2 gap-2 min-h-0 h-full">
                <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
                  <span>INPUT NODES</span>
                  <Info iconSize={14}>
                    Nodes (features/embeddings) that influence the clicked node.
                  </Info>
                </div>
                <div className="flex flex-col gap-2 overflow-y-auto no-scrollbar flex-1">
                  {inputNodes.map((item) => (
                    <FeatureRow
                      key={item.node.nodeId}
                      node={item.node}
                      weight={item.weight}
                      isHidden={hiddenIdsSet.has(item.node.nodeId)}
                      isHovered={item.node.nodeId === hoveredId}
                      isClicked={item.node.nodeId === clickedId}
                      onNodeClick={onNodeClick}
                      onNodeHover={onNodeHover}
                    />
                  ))}
                </div>
              </div>
              <div className="flex flex-col w-1/2 gap-2 min-h-0 h-full">
                <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
                  <span>OUTPUT NODES</span>
                  <Info iconSize={14}>
                    Nodes (features/logits) that the clicked node influences.
                  </Info>
                </div>
                <div className="flex flex-col gap-2 overflow-y-auto no-scrollbar flex-1">
                  {outputNodes.map((item) => (
                    <FeatureRow
                      key={item.node.nodeId}
                      node={item.node}
                      weight={item.weight}
                      isHidden={hiddenIdsSet.has(item.node.nodeId)}
                      isHovered={item.node.nodeId === hoveredId}
                      isClicked={item.node.nodeId === clickedId}
                      onNodeClick={onNodeClick}
                      onNodeHover={onNodeHover}
                    />
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>
          {hasQKResults && clickedNode.qkTracingResults && (
            <TabsContent
              value="qk"
              className="flex-1 min-h-0 mt-2 overflow-y-auto"
            >
              <QKTracingSection
                results={clickedNode.qkTracingResults}
                nodeIndex={nodeIndex}
                onNodeClick={onNodeClick}
                onNodeHover={onNodeHover}
                hoveredId={hoveredId}
                clickedId={clickedId}
              />
            </TabsContent>
          )}
        </Tabs>
      </Card>
    )
  },
)

NodeConnections.displayName = 'NodeConnections'
