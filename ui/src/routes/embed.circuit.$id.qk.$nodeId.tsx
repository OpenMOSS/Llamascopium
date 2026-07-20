import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { AlertCircle, ChevronRight, Loader2 } from 'lucide-react'
import React, { useCallback, useMemo, useState } from 'react'
import { z } from 'zod'
import type { QKTracingResults } from '@/types/circuit'
import { parseWithPrettify } from '@/utils/zod'
import {
  circuitQKNodeQueryOptions,
  circuitStatusQueryOptions,
} from '@/api/circuits'
import { createRawNodeIndex } from '@/utils/circuit-index'
import { cn } from '@/lib/utils'
import { formatFeatureId } from '@/utils/circuit'
import { getWeightStyle } from '@/utils/style'

const searchParamsSchema = z.object({
  nodeThreshold: z.coerce.number().optional(),
  edgeThreshold: z.coerce.number().optional(),
  plain: z.boolean().optional().catch(false),
})

export const Route = createFileRoute('/embed/circuit/$id/qk/$nodeId')({
  validateSearch: (search) => parseWithPrettify(searchParamsSchema, search),
  staticData: {
    fullScreen: true,
    embed: true,
  },
  component: EmbedQKTracingPage,
})

function EmbedQKTracingPage() {
  const { id: circuitId, nodeId } = Route.useParams()
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })

  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const nodeThreshold = search.nodeThreshold ?? 0.6
  const edgeThreshold = search.edgeThreshold ?? 0.8
  const plain = search.plain

  const { data: statusData, isLoading: isLoadingStatus } = useQuery(
    circuitStatusQueryOptions(circuitId),
  )

  const {
    data: qkNodeData,
    isLoading: isLoadingQKNode,
    error: qkNodeError,
  } = useQuery({
    ...circuitQKNodeQueryOptions(
      circuitId,
      nodeId,
      nodeThreshold,
      edgeThreshold,
    ),
    enabled: statusData?.status === 'completed',
  })

  const targetNode = qkNodeData?.targetNode ?? null

  const rawNodeIndex = useMemo(
    () =>
      qkNodeData
        ? createRawNodeIndex([
            qkNodeData.targetNode,
            ...qkNodeData.referencedNodes,
          ])
        : null,
    [qkNodeData],
  )

  const qkTracingResults = useMemo(() => {
    if (
      !targetNode ||
      (targetNode.featureType !== 'lorsa' &&
        targetNode.featureType !== 'cross layer transcoder')
    ) {
      return null
    }
    return targetNode.qkTracingResults || null
  }, [targetNode])

  const handleNodeClick = useCallback(
    (nodeId: string, isMultiSelect: boolean) => {
      if (isMultiSelect) return
      // Navigate to the main circuit view with this node clicked
      navigate({
        to: '/embed/circuit/$id',
        params: { id: circuitId },
        search: { clickedId: nodeId },
      })
    },
    [circuitId, navigate],
  )

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredId(nodeId)
  }, [])

  const renderNodeLabel = useCallback(
    (nodeId: string) => {
      if (!rawNodeIndex)
        return <span className="text-xs font-mono">{nodeId}</span>
      const node = rawNodeIndex.byId.get(nodeId)
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
    },
    [rawNodeIndex],
  )

  return (
    <div
      className={cn(
        'min-h-screen p-4 flex flex-col justify-center items-center',
        !plain && 'h-screen overflow-hidden bg-slate-50/50',
      )}
    >
      {isLoadingStatus ? (
        <div className="flex-1 flex flex-col items-center justify-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      ) : statusData?.status === 'pending' ||
        statusData?.status === 'running' ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 h-full">
          <div className="flex flex-col items-center gap-4 w-full max-w-md">
            <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
            <div className="text-center space-y-2 w-full">
              <h3 className="text-base font-medium text-slate-900">
                Generating Circuit
              </h3>
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-medium text-slate-700">
                  <span>{statusData?.progressPhase || 'Initializing...'}</span>
                  <span>{Math.round(statusData?.progress ?? 0)}%</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-600 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${statusData?.progress ?? 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : statusData?.status === 'failed' ? (
        <div className="flex items-center justify-center h-full p-8">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Circuit Generation Failed
            </h3>
            <p className="text-gray-600 max-w-md">
              {statusData?.errorMessage || 'An unknown error occurred'}
            </p>
          </div>
        </div>
      ) : qkNodeError ? (
        <div className="flex items-center justify-center h-full p-8">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Failed to load QK tracing data
            </h3>
            <p className="text-gray-600">
              {qkNodeError.message || 'Failed to load QK tracing data'}
            </p>
          </div>
        </div>
      ) : isLoadingQKNode ? (
        <div className="flex-1 flex flex-col items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      ) : !targetNode ? (
        <div className="flex items-center justify-center h-full p-8">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Node Not Found
            </h3>
            <p className="text-gray-600">
              Node with ID "{nodeId}" was not found in this circuit.
            </p>
          </div>
        </div>
      ) : !qkTracingResults ? (
        <div className="flex items-center justify-center h-full p-8">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-yellow-600 mb-2">
              No QK Tracing Results
            </h3>
            <p className="text-gray-600">
              This node does not have QK tracing results available.
            </p>
          </div>
        </div>
      ) : !rawNodeIndex ? (
        <div className="flex items-center justify-center h-full p-8">
          <div className="text-center">
            <Loader2 className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
            <p className="text-gray-600">Loading node index...</p>
          </div>
        </div>
      ) : (
        <div className={cn('w-full', !plain && 'max-w-4xl')}>
          <div
            className={cn(
              'flex flex-col gap-4',
              !plain &&
                'rounded-xl border bg-card text-card-foreground shadow p-6',
            )}
          >
            {/* QK Tracing Content */}
            <QKTracingContent
              results={qkTracingResults}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
              hoveredId={hoveredId}
              clickedId={nodeId}
              renderNodeLabel={renderNodeLabel}
            />
          </div>
        </div>
      )}
    </div>
  )
}

interface QKTracingContentProps {
  results: QKTracingResults
  onNodeClick: (nodeId: string, isMultiSelect: boolean) => void
  onNodeHover: (nodeId: string | null) => void
  hoveredId: string | null
  clickedId: string
  renderNodeLabel: (nodeId: string) => React.ReactElement
}

function QKTracingContent({
  results,
  onNodeClick,
  onNodeHover,
  hoveredId,
  clickedId,
  renderNodeLabel,
}: QKTracingContentProps) {
  return (
    <div className="flex flex-col gap-6">
      {/* K and Q Marginal Contributors */}
      <div className="flex gap-4 min-h-0">
        <div className="flex flex-col w-1/2 gap-2 min-h-0">
          <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
            <span>K MARGINAL</span>
          </div>
          <div className="flex flex-col gap-1.5 overflow-y-auto no-scrollbar max-h-64">
            {results.topKMarginalContributors.map(([nodeId, score], idx) => (
              <div
                key={`k-${nodeId}-${idx}`}
                className={cn(
                  'py-2 px-3 border rounded cursor-pointer transition-colors bg-gray-50 border-gray-200 flex justify-between items-center',
                  nodeId === hoveredId && 'ring-2 ring-blue-300',
                  nodeId === clickedId && 'ring-2 ring-blue-500',
                )}
                style={getWeightStyle(score)}
                onMouseEnter={() => onNodeHover(nodeId)}
                onMouseLeave={() => onNodeHover(null)}
              >
                {renderNodeLabel(nodeId)}
                <div className="text-right flex flex-col items-end shrink-0 ml-2">
                  <div className="text-xs font-mono">{score.toFixed(3)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col w-1/2 gap-2 min-h-0">
          <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
            <span>Q MARGINAL</span>
          </div>
          <div className="flex flex-col gap-1.5 overflow-y-auto no-scrollbar max-h-64">
            {results.topQMarginalContributors.map(([nodeId, score], idx) => (
              <div
                key={`q-${nodeId}-${idx}`}
                className={cn(
                  'py-2 px-3 border rounded cursor-pointer transition-colors bg-gray-50 border-gray-200 flex justify-between items-center',
                  nodeId === hoveredId && 'ring-2 ring-blue-300',
                  nodeId === clickedId && 'ring-2 ring-blue-500',
                )}
                style={getWeightStyle(score)}
                onMouseEnter={() => onNodeHover(nodeId)}
                onMouseLeave={() => onNodeHover(null)}
              >
                {renderNodeLabel(nodeId)}
                <div className="text-right flex flex-col items-end shrink-0 ml-2">
                  <div className="text-xs font-mono">{score.toFixed(3)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Pairwise Contributors */}
      <div className="flex flex-col gap-2 min-h-0">
        <div className="font-semibold tracking-tight flex items-center text-sm text-slate-700 gap-1 cursor-default shrink-0">
          <span>TOP PAIRWISE CONTRIBUTORS</span>
        </div>
        <div className="flex flex-col gap-1.5 max-h-64 overflow-y-auto no-scrollbar pr-1">
          {results.pairWiseContributors.map(([qId, kId, score], idx) => (
            <div
              key={`pw-${idx}`}
              className="mx-0 border rounded transition-colors bg-gray-50 border-gray-200 flex items-stretch"
              style={getWeightStyle(score)}
            >
              <div
                className={cn(
                  'flex-1 min-w-0 cursor-pointer px-3 py-2 transition-colors flex items-center gap-2',
                  kId === hoveredId ? 'bg-black/5' : 'hover:bg-black/5',
                  kId === clickedId &&
                    'bg-blue-100/50 shadow-[inset_0_0_0_1px_#3b82f6]',
                )}
                onMouseEnter={() => onNodeHover(kId)}
                onMouseLeave={() => onNodeHover(null)}
              >
                <span className="text-xs font-bold text-slate-400 shrink-0">
                  K:
                </span>
                {renderNodeLabel(kId)}
              </div>

              <div className="flex items-center justify-center shrink-0 px-2">
                <ChevronRight className="w-3.5 h-3.5 text-slate-400/50" />
              </div>

              <div
                className={cn(
                  'flex-1 min-w-0 cursor-pointer px-3 py-2 transition-colors flex items-center gap-2',
                  qId === hoveredId ? 'bg-black/5' : 'hover:bg-black/5',
                  qId === clickedId &&
                    'bg-blue-100/50 shadow-[inset_0_0_0_1px_#3b82f6]',
                )}
                onMouseEnter={() => onNodeHover(qId)}
                onMouseLeave={() => onNodeHover(null)}
              >
                <span className="text-xs font-bold text-slate-400 shrink-0">
                  Q:
                </span>
                {renderNodeLabel(qId)}
              </div>

              <div className="text-right flex flex-col justify-center items-end px-3 shrink-0 border-l border-black/5 bg-white/30">
                <div className="text-xs font-mono">{score.toFixed(3)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
