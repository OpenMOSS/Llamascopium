import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { AlertCircle, Loader2 } from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'
import { z } from 'zod'
import type { CircuitData, VisState } from '@/types/circuit'
import { parseWithPrettify } from '@/utils/zod'
import { circuitQueryOptions, circuitStatusQueryOptions } from '@/api/circuits'
import { LinkGraphContainer } from '@/components/circuits/link-graph-container'
import { NodeConnections } from '@/components/circuits/node-connections'
import { FeatureCardHorizontal } from '@/components/feature/feature-card-horizontal'
import { Card } from '@/components/ui/card'
import { createRawEdgeIndex, createRawNodeIndex } from '@/utils/circuit-index'

const searchParamsSchema = z.object({
  clickedId: z.string().optional(),
  hiddenIds: z.string().optional(),
  nodeThreshold: z.coerce.number().optional(),
  edgeThreshold: z.coerce.number().optional(),
})

export const Route = createFileRoute('/embed/circuit/$id/')({
  validateSearch: (search) => parseWithPrettify(searchParamsSchema, search),
  staticData: {
    fullScreen: true,
    embed: true,
  },
  component: EmbedCircuitPage,
})

function EmbedCircuitPage() {
  const { id: circuitId } = Route.useParams()
  const search = Route.useSearch()
  const navigate = useNavigate({ from: Route.fullPath })

  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const nodeThreshold = search.nodeThreshold ?? 0.6
  const edgeThreshold = search.edgeThreshold ?? 0.8

  const { data: statusData, isLoading: isLoadingStatus } = useQuery(
    circuitStatusQueryOptions(circuitId),
  )

  const {
    data: circuitData,
    isLoading: isLoadingCircuit,
    error: circuitError,
  } = useQuery({
    ...circuitQueryOptions(circuitId, nodeThreshold, edgeThreshold),
    enabled: statusData?.status === 'completed',
  })

  const clickedId = useMemo(
    () =>
      circuitData?.graphData.nodes.find((n) => n.nodeId === search.clickedId)
        ?.nodeId || null,
    [search.clickedId, circuitData],
  )
  const hiddenIds = useMemo(
    () =>
      search.hiddenIds
        ? search.hiddenIds
            .split(',')
            .filter((id: string) =>
              circuitData?.graphData.nodes.find((n) => n.nodeId === id),
            )
        : [],
    [search.hiddenIds, circuitData?.graphData.nodes],
  )

  const circuit: CircuitData | undefined = circuitData?.graphData

  const rawNodeIndex = useMemo(
    () => (circuit ? createRawNodeIndex(circuit.nodes) : null),
    [circuit],
  )

  const rawEdgeIndex = useMemo(
    () => (circuit ? createRawEdgeIndex(circuit.edges) : null),
    [circuit],
  )

  const visState: VisState = useMemo(
    () => ({
      clickedId,
      hoveredId,
      selectedIds: [],
    }),
    [clickedId, hoveredId],
  )

  const featureData = useMemo(() => {
    if (!clickedId || !circuit) return null
    const node = circuit.nodes.find((n) => n.nodeId === clickedId)
    if (
      !node ||
      (node.featureType !== 'cross layer transcoder' &&
        node.featureType !== 'lorsa' &&
        node.featureType !== 'matryoshka sae')
    )
      return null
    return node.feature
  }, [clickedId, circuit])

  const handleNodeClick = useCallback(
    (nodeId: string, isMultiSelect: boolean) => {
      if (isMultiSelect) return
      const newClickedId = clickedId === nodeId ? null : nodeId
      navigate({
        search: (prev) => ({
          ...prev,
          clickedId: newClickedId ?? undefined,
        }),
        replace: true,
      })
    },
    [clickedId, navigate],
  )

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredId(nodeId)
  }, [])

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-slate-50/50 justify-center items-center">
      {isLoadingStatus ? (
        <div className="flex-1 flex flex-col items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      ) : statusData?.status === 'pending' ||
        statusData?.status === 'running' ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8">
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
      ) : circuitError ? (
        <div className="flex items-center justify-center h-full p-8">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Failed to load circuit
            </h3>
            <p className="text-gray-600">
              {circuitError.message || 'Failed to load circuit'}
            </p>
          </div>
        </div>
      ) : isLoadingCircuit ? (
        <div className="flex-1 flex flex-col items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      ) : circuit ? (
        <div className="flex-1 flex flex-col overflow-hidden p-4 max-w-[1800px] w-full h-full">
          <div className="flex gap-4 overflow-hidden h-full">
            <div className="flex flex-col gap-4 min-w-3/5 flex-1 grow shrink-0">
              <LinkGraphContainer
                data={circuit}
                visState={visState}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
              />
              {featureData && (
                <FeatureCardHorizontal
                  className="grow"
                  feature={featureData}
                  hidePlots={true}
                />
              )}
            </div>

            <div className="flex flex-col flex-1 gap-4 min-w-0 h-full overflow-hidden max-w-[500px]">
              {clickedId && rawNodeIndex && rawEdgeIndex && (
                <NodeConnections
                  nodeIndex={rawNodeIndex}
                  edgeIndex={rawEdgeIndex}
                  clickedId={clickedId}
                  hoveredId={hoveredId}
                  hiddenIds={hiddenIds}
                  onNodeClick={handleNodeClick}
                  onNodeHover={handleNodeHover}
                  className="flex-1 min-h-0"
                />
              )}
              {!clickedId && (
                <Card className="flex flex-col h-full gap-4 p-6 items-center justify-center text-slate-500 text-center">
                  <div className="flex flex-col items-center justify-center">
                    <span className="text-base">No node selected</span>
                    <span className="text-xs text-slate-500">
                      Click a node on the left to see its connections
                    </span>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
