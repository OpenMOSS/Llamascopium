import {
  createFileRoute,
  redirect,
  useNavigate,
  useRouter,
} from '@tanstack/react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, GitFork, Loader2 } from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'
import { z } from 'zod'
import type { CircuitData, FeatureNode, VisState } from '@/types/circuit'
import type { MatryoshkaFeatureRange } from '@/api/circuits'
import { parseWithPrettify } from '@/utils/zod'
import {
  circuitQueryOptions,
  circuitStatusQueryOptions,
  circuitsQueryOptions,
  fetchMatryoshkaRanges,
  generateCircuit,
  saeSetsQueryOptions,
} from '@/api/circuits'
import { GraphSelector } from '@/components/circuits/graph-selector'
import { NewGraphDialog } from '@/components/circuits/new-graph-dialog'
import { LinkGraphContainer } from '@/components/circuits/link-graph-container'
import { SelectedFeaturesList } from '@/components/circuits/selected-features-list'
import { NodeConnections } from '@/components/circuits/node-connections'
import { ThresholdControls } from '@/components/circuits/threshold-controls'
import { MatryoshkaSubgraphControl } from '@/components/circuits/matryoshka-subgraph-control'
import { FeatureCardHorizontal } from '@/components/feature/feature-card-horizontal'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { createRawEdgeIndex, createRawNodeIndex } from '@/utils/circuit-index'
import { filterCircuitByMatryoshkaRange } from '@/utils/circuit'

const searchParamsSchema = z.object({
  clickedId: z.string().optional(),
  hiddenIds: z.string().optional(),
  selectedIds: z.string().optional(),
  nodeThreshold: z.coerce.number().optional(),
  edgeThreshold: z.coerce.number().optional(),
  matryoshkaSubrange: z.string().optional(),
})

export const Route = createFileRoute('/circuit/$id/')({
  validateSearch: (search) => parseWithPrettify(searchParamsSchema, search),
  staticData: {
    fullScreen: true,
  },
  component: CircuitPage,
  loader: async ({ context, params }) => {
    const [circuits, saeSets] = await Promise.all([
      context.queryClient.ensureQueryData(circuitsQueryOptions()),
      context.queryClient.ensureQueryData(saeSetsQueryOptions()),
    ])
    if (!circuits.find((c) => c.id === params.id)) {
      if (circuits.length > 0) {
        throw redirect({
          to: '/circuit/$id',
          params: { id: circuits[0].id },
        })
      }
      throw redirect({
        to: '/circuits',
      })
    }
    return { circuits, saeSets, circuitId: params.id }
  },
  gcTime: 0,
})

function CircuitPage() {
  const navigate = useNavigate({ from: Route.fullPath })
  const { circuits: loaderCircuits, saeSets, circuitId } = Route.useLoaderData()
  const search = Route.useSearch()
  const router = useRouter()

  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const nodeThreshold = search.nodeThreshold ?? 0.6
  const edgeThreshold = search.edgeThreshold ?? 0.8

  const { data: statusData, isLoading: isLoadingStatus } = useQuery(
    circuitStatusQueryOptions(circuitId),
  )
  const { data: queryCircuits } = useQuery(circuitsQueryOptions())
  const circuits = queryCircuits ?? loaderCircuits

  // Only fetch circuit data when it's ready
  const {
    data: circuitData,
    isLoading: isLoadingCircuit,
    isFetching: isFetchingCircuit,
    error: circuitError,
  } = useQuery({
    ...circuitQueryOptions(circuitId, nodeThreshold, edgeThreshold),
    enabled: statusData?.status === 'completed',
  })

  const tracedMatryoshkaRange =
    circuitData?.config.matryoshkaFeatureRange ?? null
  const { data: matryoshkaOptions } = useQuery({
    queryKey: ['matryoshka-ranges', circuitData?.saeSetName],
    queryFn: () =>
      fetchMatryoshkaRanges({
        data: { saeSetName: circuitData!.saeSetName },
      }),
    enabled: Boolean(circuitData?.saeSetName && tracedMatryoshkaRange),
  })

  const availableMatryoshkaSubranges = useMemo(() => {
    if (!tracedMatryoshkaRange || !matryoshkaOptions || !circuitData) return []
    const [tracedStart, tracedEnd] = tracedMatryoshkaRange
    return matryoshkaOptions.atomicSegments.filter(
      ([start, end]) =>
        start >= tracedStart &&
        end <= tracedEnd &&
        circuitData.graphData.nodes.some(
          (node) =>
            node.featureType === 'matryoshka sae' &&
            node.feature.featureIndex >= start &&
            node.feature.featureIndex < end,
        ),
    )
  }, [circuitData, matryoshkaOptions, tracedMatryoshkaRange])

  const selectedMatryoshkaSubrange = useMemo(() => {
    if (!search.matryoshkaSubrange) return null
    const values = search.matryoshkaSubrange.split(':').map(Number)
    if (values.length !== 2 || values.some((value) => !Number.isFinite(value)))
      return null
    const range: MatryoshkaFeatureRange = [values[0], values[1]]
    return availableMatryoshkaSubranges.some(
      ([start, end]) => start === range[0] && end === range[1],
    )
      ? range
      : null
  }, [availableMatryoshkaSubranges, search.matryoshkaSubrange])

  const circuit: CircuitData | undefined = useMemo(() => {
    const rawCircuit = circuitData?.graphData
    if (!rawCircuit || !selectedMatryoshkaSubrange) return rawCircuit
    return filterCircuitByMatryoshkaRange(
      rawCircuit,
      selectedMatryoshkaSubrange,
    )
  }, [circuitData?.graphData, selectedMatryoshkaSubrange])

  // Get clickedId, hiddenIds, and selectedIds and filter them to only include nodes that exist in the graph data
  const clickedId = useMemo(
    () =>
      circuit?.nodes.find((n) => n.nodeId === search.clickedId)?.nodeId || null,
    [search.clickedId, circuit],
  )
  const hiddenIds = useMemo(
    () =>
      search.hiddenIds
        ? search.hiddenIds
            .split(',')
            .filter((id) => circuit?.nodes.find((n) => n.nodeId === id))
        : [],
    [search.hiddenIds, circuit],
  )
  const selectedIds = useMemo(
    () =>
      search.selectedIds
        ? search.selectedIds
            .split(',')
            .filter((id) => circuit?.nodes.find((n) => n.nodeId === id))
        : [],
    [search.selectedIds, circuit],
  )

  const queryClient = useQueryClient()

  const { mutate: doGenerateCircuit, isPending: isGenerating } = useMutation({
    mutationFn: generateCircuit,
    onSuccess: (data) => {
      handleGraphCreated(data.circuitId)
    },
  })

  const handleThresholdsChange = useCallback(
    (nodeThreshold: number, edgeThreshold: number) => {
      navigate({
        search: (prev) => ({
          ...prev,
          nodeThreshold,
          edgeThreshold,
        }),
        replace: true,
      })
    },
    [navigate],
  )

  const handleClearSelected = useCallback(() => {
    navigate({
      search: (prev) => ({
        ...prev,
        selectedIds: undefined,
      }),
      replace: true,
    })
  }, [navigate])

  const handleRemoveSelected = useCallback(
    (idToRemove: string) => {
      navigate({
        search: (prev) => ({
          ...prev,
          selectedIds: selectedIds.filter((id) => id !== idToRemove).join(','),
        }),
        replace: true,
      })
    },
    [selectedIds, navigate],
  )

  const handleTraceSelected = useCallback(() => {
    if (!circuitData || selectedIds.length === 0) return

    const circuit = circuitData.graphData
    const listOfFeatures = selectedIds.map((id) => {
      const node = circuit.nodes.find((n) => n.nodeId === id) as FeatureNode
      return {
        saeName: node.saeName,
        featureIndex: node.feature.featureIndex,
        position: node.ctxIdx,
      }
    })

    doGenerateCircuit({
      data: {
        saeSetName: circuitData.saeSetName,
        input: circuitData.input,
        desiredLogitProb: circuitData.config.desiredLogitProb,
        maxFeatureNodes: circuitData.config.maxFeatureNodes,
        maxNLogits: circuitData.config.maxNLogits,
        qkTracingTopk: circuitData.config.qkTracingTopk,
        matryoshkaFeatureRange:
          circuitData.config.matryoshkaFeatureRange ?? null,
        listOfFeatures,
        parentId: circuitId,
      },
    })
  }, [circuitData, selectedIds, doGenerateCircuit, circuitId])

  const rawNodeIndex = useMemo(
    () => (circuit ? createRawNodeIndex(circuit.nodes) : null),
    [circuit],
  )

  const rawEdgeIndex = useMemo(
    () => (circuit ? createRawEdgeIndex(circuit.edges) : null),
    [circuit],
  )

  const handleGraphCreated = (newCircuitId: string) => {
    queryClient.invalidateQueries({ queryKey: ['circuits'] })
    router.invalidate()

    router.navigate({
      to: '/circuit/$id',
      params: { id: newCircuitId },
    })
    setHoveredId(null)
  }

  const handleMatryoshkaSubrangeChange = useCallback(
    (range: MatryoshkaFeatureRange | null) => {
      navigate({
        search: (prev) => ({
          ...prev,
          clickedId: undefined,
          selectedIds: undefined,
          matryoshkaSubrange: range?.join(':'),
        }),
        replace: true,
      })
      setHoveredId(null)
    },
    [navigate],
  )

  const visState: VisState = useMemo(
    () => ({
      clickedId,
      hoveredId,
      selectedIds,
    }),
    [clickedId, hoveredId, selectedIds],
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
      if (isMultiSelect) {
        if (!circuit) return
        const node = circuit.nodes.find((n) => n.nodeId === nodeId)
        if (
          !node ||
          (node.featureType !== 'cross layer transcoder' &&
            node.featureType !== 'lorsa' &&
            node.featureType !== 'matryoshka sae')
        )
          return

        const newSelectedIds = selectedIds.includes(nodeId)
          ? selectedIds.filter((id) => id !== nodeId)
          : [...selectedIds, nodeId]
        navigate({
          search: (prev) => ({
            ...prev,
            selectedIds: newSelectedIds.join(','),
          }),
          replace: true,
        })
      } else {
        const newClickedId = clickedId === nodeId ? null : nodeId
        navigate({
          search: (prev) => ({
            ...prev,
            clickedId: newClickedId ?? undefined,
          }),
          replace: true,
        })
      }
    },
    [circuit, clickedId, selectedIds, navigate],
  )

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredId(nodeId)
  }, [])

  const handleCircuitSelect = (selectedCircuitId: string) => {
    router.navigate({
      to: '/circuit/$id',
      params: { id: selectedCircuitId },
      search: {
        nodeThreshold: 0.6,
        edgeThreshold: 0.8,
      },
    })
    setHoveredId(null)
  }

  return (
    <div className="h-full flex flex-col overflow-hidden bg-slate-50/50">
      <div className="pt-4 pb-6 px-20 flex items-center">
        <div className="flex-1">
          {statusData?.status === 'completed' && (
            <ThresholdControls
              nodeThreshold={nodeThreshold}
              edgeThreshold={edgeThreshold}
              onThresholdsChange={handleThresholdsChange}
              isLoading={isFetchingCircuit}
            />
          )}
        </div>
        <div className="flex justify-center items-center gap-3">
          <div className="w-[500px]">
            <GraphSelector
              circuits={circuits}
              selectedCircuitId={circuitId}
              onSelect={handleCircuitSelect}
            />
          </div>
          <div className="flex gap-2">
            <NewGraphDialog
              saeSets={saeSets}
              onGraphCreated={handleGraphCreated}
            />
            {circuitData && (
              <NewGraphDialog
                saeSets={saeSets}
                onGraphCreated={handleGraphCreated}
                initialConfig={{
                  saeSetName: circuitData.saeSetName,
                  input: circuitData.input,
                  desiredLogitProb: circuitData.config.desiredLogitProb,
                  maxFeatureNodes: circuitData.config.maxFeatureNodes,
                  maxNLogits: circuitData.config.maxNLogits,
                  qkTracingTopk: circuitData.config.qkTracingTopk,
                  matryoshkaFeatureRange:
                    circuitData.config.matryoshkaFeatureRange ?? null,
                  name: circuitData.name
                    ? `${circuitData.name}-remix`
                    : undefined,
                }}
                trigger={
                  <Button
                    variant="outline"
                    className="h-14 px-4 gap-2 text-blue-700 border-blue-200 bg-blue-50 hover:bg-blue-100 hover:border-blue-300 transition-colors font-semibold"
                  >
                    <GitFork className="h-4 w-4" />
                    Remix
                  </Button>
                }
              />
            )}
          </div>
        </div>
        <div className="flex-1" />
      </div>

      {isLoadingStatus ? (
        <div className="flex-1 flex flex-col items-center justify-center pb-20">
          <div className="flex flex-col items-center gap-6 w-[400px]">
            <div className="p-4 rounded-full bg-blue-50/50 text-blue-600">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
            <div className="w-full space-y-4">
              <div className="text-center space-y-1">
                <h3 className="text-base font-medium text-slate-900">
                  Loading Circuit
                </h3>
                <p className="text-sm text-slate-500">Checking status...</p>
              </div>
            </div>
          </div>
        </div>
      ) : statusData?.status === 'pending' ||
        statusData?.status === 'running' ? (
        <div className="flex-1 flex flex-col items-center justify-center pb-20">
          <div className="flex flex-col items-center gap-6 w-[400px]">
            <div className="p-4 rounded-full bg-blue-50/50 text-blue-600">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>

            <div className="w-full space-y-4">
              <div className="text-center space-y-1">
                <h3 className="text-base font-medium text-slate-900">
                  Generating Circuit
                </h3>
                <p className="text-sm text-slate-500">
                  This may take a few minutes depending on graph complexity
                </p>
              </div>

              <div className="space-y-2">
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
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="flex justify-center mb-4">
              <AlertCircle className="h-12 w-12 text-red-500" />
            </div>
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Circuit Generation Failed
            </h3>
            <p className="text-gray-600 max-w-md">
              {statusData?.errorMessage || 'An unknown error occurred'}
            </p>
          </div>
        </div>
      ) : circuitError ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-red-600 mb-2">
              Failed to load circuit visualization
            </h3>
            <p className="text-gray-600">
              {circuitError.message || 'Failed to load circuit'}
            </p>
          </div>
        </div>
      ) : isLoadingCircuit ? (
        <div className="flex-1 flex flex-col items-center justify-center pb-20">
          <div className="flex flex-col items-center gap-6 w-[400px]">
            <div className="p-4 rounded-full bg-blue-50/50 text-blue-600">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
            <div className="w-full space-y-4">
              <div className="text-center space-y-1">
                <h3 className="text-base font-medium text-slate-900">
                  Loading Visualization
                </h3>
                <p className="text-sm text-slate-500">Fetching graph data...</p>
              </div>
            </div>
          </div>
        </div>
      ) : circuit ? (
        <div className="flex-1 flex flex-col overflow-hidden px-20 pb-20">
          <div className="flex gap-6 flex-1 overflow-hidden">
            <div className="flex flex-col gap-6 min-w-0 w-3/4 shrink-0">
              {tracedMatryoshkaRange &&
                availableMatryoshkaSubranges.length > 1 && (
                  <div className="flex justify-end shrink-0">
                    <MatryoshkaSubgraphControl
                      tracedRange={tracedMatryoshkaRange}
                      atomicSegments={availableMatryoshkaSubranges}
                      selectedRange={selectedMatryoshkaSubrange}
                      onChange={handleMatryoshkaSubrangeChange}
                    />
                  </div>
                )}
              <LinkGraphContainer
                data={circuit}
                visState={visState}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
              />
              {featureData && (
                <FeatureCardHorizontal className="grow" feature={featureData} />
              )}
            </div>

            <div className="flex flex-col flex-1 gap-4 min-w-0 h-full overflow-hidden">
              {clickedId && rawNodeIndex && rawEdgeIndex && (
                <NodeConnections
                  nodeIndex={rawNodeIndex}
                  edgeIndex={rawEdgeIndex}
                  clickedId={clickedId}
                  hoveredId={hoveredId}
                  hiddenIds={hiddenIds}
                  onNodeClick={handleNodeClick}
                  onNodeHover={handleNodeHover}
                  className={
                    selectedIds.length > 0
                      ? 'flex-[3] min-h-0'
                      : 'flex-1 min-h-0'
                  }
                />
              )}
              {selectedIds.length > 0 && circuit && (
                <SelectedFeaturesList
                  className={clickedId ? 'flex-[2] min-h-0' : 'flex-1 min-h-0'}
                  selectedIds={selectedIds}
                  circuit={circuit}
                  isGenerating={isGenerating}
                  onRemove={handleRemoveSelected}
                  onClear={handleClearSelected}
                  onTrace={handleTraceSelected}
                />
              )}
              {!clickedId && selectedIds.length === 0 && (
                <Card className="flex flex-col h-full gap-4 p-6 items-center justify-center text-slate-500 text-xl">
                  <div className="flex flex-col items-center justify-center">
                    <span>No node selected</span>
                    <span className="text-sm text-slate-500">
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
