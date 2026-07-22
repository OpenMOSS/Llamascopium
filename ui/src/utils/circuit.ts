import type { CircuitData, Edge, Node } from '@/types/circuit'
import type { MatryoshkaFeatureRange } from '@/api/circuits'

function strongestEdgesForFraction(edges: Edge[], threshold: number): Edge[] {
  if (edges.length <= 1 || threshold >= 1) return edges

  const ranked = [...edges].sort(
    (left, right) => Math.abs(right.weight) - Math.abs(left.weight),
  )
  const totalWeight = ranked.reduce(
    (total, edge) => total + Math.abs(edge.weight),
    0,
  )
  if (totalWeight === 0) return ranked.slice(0, 1)

  const targetWeight = totalWeight * Math.max(0, threshold)
  let retainedWeight = 0
  let retainedCount = 0
  do {
    retainedWeight += Math.abs(ranked[retainedCount].weight)
    retainedCount += 1
  } while (retainedCount < ranked.length && retainedWeight < targetWeight)

  return ranked.slice(0, retainedCount)
}

export function filterCircuitByMatryoshkaRange(
  data: CircuitData,
  range: MatryoshkaFeatureRange,
  upstreamEdgeThreshold: number = 1,
): CircuitData {
  const [start, end] = range
  const selectedMatryoshkaIds = new Set(
    data.nodes
      .filter(
        (node) =>
          node.featureType === 'matryoshka sae' &&
          node.feature.featureIndex >= start &&
          node.feature.featureIndex < end,
      )
      .map((node) => node.nodeId),
  )

  if (selectedMatryoshkaIds.size === 0) {
    return { ...data, nodes: [], edges: [] }
  }

  const incoming = new Map<string, Edge[]>()
  const outgoing = new Map<string, Edge[]>()
  for (const edge of data.edges) {
    incoming.set(edge.target, [...(incoming.get(edge.target) ?? []), edge])
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge])
  }

  const nodeById = new Map(data.nodes.map((node) => [node.nodeId, node]))
  const keep = new Set(selectedMatryoshkaIds)

  const isAllowed = (nodeId: string) => {
    const node = nodeById.get(nodeId)
    return Boolean(
      node &&
      (node.featureType !== 'matryoshka sae' ||
        selectedMatryoshkaIds.has(nodeId)),
    )
  }

  // Reapply the edge budget within the selected range's upstream causal cone.
  // The backend threshold is global to the full trace, so plain reachability
  // otherwise retains nearly every earlier feature connected by a weak edge.
  const upstreamFrontier = [...selectedMatryoshkaIds]
  const upstreamVisited = new Set<string>()
  while (upstreamFrontier.length > 0) {
    const current = upstreamFrontier.pop()!
    if (upstreamVisited.has(current)) continue
    upstreamVisited.add(current)

    const eligibleEdges = (incoming.get(current) ?? []).filter((edge) =>
      isAllowed(edge.source),
    )
    for (const edge of strongestEdgesForFraction(
      eligibleEdges,
      upstreamEdgeThreshold,
    )) {
      keep.add(edge.source)
      upstreamFrontier.push(edge.source)
    }
  }

  // Output paths are already globally pruned and must remain intact so the
  // selected Matryoshka features stay connected to their traced logits.
  const downstreamFrontier = [...selectedMatryoshkaIds]
  const downstreamVisited = new Set<string>()
  while (downstreamFrontier.length > 0) {
    const current = downstreamFrontier.pop()!
    if (downstreamVisited.has(current)) continue
    downstreamVisited.add(current)

    for (const edge of outgoing.get(current) ?? []) {
      if (!isAllowed(edge.target)) continue
      keep.add(edge.target)
      downstreamFrontier.push(edge.target)
    }
  }

  return {
    ...data,
    nodes: data.nodes.filter((node) => keep.has(node.nodeId)),
    edges: data.edges.filter(
      (edge) => keep.has(edge.source) && keep.has(edge.target),
    ),
  }
}

export function extractLayerAndFeature(
  nodeId: string,
): { layer: number; featureId: number; isLorsa: boolean } | null {
  try {
    const parts = nodeId.split('_')
    const layer = Math.floor(parseInt(parts[0]) / 2)
    const isLorsa = parseInt(parts[0]) % 2 === 0
    const featureId = parseInt(parts[1])
    if (isNaN(layer) || isNaN(featureId)) {
      return null
    }
    return { layer, featureId, isLorsa }
  } catch {
    console.error('Error extracting layer and feature from node ID')
    return null
  }
}

export function getNodeColor(featureType: string): string {
  switch (featureType) {
    case 'logit':
      return '#ff6b6b'
    case 'embedding':
      return '#69b3a2'
    case 'cross layer transcoder':
      return '#f59f00'
    case 'lorsa':
      return '#339af0'
    case 'matryoshka sae':
      return '#0f766e'
    default:
      return '#95a5a6'
  }
}

export function getEdgeColor(weight: number): string {
  return weight > 0 ? '#4CAF50' : '#F44336'
}

export function getEdgeStrokeWidth(weight: number): number {
  return Math.max(0.5, Math.min(3, Math.abs(weight) * 10))
}

export function formatFeatureId(node: Node, verbose: boolean = true): string {
  const layerIdx = node.layer + 1
  if (node.featureType === 'cross layer transcoder') {
    const mlpLayer = Math.floor(layerIdx / 2) - 1
    const featureId = node.feature.featureIndex
    return verbose ? `M${mlpLayer}#${featureId}@${node.ctxIdx}` : `M${mlpLayer}`
  } else if (node.featureType === 'lorsa') {
    const attnLayer = Math.floor(layerIdx / 2)
    const featureId = node.feature.featureIndex
    return verbose
      ? `A${attnLayer}#${featureId}@${node.ctxIdx}`
      : `A${attnLayer}`
  } else if (node.featureType === 'matryoshka sae') {
    const residualLayer = Math.floor(node.layer / 2) - 1
    const featureId = node.feature.featureIndex
    return verbose
      ? `R${residualLayer}#${featureId}@${node.ctxIdx}`
      : `R${residualLayer}`
  } else if (node.featureType === 'embedding') {
    return `Emb@${node.ctxIdx}: ${node.token}`
  } else if (node.featureType === 'mlp reconstruction error') {
    return `M${Math.floor(layerIdx / 2) - 1}Error@${node.ctxIdx}`
  } else if (node.featureType === 'lorsa error') {
    return `A${Math.floor(layerIdx / 2)}Error@${node.ctxIdx}`
  } else if (node.featureType === 'residual reconstruction error') {
    return `R${Math.floor(node.layer / 2) - 1}Error@${node.ctxIdx}`
  } else if (node.featureType === 'logit') {
    return `Logit@${node.ctxIdx}: ${node.token} (${(node.tokenProb * 100).toFixed(1)}%)`
  } else if (node.featureType === 'bias') {
    return `Bias@${node.ctxIdx}`
  }
  return 'Unknown feature type'
}
