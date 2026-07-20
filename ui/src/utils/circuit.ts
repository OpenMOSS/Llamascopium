import type { Node } from '@/types/circuit'

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
