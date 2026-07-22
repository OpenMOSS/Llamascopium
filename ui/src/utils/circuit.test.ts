import { describe, expect, it } from 'vitest'
import { filterCircuitByMatryoshkaRange } from './circuit'
import type { CircuitData, Node } from '@/types/circuit'

function node(
  nodeId: string,
  featureType: Node['featureType'],
  featureIndex?: number,
): Node {
  return {
    nodeId,
    featureType,
    layer: 0,
    ctxIdx: 0,
    isFromQkTracing: false,
    ...(featureType === 'matryoshka sae'
      ? {
          isTargetLogit: false,
          saeName: 'matry',
          activation: 1,
          feature: { featureIndex },
          qkTracingResults: null,
          matryoshkaFeatureRange: [2048, 16384],
        }
      : featureType === 'logit'
        ? { tokenProb: 1, token: 'x' }
        : featureType === 'embedding'
          ? { token: 'x' }
          : {}),
  } as Node
}

describe('filterCircuitByMatryoshkaRange', () => {
  it('keeps only ancestors and descendants of the selected subsegment', () => {
    const data: CircuitData = {
      metadata: { promptTokens: ['x'] },
      nodes: [
        node('source-a', 'embedding'),
        node('source-b', 'embedding'),
        node('matry-a', 'matryoshka sae', 3000),
        node('matry-b', 'matryoshka sae', 9000),
        node('logit', 'logit'),
      ],
      edges: [
        { source: 'source-a', target: 'matry-a', weight: 1 },
        { source: 'source-b', target: 'matry-b', weight: 1 },
        { source: 'matry-a', target: 'logit', weight: 1 },
        { source: 'matry-b', target: 'logit', weight: 1 },
      ],
    }

    const filtered = filterCircuitByMatryoshkaRange(data, [2048, 4096])

    expect(filtered.nodes.map((item) => item.nodeId)).toEqual([
      'source-a',
      'matry-a',
      'logit',
    ])
    expect(filtered.edges).toEqual([
      { source: 'source-a', target: 'matry-a', weight: 1 },
      { source: 'matry-a', target: 'logit', weight: 1 },
    ])
  })

  it('reapplies the edge budget to the selected range upstream', () => {
    const data: CircuitData = {
      metadata: { promptTokens: ['x'] },
      nodes: [
        node('source-strong', 'embedding'),
        node('source-weak', 'embedding'),
        node('mlp-strong', 'cross layer transcoder'),
        node('mlp-weak', 'cross layer transcoder'),
        node('matry', 'matryoshka sae', 3000),
        node('logit', 'logit'),
      ],
      edges: [
        { source: 'source-strong', target: 'mlp-strong', weight: 1 },
        { source: 'source-weak', target: 'mlp-weak', weight: 1 },
        { source: 'mlp-strong', target: 'matry', weight: 0.9 },
        { source: 'mlp-weak', target: 'matry', weight: 0.1 },
        { source: 'matry', target: 'logit', weight: 1 },
      ],
    }

    const filtered = filterCircuitByMatryoshkaRange(data, [2048, 4096], 0.8)

    expect(filtered.nodes.map((item) => item.nodeId)).toEqual([
      'source-strong',
      'mlp-strong',
      'matry',
      'logit',
    ])
    expect(filtered.edges).toEqual([
      { source: 'source-strong', target: 'mlp-strong', weight: 1 },
      { source: 'mlp-strong', target: 'matry', weight: 0.9 },
      { source: 'matry', target: 'logit', weight: 1 },
    ])
  })
})
