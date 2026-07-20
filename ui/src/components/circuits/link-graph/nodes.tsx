import React, { useCallback, useMemo } from 'react'
import type { PositionedNode } from '@/types/circuit'
import type { EdgeIndex } from '@/utils/circuit-index'
import { getNodeColor } from '@/utils/circuit'
import { isNodeConnected } from '@/utils/circuit-index'

interface NodesProps {
  positionedNodes: PositionedNode[]
  edgeIndex: EdgeIndex
  visState: {
    clickedId: string | null
    hoveredId: string | null
    selectedIds?: string[]
  }
}

export const Nodes: React.FC<NodesProps> = React.memo(
  ({ positionedNodes, edgeIndex, visState }) => {
    const isConnected = useCallback(
      (nodeId: string) => {
        if (!visState.clickedId) return false
        return isNodeConnected(edgeIndex, visState.clickedId, nodeId)
      },
      [visState.clickedId, edgeIndex],
    )

    const isErrorNode = useCallback(
      (d: PositionedNode) =>
        d.featureType === 'lorsa error' ||
        d.featureType === 'mlp reconstruction error' ||
        d.featureType === 'residual reconstruction error',
      [],
    )

    const { regularNodes, errorNodes } = useMemo(() => {
      const regular: PositionedNode[] = []
      const error: PositionedNode[] = []
      for (const node of positionedNodes) {
        if (isErrorNode(node)) {
          error.push(node)
        } else {
          regular.push(node)
        }
      }
      return { regularNodes: regular, errorNodes: error }
    }, [positionedNodes, isErrorNode])

    return (
      <g>
        {regularNodes.map((d) => (
          <circle
            key={d.nodeId}
            className="node"
            cx={d.pos[0]}
            cy={d.pos[1]}
            r={3}
            fill={getNodeColor(d.featureType)}
            stroke={
              d.nodeId === visState.clickedId
                ? '#ef4444'
                : visState.selectedIds?.includes(d.nodeId)
                  ? '#22c55e'
                  : '#000'
            }
            strokeWidth={
              d.nodeId === visState.clickedId ||
              visState.selectedIds?.includes(d.nodeId) ||
              isConnected(d.nodeId)
                ? '1.5'
                : '0.5'
            }
            style={{
              pointerEvents: 'none',
              transition: 'all 0.2s ease',
            }}
          />
        ))}

        {errorNodes.map((d) => (
          <rect
            key={d.nodeId}
            className="node"
            x={d.pos[0] - 3}
            y={d.pos[1] - 3}
            width={6}
            height={6}
            fill={getNodeColor(d.featureType)}
            stroke={
              d.nodeId === visState.clickedId
                ? '#ef4444'
                : visState.selectedIds?.includes(d.nodeId)
                  ? '#22c55e'
                  : '#000'
            }
            strokeWidth={
              d.nodeId === visState.clickedId ||
              visState.selectedIds?.includes(d.nodeId) ||
              isConnected(d.nodeId)
                ? '1.5'
                : '0.5'
            }
            style={{
              pointerEvents: 'none',
              transition: 'all 0.2s ease',
            }}
          />
        ))}

        {regularNodes.map((d) => (
          <circle
            key={`hover-${d.nodeId}`}
            className="hover-indicator"
            cx={d.pos[0]}
            cy={d.pos[1]}
            r={6}
            stroke="#f0f"
            strokeWidth={2}
            fill="none"
            style={{
              pointerEvents: 'none',
              opacity: d.nodeId === visState.hoveredId ? 1 : 0,
            }}
          />
        ))}

        {errorNodes.map((d) => (
          <rect
            key={`hover-${d.nodeId}`}
            className="hover-indicator"
            x={d.pos[0] - 6}
            y={d.pos[1] - 6}
            width={12}
            height={12}
            stroke="#f0f"
            strokeWidth={2}
            fill="none"
            style={{
              pointerEvents: 'none',
              opacity: d.nodeId === visState.hoveredId ? 1 : 0,
            }}
          />
        ))}
      </g>
    )
  },
)

Nodes.displayName = 'Nodes'
