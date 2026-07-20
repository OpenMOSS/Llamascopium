import React from 'react'
import type { PositionedNode } from '@/types/circuit'

interface TooltipsProps {
  positionedNodes: PositionedNode[]
  visState: { hoveredId: string | null }
  dimensions: { width: number; height: number }
}

const BOTTOM_PADDING = 40

export const Tooltips: React.FC<TooltipsProps> = React.memo(
  ({ positionedNodes, visState, dimensions }) => {
    if (!visState.hoveredId) {
      return null
    }

    const hoveredNode = positionedNodes.find(
      (d) => d.nodeId === visState.hoveredId,
    )

    if (!hoveredNode) {
      return null
    }

    const tooltipText =
      hoveredNode.featureType === 'cross layer transcoder' ||
      hoveredNode.featureType === 'lorsa' ||
      hoveredNode.featureType === 'matryoshka sae'
        ? hoveredNode.feature.interpretation
          ? `${hoveredNode.feature.interpretation.text}`
          : `Feature ${hoveredNode.feature.featureIndex}@${hoveredNode.saeName}`
        : hoveredNode.featureType === 'embedding'
          ? `Embedding@${hoveredNode.ctxIdx}: ${hoveredNode.token}`
          : hoveredNode.featureType === 'mlp reconstruction error'
            ? `MLP Reconstruction Error@${hoveredNode.ctxIdx}`
            : hoveredNode.featureType === 'lorsa error'
              ? `Lorsa Error@${hoveredNode.ctxIdx}`
              : hoveredNode.featureType === 'residual reconstruction error'
                ? `Residual Reconstruction Error@${hoveredNode.ctxIdx}`
                : hoveredNode.featureType === 'logit'
                  ? `Logit@${hoveredNode.ctxIdx}: ${hoveredNode.token} (${(hoveredNode.tokenProb * 100).toFixed(1)}%)`
                  : `Unknown Feature Type@${hoveredNode.ctxIdx}`

    const textWidth = tooltipText.length * 6
    const tooltipWidth = Math.max(120, textWidth + 20)
    const tooltipHeight = 20 + ('activation' in hoveredNode ? 13 : 0)
    const padding = 10

    let tooltipX = hoveredNode.pos[0] + padding
    let tooltipY = hoveredNode.pos[1] - 15

    if (tooltipX + tooltipWidth > dimensions.width - padding) {
      tooltipX = hoveredNode.pos[0] - tooltipWidth - padding
    }

    if (tooltipY < padding) {
      tooltipY = hoveredNode.pos[1] + padding
    }

    if (
      tooltipY + tooltipHeight >
      dimensions.height - BOTTOM_PADDING - padding
    ) {
      tooltipY = hoveredNode.pos[1] - tooltipHeight - padding
    }

    return (
      <g>
        <rect
          x={tooltipX}
          y={tooltipY}
          width={tooltipWidth}
          height={tooltipHeight}
          fill="rgba(0, 0, 0, 0.8)"
          rx={2}
        />
        <text
          x={tooltipX + 5}
          y={tooltipY + 13}
          fill="white"
          fontSize="10px"
          style={{ userSelect: 'none' }}
        >
          {tooltipText}
        </text>
        {'activation' in hoveredNode && (
          <text
            x={tooltipX + 5}
            y={tooltipY + 26}
            fill="orange"
            fontSize="10px"
            style={{ userSelect: 'none' }}
          >
            {hoveredNode.activation.toFixed(3)}
          </text>
        )}
      </g>
    )
  },
)

Tooltips.displayName = 'Tooltips'
