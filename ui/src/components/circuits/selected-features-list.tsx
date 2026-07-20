import { memo, useMemo } from 'react'
import { Play, X } from 'lucide-react'
import { Button } from '../ui/button'
import { Card } from '../ui/card'
import type { CircuitData, FeatureNode } from '@/types/circuit'
import { formatFeatureId } from '@/utils/circuit'
import { cn } from '@/lib/utils'

interface SelectedFeaturesListProps {
  selectedIds: string[]
  circuit: CircuitData
  isGenerating: boolean
  onRemove: (id: string) => void
  onClear: () => void
  onTrace: () => void
  className?: string
}

export const SelectedFeaturesList = memo(
  ({
    selectedIds,
    circuit,
    isGenerating,
    onRemove,
    onClear,
    onTrace,
    className,
  }: SelectedFeaturesListProps) => {
    const selectedNodes = useMemo(() => {
      return selectedIds
        .map((id) => circuit.nodes.find((n) => n.nodeId === id))
        .filter(
          (n): n is FeatureNode =>
            !!n &&
            (n.featureType === 'lorsa' ||
              n.featureType === 'cross layer transcoder' ||
              n.featureType === 'matryoshka sae'),
        )
    }, [selectedIds, circuit.nodes])

    if (selectedNodes.length === 0) return null

    return (
      <Card className={cn('flex flex-col min-w-0 gap-4 p-4', className)}>
        <div className="flex items-center justify-between border-b pb-2 shrink-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm text-slate-700">
              SELECTED FEATURES ({selectedNodes.length})
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onClear}
              className="h-7 text-xs"
            >
              Clear
            </Button>
            <Button
              size="sm"
              onClick={onTrace}
              disabled={isGenerating}
              className="h-7 text-xs gap-1.5"
            >
              <Play className="w-3 h-3" />
              Trace
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto min-h-0 flex flex-col gap-1 pr-1">
          {selectedNodes.map((node) => (
            <div
              key={node.nodeId}
              className="flex items-center justify-between p-2 rounded border bg-slate-50 border-slate-200 group hover:border-slate-300 transition-colors"
            >
              <div className="flex flex-col min-w-0 gap-0.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-gray-500 shrink-0">
                    {formatFeatureId(node, false)}
                  </span>
                  <span
                    className="text-xs font-medium truncate"
                    title={node.feature.interpretation?.text}
                  >
                    {node.feature.interpretation?.text || 'No interpretation'}
                  </span>
                </div>
                <div className="text-[10px] text-gray-400">
                  Act: {node.activation.toFixed(3)} • Ctx: {node.ctxIdx}
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                onClick={() => onRemove(node.nodeId)}
              >
                <X className="h-3.5 w-3.5 text-gray-400 hover:text-red-500" />
              </Button>
            </div>
          ))}
        </div>
      </Card>
    )
  },
)

SelectedFeaturesList.displayName = 'SelectedFeaturesList'
