import { Layers3 } from 'lucide-react'
import type { MatryoshkaFeatureRange } from '@/api/circuits'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface MatryoshkaSubgraphControlProps {
  tracedRange: MatryoshkaFeatureRange
  atomicSegments: MatryoshkaFeatureRange[]
  selectedRange: MatryoshkaFeatureRange | null
  onChange: (range: MatryoshkaFeatureRange | null) => void
}

export function MatryoshkaSubgraphControl({
  tracedRange,
  atomicSegments,
  selectedRange,
  onChange,
}: MatryoshkaSubgraphControlProps) {
  const value = selectedRange?.join(':') ?? 'all'

  return (
    <div className="flex items-center gap-2">
      <Layers3 className="h-4 w-4 text-slate-500" />
      <Select
        value={value}
        onValueChange={(nextValue) => {
          if (nextValue === 'all') {
            onChange(null)
            return
          }
          const [start, end] = nextValue.split(':').map(Number)
          onChange([start, end])
        }}
      >
        <SelectTrigger className="h-9 w-[230px] bg-white">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">
            All [{tracedRange[0]}, {tracedRange[1]})
          </SelectItem>
          {atomicSegments.map(([start, end]) => (
            <SelectItem key={`${start}-${end}`} value={`${start}:${end}`}>
              Subsegment [{start}, {end})
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
