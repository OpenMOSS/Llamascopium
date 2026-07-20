import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  AlertCircle,
  ChevronDown,
  GitBranch,
  Loader2,
  MessageCircle,
  Trash2,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from '@tanstack/react-router'
import type {
  CircuitFeatureTargetInput,
  CircuitInput,
  CircuitListItem,
  CircuitStatus,
} from '@/api/circuits'
import { deleteCircuit } from '@/api/circuits'
import { cn } from '@/lib/utils'

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSeconds = Math.floor(diffMs / 1000)
  const diffMinutes = Math.floor(diffSeconds / 60)
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffSeconds < 60) {
    return 'just now'
  }
  if (diffMinutes < 60) {
    return `${diffMinutes} minute${diffMinutes === 1 ? '' : 's'} ago`
  }
  if (diffHours < 24) {
    return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`
  }
  if (diffDays < 7) {
    return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`
  }

  return date.toLocaleDateString()
}

function formatFeaturesDisplay(
  features: CircuitFeatureTargetInput[] | undefined,
  maxLength: number,
): string {
  if (!features || features.length === 0) return 'No features'

  const featureStrings = features.map((f) => {
    if (Array.isArray(f)) {
      const [layer, index, pos, isLorsa] = f
      return `${isLorsa ? 'A' : 'M'}${layer}#${index}@${pos}`
    }
    const layer = f.saeName.match(/layer(\d+)/)?.[1] ?? '?'
    const prefix = f.saeName.includes('matryoshka')
      ? 'R'
      : f.saeName.includes('lorsa')
        ? 'A'
        : 'M'
    return `${prefix}${layer}#${f.featureIndex}@${f.position}`
  })

  const combined = featureStrings.join(', ')
  return combined.length > maxLength
    ? `${combined.slice(0, maxLength)}...`
    : combined
}

function formatInputDisplay(
  input: CircuitInput | undefined,
  prompt: string,
  maxLength: number,
): { text: string; isChatTemplate: boolean } {
  if (!input || input.inputType === 'plain_text') {
    return {
      text:
        prompt.length > maxLength ? `${prompt.slice(0, maxLength)}...` : prompt,
      isChatTemplate: false,
    }
  }

  // For chat template, show a summary of messages
  const messages = input.messages
  if (messages.length === 0) {
    return { text: '(empty chat)', isChatTemplate: true }
  }

  // Create a compact representation of the conversation
  const parts = messages.map((m) => {
    const rolePrefix =
      m.role === 'user' ? 'U' : m.role === 'assistant' ? 'A' : 'S'
    const contentPreview =
      m.content.length > 30 ? `${m.content.slice(0, 30)}...` : m.content
    return `[${rolePrefix}] ${contentPreview}`
  })

  const combined = parts.join(' → ')
  return {
    text:
      combined.length > maxLength
        ? `${combined.slice(0, maxLength)}...`
        : combined,
    isChatTemplate: true,
  }
}

function StatusIndicator({
  status,
  progress,
}: {
  status: CircuitStatus
  progress: number
}) {
  if (status === 'completed') {
    return null
  }

  if (status === 'pending' || status === 'running') {
    return (
      <div className="flex items-center gap-1.5 text-blue-600">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <span className="text-[10px] font-medium">
          {status === 'pending' ? 'Pending...' : `${Math.round(progress)}%`}
        </span>
      </div>
    )
  }

  if (status === 'failed') {
    return (
      <div className="flex items-center gap-1 text-red-600">
        <AlertCircle className="h-3.5 w-3.5" />
        <span className="text-[10px] font-medium">Failed</span>
      </div>
    )
  }

  return null
}

interface GraphSelectorProps {
  circuits: CircuitListItem[]
  selectedCircuitId: string
  onSelect: (circuitId: string) => void
}

export function GraphSelector({
  circuits,
  selectedCircuitId,
  onSelect,
}: GraphSelectorProps) {
  const router = useRouter()
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const queryClient = useQueryClient()

  const { mutate: mutateDeleteCircuit } = useMutation({
    mutationFn: deleteCircuit,
    onSuccess: async (_, variables) => {
      await queryClient.invalidateQueries({ queryKey: ['circuits'] })
      router.invalidate()
      if (variables.data.circuitId === selectedCircuitId) {
        router.navigate({
          to: '/circuits',
        })
      }
    },
  })

  const handleDeleteCircuit = (
    e: React.MouseEvent,
    circuit: CircuitListItem,
  ) => {
    e.preventDefault()
    e.stopPropagation()
    if (!confirm(`Delete graph "${circuit.name || circuit.id}"?`)) return
    mutateDeleteCircuit({ data: { circuitId: circuit.id } })
  }

  const handleSelect = (circuit: CircuitListItem) => {
    // Don't allow selection of non-completed circuits
    if (circuit.status !== 'completed') {
      return
    }
    onSelect(circuit.id)
    setIsOpen(false)
  }

  const groupedCircuits = useMemo(() => {
    const groups: Record<string, CircuitListItem[]> = {}
    circuits.forEach((c) => {
      const g = c.group || 'Ungrouped'
      if (!groups[g]) groups[g] = []
      groups[g].push(c)
    })
    return groups
  }, [circuits])

  const groupOrder = useMemo(() => {
    const groups = Object.keys(groupedCircuits).sort()
    const ungroupedIdx = groups.indexOf('Ungrouped')
    if (ungroupedIdx !== -1) {
      groups.splice(ungroupedIdx, 1)
      groups.push('Ungrouped')
    }
    return groups
  }, [groupedCircuits])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedCircuit = circuits.find((c) => c.id === selectedCircuitId)

  return (
    <div ref={containerRef} className="relative w-full">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex h-14 w-full items-center justify-between rounded-md border border-input bg-white px-4 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
      >
        {selectedCircuit ? (
          <div className="flex flex-col items-start text-left gap-0.5 overflow-hidden flex-1 mr-2">
            {(() => {
              const isSubgraph = !!selectedCircuit.parentId
              const displayText =
                isSubgraph && selectedCircuit.config.listOfFeatures
                  ? formatFeaturesDisplay(
                      selectedCircuit.config.listOfFeatures,
                      60,
                    )
                  : formatInputDisplay(
                      selectedCircuit.input,
                      selectedCircuit.prompt,
                      60,
                    ).text
              const isChatTemplate =
                !isSubgraph &&
                selectedCircuit.input?.inputType === 'chat_template'

              return (
                <span className="text-sm font-medium truncate w-full flex items-center gap-1.5">
                  {isSubgraph ? (
                    <GitBranch className="h-3.5 w-3.5 shrink-0 text-slate-400 rotate-180" />
                  ) : isChatTemplate ? (
                    <MessageCircle className="h-3.5 w-3.5 shrink-0 text-primary" />
                  ) : null}
                  {displayText}
                </span>
              )
            })()}
            <div className="flex items-center gap-2 text-xs text-slate-500 truncate w-full">
              <span className="truncate">
                {selectedCircuit.name || selectedCircuit.id}
              </span>
              <span className="shrink-0">·</span>
              <span className="shrink-0">
                {formatRelativeTime(selectedCircuit.createdAt)}
              </span>
            </div>
          </div>
        ) : (
          <span className="text-slate-500">Select a graph</span>
        )}
        <ChevronDown
          className={cn(
            'h-4 w-4 opacity-50 transition-transform shrink-0',
            isOpen && 'rotate-180',
          )}
        />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 z-50 mt-1 max-h-[500px] flex flex-col rounded-md border bg-white shadow-md">
          <div className="overflow-auto flex-1">
            {circuits.length === 0 ? (
              <div className="py-4 px-3 text-sm text-slate-500 text-center">
                No graphs available. Create one to get started.
              </div>
            ) : (
              <div className="p-1">
                {groupOrder.map((group) => {
                  const groupItems = groupedCircuits[group]
                  return (
                    <div key={group}>
                      {group !== 'Ungrouped' ||
                      Object.keys(groupedCircuits).length > 1 ? (
                        <div className="px-3 py-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-wider bg-slate-50/50">
                          {group}
                        </div>
                      ) : null}
                      {groupItems.map((c) => {
                        if (
                          c.parentId &&
                          groupItems.some((p) => p.id === c.parentId)
                        ) {
                          return null
                        }

                        const renderCircuitItem = (
                          item: CircuitListItem,
                          depth = 0,
                        ) => {
                          const itemChildren = groupItems.filter(
                            (child) => child.parentId === item.id,
                          )
                          const isSelectable = item.status === 'completed'
                          const isInProgress =
                            item.status === 'pending' ||
                            item.status === 'running'
                          const isFailed = item.status === 'failed'

                          return (
                            <div key={item.id}>
                              <div
                                className={cn(
                                  'flex items-center justify-between gap-2 rounded-sm px-3 py-2 group transition-colors',
                                  isSelectable && 'cursor-pointer',
                                  !isSelectable && 'cursor-not-allowed',
                                  item.id === selectedCircuitId
                                    ? 'bg-slate-100'
                                    : isSelectable
                                      ? 'hover:bg-slate-50'
                                      : 'opacity-70',
                                  depth > 0 &&
                                    'ml-4 border-l-2 border-slate-200 pl-4 py-1.5',
                                  isInProgress && 'bg-blue-50/50',
                                  isFailed && 'bg-red-50/50',
                                )}
                                onClick={() => handleSelect(item)}
                              >
                                <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                                  {(() => {
                                    const isSubgraph = !!item.parentId
                                    const displayText =
                                      isSubgraph && item.config.listOfFeatures
                                        ? formatFeaturesDisplay(
                                            item.config.listOfFeatures,
                                            depth > 0 ? 40 : 50,
                                          )
                                        : formatInputDisplay(
                                            item.input,
                                            item.prompt,
                                            depth > 0 ? 40 : 50,
                                          ).text
                                    const isChatTemplate =
                                      !isSubgraph &&
                                      item.input?.inputType === 'chat_template'

                                    return (
                                      <span
                                        className={cn(
                                          'text-sm font-medium truncate flex items-center gap-1.5',
                                          depth > 0 && 'text-slate-600',
                                          isInProgress && 'text-blue-700',
                                          isFailed && 'text-red-700',
                                        )}
                                      >
                                        {isSubgraph ? (
                                          <GitBranch className="h-3 w-3 shrink-0 rotate-180 text-slate-400" />
                                        ) : isChatTemplate ? (
                                          <MessageCircle className="h-3.5 w-3.5 shrink-0 text-primary" />
                                        ) : null}
                                        {displayText}
                                      </span>
                                    )
                                  })()}
                                  <div className="flex items-center gap-2 text-[10px] text-slate-500">
                                    <span className="truncate">
                                      {item.name || item.id}
                                    </span>
                                    <span className="shrink-0">·</span>
                                    <span className="shrink-0">
                                      {formatRelativeTime(item.createdAt)}
                                    </span>
                                    {item.status !== 'completed' && (
                                      <>
                                        <span className="shrink-0">·</span>
                                        <StatusIndicator
                                          status={item.status}
                                          progress={item.progress}
                                        />
                                      </>
                                    )}
                                  </div>
                                </div>
                                <button
                                  type="button"
                                  onClick={(e) => handleDeleteCircuit(e, item)}
                                  className="p-1.5 rounded hover:bg-red-100 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                                  title="Delete graph"
                                >
                                  <Trash2 className="h-4 w-4 text-red-500" />
                                </button>
                              </div>
                              {itemChildren.map((child) =>
                                renderCircuitItem(child, depth + 1),
                              )}
                            </div>
                          )
                        }

                        return renderCircuitItem(c)
                      })}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
