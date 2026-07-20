import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, MessageSquare, Plus, Type, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { CreateSaeSetDialog } from './create-sae-set-dialog'
import type {
  ChatMessage,
  CircuitInput,
  GenerateCircuitParams,
  MatryoshkaFeatureRange,
} from '@/api/circuits'
import {
  fetchMatryoshkaRanges,
  generateCircuit,
  previewInput,
} from '@/api/circuits'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Spinner } from '@/components/ui/spinner'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useDebounce } from '@/hooks/use-debounce'

interface NewGraphDialogProps {
  saeSets: string[]
  onGraphCreated: (circuitId: string) => void
  initialConfig?: {
    saeSetName?: string
    input?: CircuitInput
    desiredLogitProb?: number
    maxFeatureNodes?: number
    maxNLogits?: number
    qkTracingTopk?: number
    matryoshkaFeatureRange?: MatryoshkaFeatureRange | null
    name?: string
  }
  trigger?: React.ReactNode
}

export function NewGraphDialog({
  saeSets,
  onGraphCreated,
  initialConfig,
  trigger,
}: NewGraphDialogProps) {
  const queryClient = useQueryClient()
  const [dialogOpen, setDialogOpen] = useState(false)

  const [selectedSaeSet, setSelectedSaeSet] = useState<string>(
    initialConfig?.saeSetName ?? '',
  )
  const [customGraphId, setCustomGraphId] = useState(initialConfig?.name ?? '')
  const [useChatTemplate, setUseChatTemplate] = useState(
    initialConfig?.input?.inputType === 'chat_template',
  )
  const [prompt, setPrompt] = useState(
    initialConfig?.input?.inputType === 'plain_text'
      ? initialConfig.input.text
      : '',
  )
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(
    initialConfig?.input?.inputType === 'chat_template'
      ? initialConfig.input.messages
      : [
          { role: 'user', content: '' },
          { role: 'assistant', content: '' },
        ],
  )

  const [desiredLogitProb, setDesiredLogitProb] = useState(
    initialConfig?.desiredLogitProb ?? 0.98,
  )
  const [maxNodes, setMaxNodes] = useState(
    initialConfig?.maxFeatureNodes ?? 256,
  )
  const [maxLogits, setMaxLogits] = useState(initialConfig?.maxNLogits ?? 1)
  const [qkTracingTopk, setQkTracingTopk] = useState(
    initialConfig?.qkTracingTopk ?? 10,
  )
  const [matryoshkaFeatureRange, setMatryoshkaFeatureRange] =
    useState<MatryoshkaFeatureRange | null>(
      initialConfig?.matryoshkaFeatureRange ?? null,
    )

  const { data: matryoshkaOptions } = useQuery({
    queryKey: ['matryoshka-ranges', selectedSaeSet],
    queryFn: () =>
      fetchMatryoshkaRanges({ data: { saeSetName: selectedSaeSet } }),
    enabled: !!selectedSaeSet,
  })

  useEffect(() => {
    if (!matryoshkaOptions || matryoshkaFeatureRange === null) return
    const value = matryoshkaFeatureRange.join(':')
    const validRanges = [
      ...matryoshkaOptions.prefixes,
      ...matryoshkaOptions.segments,
    ]
    if (!validRanges.some((range) => range.join(':') === value)) {
      setMatryoshkaFeatureRange(null)
    }
  }, [matryoshkaFeatureRange, matryoshkaOptions])

  const {
    mutate: mutateGenerateCircuit,
    isPending: isGenerating,
    error: generateError,
  } = useMutation({
    mutationFn: generateCircuit,
    onSuccess: async (data) => {
      await queryClient.invalidateQueries({ queryKey: ['circuits'] })
      onGraphCreated(data.circuitId)
      handleDialogClose()
    },
  })

  const handleAddMessage = () => {
    const lastRole = chatMessages[chatMessages.length - 1]?.role
    const nextRole: ChatMessage['role'] =
      lastRole === 'user' ? 'assistant' : 'user'
    setChatMessages([...chatMessages, { role: nextRole, content: '' }])
  }

  const handleRemoveMessage = (index: number) => {
    if (chatMessages.length <= 1) return
    setChatMessages(chatMessages.filter((_, i) => i !== index))
  }

  const handleMessageChange = (index: number, content: string) => {
    setChatMessages(
      chatMessages.map((msg, i) => (i === index ? { ...msg, content } : msg)),
    )
  }

  const handleMessageRoleChange = (
    index: number,
    role: ChatMessage['role'],
  ) => {
    setChatMessages(
      chatMessages.map((msg, i) => (i === index ? { ...msg, role } : msg)),
    )
  }

  const hasValidInput = useChatTemplate || prompt.trim()

  const debouncedChatMessages = useDebounce(chatMessages, 500)
  const debouncedPrompt = useDebounce(prompt, 500)

  const { data: previewData, isFetching: isPreviewLoading } = useQuery({
    queryKey: [
      'preview-input',
      selectedSaeSet,
      debouncedChatMessages,
      debouncedPrompt,
      useChatTemplate,
    ],
    queryFn: () => {
      const input: CircuitInput = useChatTemplate
        ? {
            inputType: 'chat_template',
            messages: debouncedChatMessages,
          }
        : { inputType: 'plain_text', text: debouncedPrompt }

      return previewInput({
        data: {
          saeSetName: selectedSaeSet,
          input,
        },
      })
    },
    enabled: !!selectedSaeSet && (useChatTemplate || !!debouncedPrompt),
    placeholderData: (previousData, previousQuery) => {
      if (
        previousQuery?.options.queryKey?.[4] === useChatTemplate &&
        !!selectedSaeSet &&
        (useChatTemplate || !!debouncedPrompt)
      ) {
        return previousData
      }
      return undefined
    },
  })

  const previewPrompt = previewData?.prompt ?? ''
  const nextTokens = previewData?.nextTokens ?? []

  const handleStartGeneration = () => {
    if (!selectedSaeSet || !hasValidInput) return

    const input: CircuitInput = useChatTemplate
      ? {
          inputType: 'chat_template',
          messages: chatMessages,
        }
      : { inputType: 'plain_text', text: prompt }

    const params: GenerateCircuitParams = {
      saeSetName: selectedSaeSet,
      input,
      name: customGraphId.trim() || undefined,
      desiredLogitProb,
      maxFeatureNodes: maxNodes,
      maxNLogits: maxLogits,
      qkTracingTopk,
      matryoshkaFeatureRange,
    }

    mutateGenerateCircuit({ data: params })
  }

  const handleReset = () => {
    if (initialConfig?.saeSetName !== undefined) {
      setSelectedSaeSet(initialConfig.saeSetName)
    }
    setCustomGraphId(initialConfig?.name ?? '')
    setUseChatTemplate(initialConfig?.input?.inputType === 'chat_template')
    setPrompt(
      initialConfig?.input?.inputType === 'plain_text'
        ? initialConfig.input.text
        : '',
    )
    setChatMessages(
      initialConfig?.input?.inputType === 'chat_template'
        ? initialConfig.input.messages
        : [
            { role: 'user', content: '' },
            { role: 'assistant', content: '' },
          ],
    )
    setDesiredLogitProb(initialConfig?.desiredLogitProb ?? 0.98)
    setMaxNodes(initialConfig?.maxFeatureNodes ?? 256)
    setMaxLogits(initialConfig?.maxNLogits ?? 1)
    setQkTracingTopk(initialConfig?.qkTracingTopk ?? 10)
    setMatryoshkaFeatureRange(initialConfig?.matryoshkaFeatureRange ?? null)
  }

  const handleDialogClose = () => {
    setDialogOpen(false)
    handleReset()
  }

  const handleDialogOpenChange = (open: boolean) => {
    if (open) {
      setDialogOpen(true)
      handleReset()
    } else {
      handleDialogClose()
    }
  }

  const canGenerate = selectedSaeSet && hasValidInput && !isGenerating

  return (
    <Dialog open={dialogOpen} onOpenChange={handleDialogOpenChange}>
      <DialogTrigger asChild>
        {trigger || (
          <Button className="h-14 px-4 gap-2 font-semibold">
            <Plus className="h-4 w-4" />
            New Graph
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="max-w-6xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="text-xl">Generate New Graph</DialogTitle>
          <DialogDescription>
            Generate a new attribution graph for a custom prompt. Pruning
            thresholds can be adjusted after generation.
          </DialogDescription>
        </DialogHeader>

        <div className="relative flex gap-6 py-4">
          {/* Left Column: Settings */}
          <div className="w-[calc(50%-0.75rem)] flex flex-col gap-6">
            {/* Model and Source Set Selection */}
            <div className="flex gap-4">
              <div className="flex-1">
                <label className="text-xs font-semibold uppercase tracking-tight text-slate-500 mb-1.5 block">
                  Source Set
                </label>
                <div className="flex gap-2">
                  <Select
                    value={selectedSaeSet}
                    onValueChange={setSelectedSaeSet}
                  >
                    <SelectTrigger className="h-10 bg-white flex-1">
                      <SelectValue placeholder="Select a source set" />
                    </SelectTrigger>
                    <SelectContent>
                      {saeSets.map((saeSet) => (
                        <SelectItem key={saeSet} value={saeSet}>
                          {saeSet}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <CreateSaeSetDialog onSaeSetCreated={setSelectedSaeSet} />
                </div>
              </div>

              <div className="flex-1">
                <label className="text-xs font-semibold uppercase tracking-tight text-slate-500 mb-1.5 block">
                  Custom Graph ID (Optional)
                </label>
                <Input
                  value={customGraphId}
                  onChange={(e) => setCustomGraphId(e.target.value)}
                  placeholder="my-graph"
                  className="h-10"
                />
              </div>
            </div>

            {/* Advanced Settings */}
            <div className="space-y-6 border rounded-lg p-4 bg-slate-50">
              <div className="flex items-center gap-2 pb-2 border-b border-slate-200/60">
                <h4 className="text-sm font-semibold text-slate-700">
                  Attribution Settings
                </h4>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Slider
                    label="Desired Logit Probability"
                    value={desiredLogitProb}
                    onChange={setDesiredLogitProb}
                    min={0}
                    max={1}
                    step={0.01}
                    showValue={false}
                  />
                  <Slider
                    label="Max # Nodes"
                    value={maxNodes}
                    onChange={setMaxNodes}
                    min={100}
                    max={10000}
                    step={100}
                    showValue={false}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <Slider
                    label="Max # Logits"
                    value={maxLogits}
                    onChange={setMaxLogits}
                    min={1}
                    max={50}
                    step={1}
                    showValue={false}
                  />
                  <Slider
                    label="QK Tracing Top-K"
                    value={qkTracingTopk}
                    onChange={setQkTracingTopk}
                    min={0}
                    max={50}
                    step={1}
                    showValue={false}
                  />
                </div>
                {matryoshkaOptions && matryoshkaOptions.saeNames.length > 0 && (
                  <div>
                    <label className="text-xs font-medium text-slate-700 mb-1.5 block">
                      Matryoshka Residual Features
                    </label>
                    <Select
                      value={matryoshkaFeatureRange?.join(':') ?? 'disabled'}
                      onValueChange={(value) => {
                        if (value === 'disabled') {
                          setMatryoshkaFeatureRange(null)
                          return
                        }
                        const [start, end] = value.split(':').map(Number)
                        setMatryoshkaFeatureRange([start, end])
                      }}
                    >
                      <SelectTrigger className="h-10 bg-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="disabled">Disabled</SelectItem>
                        {matryoshkaOptions.prefixes.map(([start, end]) => (
                          <SelectItem
                            key={`prefix-${start}-${end}`}
                            value={`${start}:${end}`}
                          >
                            Prefix [{start}, {end})
                          </SelectItem>
                        ))}
                        {matryoshkaOptions.segments
                          .filter(([start]) => start > 0)
                          .map(([start, end]) => (
                            <SelectItem
                              key={`segment-${start}-${end}`}
                              value={`${start}:${end}`}
                            >
                              Segment [{start}, {end})
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Input */}
          <div className="absolute right-0 top-4 bottom-4 w-[calc(50%-0.5rem)] flex flex-col gap-6 overflow-y-auto px-1 [scrollbar-gutter:stable]">
            <div className="space-y-3 flex-1 flex flex-col">
              <div className="flex items-center justify-between shrink-0">
                <label className="text-xs font-semibold uppercase tracking-tight text-slate-500">
                  Input
                </label>
                {/* Input Mode Tabs */}
                <div className="flex rounded-lg bg-slate-100 p-0.5">
                  <button
                    type="button"
                    onClick={() => setUseChatTemplate(false)}
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all',
                      !useChatTemplate
                        ? 'bg-white text-slate-900 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900',
                    )}
                  >
                    <Type className="h-3.5 w-3.5" />
                    Plain Text
                  </button>
                  <button
                    type="button"
                    onClick={() => setUseChatTemplate(true)}
                    className={cn(
                      'flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all',
                      useChatTemplate
                        ? 'bg-white text-slate-900 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900',
                    )}
                  >
                    <MessageSquare className="h-3.5 w-3.5" />
                    Chat
                  </button>
                </div>
              </div>

              <div className="min-h-0 flex flex-col">
                {useChatTemplate ? (
                  <div className="space-y-2 flex-1 flex flex-col min-h-0">
                    <p className="text-xs text-slate-500 shrink-0">
                      Build a conversation. The model will predict the next
                      token after your messages.
                    </p>
                    <div className="space-y-2 rounded-lg border bg-slate-50/50 p-3 flex-1 overflow-y-auto min-h-0">
                      {chatMessages.map((message, index) => {
                        return (
                          <div
                            key={index}
                            className="relative rounded-lg border bg-white p-3 shadow-sm"
                          >
                            <div className="flex items-start gap-2">
                              <div className="flex-1 space-y-2">
                                <div className="flex items-center gap-2">
                                  <Select
                                    value={message.role}
                                    onValueChange={(
                                      value: ChatMessage['role'],
                                    ) => handleMessageRoleChange(index, value)}
                                  >
                                    <SelectTrigger className="h-6 w-auto gap-1 border-0 px-2 py-0 text-[10px] font-bold uppercase tracking-tight shadow-none bg-slate-100 text-slate-700">
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="user">User</SelectItem>
                                      <SelectItem value="assistant">
                                        Assistant
                                      </SelectItem>
                                      <SelectItem value="system">
                                        System
                                      </SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>
                                <Textarea
                                  value={message.content}
                                  onChange={(e) =>
                                    handleMessageChange(index, e.target.value)
                                  }
                                  placeholder={`What does the ${message.role} say?`}
                                  className="min-h-[50px] resize-none border-0 bg-transparent p-0 text-sm shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 overflow-hidden field-sizing-content"
                                />
                              </div>
                              {chatMessages.length > 1 && (
                                <button
                                  type="button"
                                  onClick={() => handleRemoveMessage(index)}
                                  className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                                  title="Remove message"
                                >
                                  <X className="h-4 w-4" />
                                </button>
                              )}
                            </div>
                          </div>
                        )
                      })}
                      <button
                        type="button"
                        onClick={handleAddMessage}
                        className="flex w-full items-center justify-center gap-1.5 rounded-lg border-2 border-dashed border-slate-200 py-2 text-xs font-medium text-slate-500 transition-colors hover:border-slate-300 hover:bg-white/50 hover:text-slate-700"
                      >
                        <Plus className="h-3.5 w-3.5" />
                        Add{' '}
                        {chatMessages[chatMessages.length - 1]?.role === 'user'
                          ? 'Assistant'
                          : 'User'}{' '}
                        Message
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2 flex flex-col min-h-0">
                    <p className="text-xs text-slate-500 shrink-0">
                      Enter a prompt ending mid-sentence. We&apos;ll analyze how
                      the model predicts the next token.
                    </p>
                    <Textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder='e.g., "The capital of the state containing Dallas is"'
                      className="min-h-[150px] resize-none text-sm overflow-hidden field-sizing-content"
                    />
                  </div>
                )}
              </div>

              {nextTokens.length > 0 && (
                <div className="space-y-1.5 shrink-0">
                  <div className="flex items-center justify-between">
                    <label className="text-[10px] font-bold uppercase tracking-tight text-slate-400 flex items-center gap-1.5">
                      Likely Next Tokens
                      {isPreviewLoading && (
                        <Loader2 className="h-3 w-3 animate-spin text-slate-400" />
                      )}
                    </label>
                  </div>
                  <div className="grid grid-cols-5 gap-2">
                    {nextTokens.map((token, i) => (
                      <div
                        key={i}
                        className="flex flex-col items-center justify-center gap-1 rounded-md border bg-white p-2 text-center shadow-xs"
                      >
                        <div className="font-mono text-xs font-medium text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded-sm w-full truncate">
                          {token.token.replace(/\n/g, '\\n')}
                        </div>
                        <div className="text-[10px] text-slate-500">
                          {(token.prob * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {previewPrompt && (
                <div className="space-y-1.5 shrink-0">
                  <label className="text-[10px] font-bold uppercase tracking-tight text-slate-400 flex items-center gap-1.5">
                    Formatted Prompt Preview
                    {isPreviewLoading && (
                      <Loader2 className="h-3 w-3 animate-spin text-slate-400" />
                    )}
                  </label>
                  <div className="rounded-md border bg-slate-50 p-2.5 font-mono text-xs text-slate-600 break-all whitespace-pre-wrap max-h-[300px] overflow-y-auto">
                    {previewPrompt.split('\n').map((line, i, arr) => (
                      <span key={i}>
                        {line}
                        {i < arr.length - 1 && (
                          <>
                            <span className="select-none text-slate-300">
                              ↵
                            </span>
                            {'\n'}
                          </>
                        )}
                      </span>
                    ))}
                    {previewPrompt.endsWith('\n') && (
                      <>
                        <span className="select-none text-slate-300">↵</span>
                        {'\u200B'}
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>

            {generateError && (
              <p className="text-sm text-red-600">
                {generateError.message || 'Failed to generate graph'}
              </p>
            )}
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button
            variant="outline"
            onClick={handleReset}
            disabled={isGenerating}
          >
            Reset
          </Button>
          <Button onClick={handleStartGeneration} disabled={!canGenerate}>
            {isGenerating ? (
              <>
                <Spinner isAnimating={true} className="mr-2" />
                Starting...
              </>
            ) : (
              'Start Generation'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
