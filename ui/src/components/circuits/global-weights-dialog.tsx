import { Link } from '@tanstack/react-router'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowUpRight, GitCompareArrows, Loader2 } from 'lucide-react'
import { useState } from 'react'
import type { GlobalWeightEdge, GlobalWeightResult } from '@/api/circuits'
import {
  fetchGlobalWeightInterpretations,
  fetchGlobalWeightJob,
  startGlobalWeightJob,
} from '@/api/circuits'
import { GlobalWeightsGraph } from '@/components/circuits/global-weights-graph'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'

interface Props {
  circuitId: string
  saeName: string
  featureIndex: number
  prompt: string
}

function number(value: number | undefined) {
  return value !== undefined && Number.isFinite(value)
    ? value.toPrecision(3)
    : '—'
}

function EdgeTable({
  edges,
  direction,
  normalized,
}: {
  edges: GlobalWeightEdge[]
  direction: 'upstream' | 'downstream'
  normalized: boolean
}) {
  if (edges.length === 0) return null
  const inhibitory = edges[0].kind === 'inhibitory'
  return (
    <section className="min-w-0 space-y-2">
      <h3 className="text-sm font-semibold text-slate-800 capitalize">
        {direction} · {edges.length}
      </h3>
      <div className="overflow-x-auto border rounded-md">
        <table className="w-full text-sm text-left tabular-nums">
          <thead className="bg-slate-50 text-slate-500 border-b">
            <tr>
              <th className="px-3 py-2 font-medium">Feature</th>
              <th className="px-3 py-2 font-medium text-right">
                {inhibitory ? 'Score' : 'Weight'}
              </th>
              {inhibitory && (
                <>
                  <th className="px-3 py-2 font-medium text-right">Δ</th>
                  <th className="px-3 py-2 font-medium text-right">VW</th>
                  <th className="px-3 py-2 font-medium text-right">On</th>
                  <th className="px-3 py-2 font-medium text-right">Off</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {edges.map((edge) => {
              const saeName =
                direction === 'upstream'
                  ? edge.sourceSaeName
                  : edge.targetSaeName
              const feature =
                direction === 'upstream'
                  ? edge.sourceFeature
                  : edge.targetFeature
              const value = inhibitory
                ? normalized
                  ? edge.normalizedInhibitoryScore
                  : edge.score
                : edge.weight
              return (
                <tr
                  key={`${saeName}-${feature}`}
                  className="border-b last:border-0"
                >
                  <td className="px-3 py-2 max-w-[350px]">
                    <Link
                      to="/dictionaries/$dictionaryName/features/$featureIndex"
                      params={{
                        dictionaryName: saeName,
                        featureIndex: String(feature),
                      }}
                      className="inline-flex items-center gap-1 text-sky-700 hover:underline max-w-full"
                      title={saeName}
                    >
                      <span className="truncate">
                        {saeName} #{feature}
                      </span>
                      <ArrowUpRight className="h-3.5 w-3.5 shrink-0" />
                    </Link>
                  </td>
                  <td
                    className={`px-3 py-2 text-right font-medium ${inhibitory ? 'text-amber-700' : (value ?? 0) < 0 ? 'text-red-600' : 'text-teal-700'}`}
                  >
                    {number(value)}
                  </td>
                  {inhibitory && (
                    <>
                      <td className="px-3 py-2 text-right">
                        {number(edge.delta)}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {number(edge.virtualWeight)}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {number(edge.muOn)}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {number(edge.muOff)}
                      </td>
                    </>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export function GlobalWeightsResults({
  result,
  normalized,
}: {
  result: GlobalWeightResult
  normalized: boolean
}) {
  const features = [
    result.target,
    ...result.upstream.flatMap((edge) => [
      { saeName: edge.sourceSaeName, featureIndex: edge.sourceFeature },
      { saeName: edge.targetSaeName, featureIndex: edge.targetFeature },
    ]),
    ...result.downstream.flatMap((edge) => [
      { saeName: edge.sourceSaeName, featureIndex: edge.sourceFeature },
      { saeName: edge.targetSaeName, featureIndex: edge.targetFeature },
    ]),
  ]
  const uniqueFeatures = Array.from(
    new Map(
      features.map((feature) => [
        `${feature.saeName}:${feature.featureIndex}`,
        feature,
      ]),
    ).values(),
  )
  const needsInterpretations =
    result.target.interpretation === undefined ||
    [...result.upstream, ...result.downstream].some(
      (edge) =>
        edge.sourceInterpretation === undefined ||
        edge.targetInterpretation === undefined,
    )
  const { data: fetchedInterpretations, error: interpretationError } = useQuery(
    {
      queryKey: ['global-weight-interpretations', uniqueFeatures],
      queryFn: () =>
        fetchGlobalWeightInterpretations({
          data: { features: uniqueFeatures },
        }),
      enabled: needsInterpretations,
    },
  )
  const interpretationByFeature = new Map(
    fetchedInterpretations?.interpretations.map((item) => [
      `${item.saeName}:${item.featureIndex}`,
      item.text,
    ]) ?? [],
  )
  const withInterpretations: GlobalWeightResult = {
    ...result,
    target: {
      ...result.target,
      interpretation:
        result.target.interpretation ??
        interpretationByFeature.get(
          `${result.target.saeName}:${result.target.featureIndex}`,
        ),
    },
    upstream: result.upstream.map((edge) => ({
      ...edge,
      sourceInterpretation:
        edge.sourceInterpretation ??
        interpretationByFeature.get(
          `${edge.sourceSaeName}:${edge.sourceFeature}`,
        ),
      targetInterpretation:
        edge.targetInterpretation ??
        interpretationByFeature.get(
          `${edge.targetSaeName}:${edge.targetFeature}`,
        ),
    })),
    downstream: result.downstream.map((edge) => ({
      ...edge,
      sourceInterpretation:
        edge.sourceInterpretation ??
        interpretationByFeature.get(
          `${edge.sourceSaeName}:${edge.sourceFeature}`,
        ),
      targetInterpretation:
        edge.targetInterpretation ??
        interpretationByFeature.get(
          `${edge.targetSaeName}:${edge.targetFeature}`,
        ),
    })),
  }
  const empty = result.upstream.length === 0 && result.downstream.length === 0
  return (
    <div className="min-w-0 border-t pt-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-600">
        <span>
          {result.numSamples} samples
          {result.numPositions !== null &&
            ` · ${result.numPositions} positions`}
        </span>
        <span>
          {result.upstream.length} upstream
          {result.mode === 'global' &&
            ` · ${result.downstream.length} downstream`}
        </span>
      </div>
      {empty ? (
        <p className="py-4 text-sm text-slate-500">
          No connections found for these samples.
        </p>
      ) : (
        <>
          {interpretationError && (
            <p className="text-xs text-red-600">
              Could not load feature interpretations.
            </p>
          )}
          <GlobalWeightsGraph
            result={withInterpretations}
            normalized={normalized}
          />
          <EdgeTable
            edges={result.upstream}
            direction="upstream"
            normalized={normalized}
          />
          <EdgeTable
            edges={result.downstream}
            direction="downstream"
            normalized={normalized}
          />
        </>
      )}
    </div>
  )
}

export function GlobalWeightsDialog({
  circuitId,
  saeName,
  featureIndex,
  prompt,
}: Props) {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<'global' | 'inhibitory'>('global')
  const [promptText, setPromptText] = useState(prompt)
  const [topK, setTopK] = useState(10)
  const [normalized, setNormalized] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const prompts = promptText
    .split(/\n\s*\n/)
    .map((sample) => sample.trim())
    .filter(Boolean)

  const {
    mutate: start,
    isPending,
    error: startError,
  } = useMutation({
    mutationFn: startGlobalWeightJob,
    onSuccess: (job) => setJobId(job.jobId),
  })
  const { data: job, error: statusError } = useQuery({
    queryKey: ['global-weight-job', circuitId, jobId],
    queryFn: () => fetchGlobalWeightJob({ data: { circuitId, jobId: jobId! } }),
    enabled: open && !!jobId,
    refetchInterval: (query) =>
      query.state.data?.status === 'completed' ||
      query.state.data?.status === 'failed'
        ? false
        : 2000,
  })
  const running =
    isPending || job?.status === 'pending' || job?.status === 'running'
  const canRun =
    prompts.length > 0 &&
    prompts.length <= 32 &&
    !running &&
    Number.isInteger(topK) &&
    topK >= 1 &&
    topK <= 50

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="h-14 gap-2 px-4">
          <GitCompareArrows className="h-4 w-4" /> Global weights
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[min(96vw,1060px)] max-h-[90vh] overflow-y-auto gap-5">
        <DialogHeader>
          <DialogTitle>Global feature connections</DialogTitle>
          <div className="text-sm text-slate-500 font-mono break-all">
            {saeName} · {featureIndex}
          </div>
        </DialogHeader>
        <Tabs
          value={mode}
          onValueChange={(value) => setMode(value as 'global' | 'inhibitory')}
        >
          <TabsList className="w-full max-w-[360px]">
            <TabsTrigger value="global">Global weight</TabsTrigger>
            <TabsTrigger value="inhibitory">Inhibitory</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="grid gap-4 sm:grid-cols-[1fr_150px]">
          <label className="block space-y-2 min-w-0">
            <span className="text-sm font-medium text-slate-700">
              Sample prompts
            </span>
            <Textarea
              value={promptText}
              onChange={(event) => setPromptText(event.target.value)}
              rows={5}
              className="resize-y font-mono text-sm"
              placeholder="Separate prompts with a blank line"
            />
            <span className="block text-xs text-slate-500">
              {prompts.length} / 32 samples
            </span>
          </label>
          <div className="space-y-4">
            <label className="block space-y-2">
              <span className="text-sm font-medium text-slate-700">
                Top connections
              </span>
              <Input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>
            {mode === 'inhibitory' && (
              <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
                <Checkbox
                  checked={normalized}
                  onCheckedChange={(value) => setNormalized(value === true)}
                />{' '}
                Normalize score
              </label>
            )}
          </div>
        </div>
        {(startError || statusError || job?.status === 'failed') && (
          <p role="alert" className="text-sm text-red-600">
            {startError?.message ||
              statusError?.message ||
              job?.error ||
              'Analysis failed'}
          </p>
        )}
        {running && (
          <div
            className="flex items-center gap-2 text-sm text-slate-600"
            role="status"
          >
            <Loader2 className="h-4 w-4 animate-spin" /> Computing{' '}
            {mode === 'global' ? 'global weights' : 'inhibitory connections'}
          </div>
        )}
        {job?.status === 'completed' && job.result?.mode === mode && (
          <GlobalWeightsResults result={job.result} normalized={normalized} />
        )}
        <DialogFooter>
          <Button
            disabled={!canRun}
            onClick={() => {
              setJobId(null)
              start({
                data: {
                  circuitId,
                  saeName,
                  featureIndex,
                  prompts,
                  topK,
                  normalized,
                  mode,
                },
              })
            }}
          >
            {isPending && <Loader2 className="h-4 w-4 animate-spin" />} Run
            analysis
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
