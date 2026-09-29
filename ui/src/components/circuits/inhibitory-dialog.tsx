import { Link } from '@tanstack/react-router'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowUpRight, Loader2, MinusCircle } from 'lucide-react'
import { useState } from 'react'
import { fetchInhibitoryJob, startInhibitoryJob } from '@/api/circuits'
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
import { Textarea } from '@/components/ui/textarea'

interface InhibitoryDialogProps {
  circuitId: string
  saeName: string
  featureIndex: number
  prompt: string
}

function formatNumber(value: number) {
  return Number.isFinite(value) ? value.toPrecision(3) : '—'
}

export function InhibitoryDialog({
  circuitId,
  saeName,
  featureIndex,
  prompt,
}: InhibitoryDialogProps) {
  const [open, setOpen] = useState(false)
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
    mutationFn: startInhibitoryJob,
    onSuccess: (job) => setJobId(job.jobId),
  })
  const { data: job, error: statusError } = useQuery({
    queryKey: ['inhibitory-job', circuitId, jobId],
    queryFn: () => fetchInhibitoryJob({ data: { circuitId, jobId: jobId! } }),
    enabled: open && !!jobId,
    refetchInterval: (query) =>
      query.state.data?.status === 'completed' ||
      query.state.data?.status === 'failed'
        ? false
        : 2000,
  })

  const running =
    isPending || job?.status === 'pending' || job?.status === 'running'
  const canRun = prompts.length > 0 && prompts.length <= 32 && !running

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="h-14 gap-2 px-4">
          <MinusCircle className="h-4 w-4" />
          Inhibition
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto gap-5">
        <DialogHeader>
          <DialogTitle>Inhibitory connections</DialogTitle>
          <div className="text-sm text-slate-500 font-mono break-all">
            {saeName} · {featureIndex}
          </div>
        </DialogHeader>

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
            <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
              <Checkbox
                checked={normalized}
                onCheckedChange={(value) => setNormalized(value === true)}
              />
              Normalize score
            </label>
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
            <Loader2 className="h-4 w-4 animate-spin" />
            Computing inhibitory connections
          </div>
        )}

        {job?.status === 'completed' && job.result && (
          <div className="min-w-0 border-t pt-4 space-y-3">
            <div className="flex items-center justify-between text-sm text-slate-600">
              <span>
                {job.result.numSamples} samples · {job.result.numPositions}{' '}
                positions
              </span>
              <span>{job.result.edges.length} connections</span>
            </div>
            {job.result.edges.length === 0 ? (
              <p className="text-sm text-slate-500 py-4">
                No inhibitory connections found.
              </p>
            ) : (
              <div className="overflow-x-auto border rounded-md">
                <table className="w-full text-sm text-left tabular-nums">
                  <thead className="bg-slate-50 text-slate-500 border-b">
                    <tr>
                      <th className="px-3 py-2 font-medium">
                        Upstream feature
                      </th>
                      <th className="px-3 py-2 font-medium text-right">
                        Score
                      </th>
                      <th className="px-3 py-2 font-medium text-right">Δ</th>
                      <th className="px-3 py-2 font-medium text-right">VW</th>
                      <th className="px-3 py-2 font-medium text-right">On</th>
                      <th className="px-3 py-2 font-medium text-right">Off</th>
                    </tr>
                  </thead>
                  <tbody>
                    {job.result.edges.map((edge) => (
                      <tr
                        key={`${edge.sourceSaeName}-${edge.sourceFeature}`}
                        className="border-b last:border-0"
                      >
                        <td className="px-3 py-2 max-w-[240px]">
                          <Link
                            to="/dictionaries/$dictionaryName/features/$featureIndex"
                            params={{
                              dictionaryName: edge.sourceSaeName,
                              featureIndex: String(edge.sourceFeature),
                            }}
                            className="inline-flex items-center gap-1 text-sky-700 hover:underline max-w-full"
                            title={edge.sourceSaeName}
                          >
                            <span className="truncate">
                              {edge.sourceSaeName} #{edge.sourceFeature}
                            </span>
                            <ArrowUpRight className="h-3.5 w-3.5 shrink-0" />
                          </Link>
                        </td>
                        <td className="px-3 py-2 text-right font-medium text-emerald-700">
                          {formatNumber(
                            normalized
                              ? edge.normalizedInhibitoryScore
                              : edge.score,
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {formatNumber(edge.delta)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {formatNumber(edge.virtualWeight)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {formatNumber(edge.muOn)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {formatNumber(edge.muOff)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          <Button
            disabled={
              !canRun || !Number.isInteger(topK) || topK < 1 || topK > 50
            }
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
                },
              })
            }}
          >
            {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            Run analysis
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
