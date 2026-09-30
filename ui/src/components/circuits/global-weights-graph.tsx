import { Link } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'
import type { GlobalWeightEdge, GlobalWeightResult } from '@/api/circuits'

type Point = { x: number; y: number }
type Geometry = {
  width: number
  height: number
  center: Point
  nodes: Record<string, Point>
}

function shortName(hook: string, feature: number) {
  const layer = hook.match(/^blocks\.(\d+)\./)?.[1] ?? '?'
  return `L${layer}${hook.includes('attn') ? 'A' : 'M'}#${feature}`
}

function edgeValue(edge: GlobalWeightEdge, normalized: boolean) {
  return edge.kind === 'inhibitory'
    ? normalized
      ? (edge.normalizedInhibitoryScore ?? 0)
      : (edge.score ?? 0)
    : (edge.weight ?? 0)
}

function edgeColor(edge: GlobalWeightEdge) {
  return edge.kind === 'inhibitory'
    ? '#e8a34d'
    : (edge.weight ?? 0) < 0
      ? '#fb7185'
      : '#60a5fa'
}

function lanes(edges: GlobalWeightEdge[], left: boolean) {
  const groups: { edge: GlobalWeightEdge; index: number }[][] = [[], [], []]
  edges.forEach((edge, index) => {
    const lane = left ? 2 - (index % 3) : index % 3
    groups[lane].push({ edge, index })
  })
  return groups
}

export function GlobalWeightsGraph({
  result,
  normalized,
  missingInterpretationLabel = 'No interpretation available',
}: {
  result: GlobalWeightResult
  normalized: boolean
  missingInterpretationLabel?: string
}) {
  const graphRef = useRef<HTMLDivElement>(null)
  const centerRef = useRef<HTMLSpanElement>(null)
  const nodeRefs = useRef<Record<string, HTMLSpanElement | null>>({})
  const [geometry, setGeometry] = useState<Geometry | null>(null)
  const upstream = result.upstream
  const downstream = result.downstream
  const maxValue = Math.max(
    1e-8,
    ...[...upstream, ...downstream].map((edge) =>
      Math.abs(edgeValue(edge, normalized)),
    ),
  )

  useEffect(() => {
    const graph = graphRef.current
    const center = centerRef.current
    if (!graph || !center) return
    const update = () => {
      const rect = graph.getBoundingClientRect()
      const point = (element: Element): Point => {
        const bounds = element.getBoundingClientRect()
        return {
          x: bounds.left - rect.left + bounds.width / 2,
          y: bounds.top - rect.top + bounds.height / 2,
        }
      }
      const nodes: Record<string, Point> = {}
      for (const [key, element] of Object.entries(nodeRefs.current)) {
        if (element) nodes[key] = point(element)
      }
      setGeometry({
        width: rect.width,
        height: rect.height,
        center: point(center),
        nodes,
      })
    }
    const observer = new ResizeObserver(update)
    observer.observe(graph)
    observer.observe(center)
    for (const element of Object.values(nodeRefs.current)) {
      if (element) observer.observe(element.parentElement ?? element)
    }
    update()
    return () => observer.disconnect()
  }, [result])

  const renderNode = (edge: GlobalWeightEdge, index: number, left: boolean) => {
    const key = `${left ? 'up' : 'down'}-${index}`
    const saeName = left ? edge.sourceSaeName : edge.targetSaeName
    const featureIndex = left ? edge.sourceFeature : edge.targetFeature
    const hook = left ? edge.source : edge.target
    const interpretation = left
      ? edge.sourceInterpretation
      : edge.targetInterpretation
    const label = (
      <span className="min-w-0 flex-1 text-left">
        <span className="block font-mono text-[11px] font-semibold text-slate-100">
          {shortName(hook, featureIndex)}
        </span>
        <span className="block break-words text-[11px] leading-[1.35] text-slate-300">
          {interpretation || missingInterpretationLabel}
        </span>
      </span>
    )
    return (
      <Link
        key={key}
        to="/dictionaries/$dictionaryName/features/$featureIndex"
        params={{ dictionaryName: saeName, featureIndex: String(featureIndex) }}
        className="relative z-10 flex min-h-12 items-start gap-2 rounded-sm px-1 py-1 hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-sky-300"
        title={`${saeName} #${featureIndex}`}
      >
        {left && label}
        <span
          ref={(element) => {
            nodeRefs.current[key] = element
          }}
          className={`mt-0.5 size-3.5 shrink-0 rounded-full border border-slate-100/80 ${hook.includes('attn') ? 'bg-sky-500' : 'bg-orange-500'}`}
        />
        {!left && label}
      </Link>
    )
  }

  const renderLane = (
    items: { edge: GlobalWeightEdge; index: number }[],
    left: boolean,
    lane: number,
  ) => (
    <div
      key={`${left ? 'up' : 'down'}-lane-${lane}`}
      className="relative z-10 flex min-w-0 flex-col justify-center gap-6 py-12"
      style={{ paddingTop: 48 + lane * 24, paddingBottom: 96 - lane * 24 }}
    >
      {items.map(({ edge, index }) => renderNode(edge, index, left))}
    </div>
  )

  const lines = [
    ...upstream.map((edge, index) => ({
      edge,
      key: `up-${index}`,
      left: true,
    })),
    ...downstream.map((edge, index) => ({
      edge,
      key: `down-${index}`,
      left: false,
    })),
  ]
  const centerHook = upstream[0]?.target ?? downstream[0]?.source ?? ''

  return (
    <div className="border border-slate-800 bg-[#151820]">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-b border-white/10 px-4 py-2 text-xs text-slate-300">
        <span>Upstream {upstream.length}</span>
        <span>Downstream {downstream.length}</span>
        {result.mode === 'global' ? (
          <>
            <span className="inline-flex items-center gap-1.5">
              <span className="size-2 rounded-full bg-sky-400" />
              Positive
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="size-2 rounded-full bg-rose-400" />
              Negative
            </span>
          </>
        ) : (
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-amber-400" />
            Inhibitory
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <div
          ref={graphRef}
          className="relative grid min-h-[560px] min-w-[1560px] grid-cols-[repeat(3,minmax(0,1fr))_180px_repeat(3,minmax(0,1fr))] gap-4 px-5"
        >
          {geometry && (
            <svg
              className="pointer-events-none absolute inset-0"
              width={geometry.width}
              height={geometry.height}
              aria-hidden="true"
            >
              {lines.map(({ edge, key, left }) => {
                const point = geometry.nodes[key]
                if (!point) return null
                const source = left ? point : geometry.center
                const target = left ? geometry.center : point
                const strength =
                  Math.abs(edgeValue(edge, normalized)) / maxValue
                return (
                  <path
                    key={key}
                    d={`M ${source.x} ${source.y} L ${target.x} ${target.y}`}
                    fill="none"
                    stroke={edgeColor(edge)}
                    strokeWidth={0.75 + 2.75 * strength}
                    strokeOpacity={0.25 + 0.6 * strength}
                  />
                )
              })}
            </svg>
          )}
          {lanes(upstream, true).map((items, lane) =>
            renderLane(items, true, lane),
          )}
          <div className="relative z-10 flex min-w-0 flex-col items-center justify-center text-center">
            <span
              ref={centerRef}
              className="size-6 rounded-full border-[3px] border-slate-100 bg-sky-500 shadow-[0_0_0_3px_#60a5fa55]"
            />
            <Link
              to="/dictionaries/$dictionaryName/features/$featureIndex"
              params={{
                dictionaryName: result.target.saeName,
                featureIndex: String(result.target.featureIndex),
              }}
              className="mt-2 max-w-full break-all font-mono text-xs font-semibold text-white hover:underline"
              title={`${result.target.saeName} #${result.target.featureIndex}`}
            >
              {shortName(centerHook, result.target.featureIndex)}
            </Link>
            <span className="mt-1 max-w-full break-words text-xs leading-4 text-slate-200">
              {result.target.interpretation || missingInterpretationLabel}
            </span>
          </div>
          {lanes(downstream, false).map((items, lane) =>
            renderLane(items, false, lane),
          )}
        </div>
      </div>
    </div>
  )
}
