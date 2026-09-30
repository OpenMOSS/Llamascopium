import { Link } from '@tanstack/react-router'
import { ArrowUpRight } from 'lucide-react'
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
    ? '#b45309'
    : (edge.weight ?? 0) < 0
      ? '#dc5454'
      : '#4e92d9'
}

export function GlobalWeightsGraph({
  result,
  normalized,
}: {
  result: GlobalWeightResult
  normalized: boolean
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
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1 text-xs font-medium text-slate-800">
          {shortName(hook, featureIndex)} <ArrowUpRight className="size-3" />
        </span>
        <span className="block break-words text-xs leading-5 text-slate-600">
          {interpretation || 'No interpretation available'}
        </span>
      </span>
    )
    return (
      <Link
        key={key}
        to="/dictionaries/$dictionaryName/features/$featureIndex"
        params={{ dictionaryName: saeName, featureIndex: String(featureIndex) }}
        className="relative z-10 flex min-h-16 items-start gap-2 py-2 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-sky-600"
        title={`${saeName} #${featureIndex}`}
      >
        {left && label}
        <span
          ref={(element) => {
            nodeRefs.current[key] = element
          }}
          className={`mt-1.5 size-3 shrink-0 rounded-full border-2 border-white shadow-[0_0_0_1px_#94a3b8] ${hook.includes('attn') ? 'bg-sky-500' : 'bg-amber-500'}`}
        />
        {!left && label}
      </Link>
    )
  }

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

  return (
    <div className="border border-slate-200 bg-white">
      <div className="flex flex-wrap gap-x-5 gap-y-1 border-b px-4 py-2 text-xs text-slate-600">
        {result.mode === 'global' ? (
          <>
            <span>Blue: positive weight</span>
            <span>Red: negative weight</span>
          </>
        ) : (
          <span>Amber: inhibitory score</span>
        )}
        <span>Line width: connection strength</span>
      </div>
      <div className="overflow-x-auto">
        <div
          ref={graphRef}
          className="relative grid min-w-[980px] grid-cols-[minmax(250px,1fr)_280px_minmax(250px,1fr)] gap-3 px-5 py-5"
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
                const middle = (source.x + target.x) / 2
                return (
                  <path
                    key={key}
                    d={`M ${source.x} ${source.y} C ${middle} ${source.y}, ${middle} ${target.y}, ${target.x} ${target.y}`}
                    fill="none"
                    stroke={edgeColor(edge)}
                    strokeWidth={
                      1 + (3 * Math.abs(edgeValue(edge, normalized))) / maxValue
                    }
                    strokeOpacity="0.55"
                  />
                )
              })}
            </svg>
          )}
          <div className="relative z-10 min-w-0">
            <h3 className="mb-3 text-xs font-semibold uppercase text-slate-500">
              Upstream · {upstream.length}
            </h3>
            <div className="divide-y divide-slate-100">
              {upstream.map((edge, index) => renderNode(edge, index, true))}
            </div>
          </div>
          <div className="relative z-10 flex min-h-32 flex-col items-center justify-center text-center">
            <span className="mb-2 text-xs font-semibold uppercase text-slate-500">
              Selected feature
            </span>
            <span
              ref={centerRef}
              className="size-6 rounded-full border-[3px] border-slate-800 bg-sky-500"
            />
            <Link
              to="/dictionaries/$dictionaryName/features/$featureIndex"
              params={{
                dictionaryName: result.target.saeName,
                featureIndex: String(result.target.featureIndex),
              }}
              className="mt-2 text-sm font-semibold text-sky-800 hover:underline"
            >
              {result.target.saeName} #{result.target.featureIndex}
            </Link>
            <span className="mt-1 max-w-full break-words text-xs leading-5 text-slate-600">
              {result.target.interpretation || 'No interpretation available'}
            </span>
          </div>
          <div className="relative z-10 min-w-0">
            <h3 className="mb-3 text-xs font-semibold uppercase text-slate-500">
              Downstream · {downstream.length}
            </h3>
            <div className="divide-y divide-slate-100">
              {downstream.map((edge, index) => renderNode(edge, index, false))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
