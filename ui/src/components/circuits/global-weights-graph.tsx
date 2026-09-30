import type { GlobalWeightEdge, GlobalWeightResult } from '@/api/circuits'

function shortName(hook: string, feature: number) {
  const layer = hook.match(/^blocks\.(\d+)\./)?.[1] ?? '?'
  const type = hook.includes('attn') ? 'A' : 'M'
  return `L${layer} ${type}#${feature}`
}

function edgeValue(edge: GlobalWeightEdge, normalized: boolean) {
  return edge.kind === 'inhibitory'
    ? normalized
      ? (edge.normalizedInhibitoryScore ?? 0)
      : (edge.score ?? 0)
    : (edge.weight ?? 0)
}

function edgeColor(edge: GlobalWeightEdge) {
  if (edge.kind === 'inhibitory') return '#b45309'
  return (edge.weight ?? 0) < 0 ? '#dc5454' : '#168a80'
}

export function GlobalWeightsGraph({
  result,
  normalized,
}: {
  result: GlobalWeightResult
  normalized: boolean
}) {
  const upstream = result.upstream.slice(0, 12)
  const downstream = result.downstream.slice(0, 12)
  const count = Math.max(upstream.length, downstream.length, 1)
  const height = Math.max(250, count * 55 + 72)
  const centerY = height / 2
  const maxValue = Math.max(
    1e-8,
    ...[...upstream, ...downstream].map((edge) =>
      Math.abs(edgeValue(edge, normalized)),
    ),
  )
  const rowY = (index: number, length: number) =>
    centerY + (index - (length - 1) / 2) * 55

  const renderEdge = (
    edge: GlobalWeightEdge,
    index: number,
    side: 'upstream' | 'downstream',
  ) => {
    const left = side === 'upstream'
    const y = rowY(index, left ? upstream.length : downstream.length)
    const startX = left ? 225 : 535
    const endX = left ? 385 : 695
    const color = edgeColor(edge)
    const value = edgeValue(edge, normalized)
    const label = left
      ? shortName(edge.source, edge.sourceFeature)
      : shortName(edge.target, edge.targetFeature)
    const fullLabel = left
      ? `${edge.sourceSaeName} #${edge.sourceFeature}`
      : `${edge.targetSaeName} #${edge.targetFeature}`
    return (
      <g key={`${side}-${fullLabel}`}>
        <path
          d={
            left
              ? `M ${startX} ${y} C 295 ${y}, 315 ${centerY}, ${endX} ${centerY}`
              : `M ${startX} ${centerY} C 605 ${centerY}, 625 ${y}, ${endX} ${y}`
          }
          fill="none"
          stroke={color}
          strokeWidth={1.5 + (3 * Math.abs(value)) / maxValue}
          strokeOpacity="0.75"
          markerEnd={`url(#arrow-${edge.kind === 'inhibitory' ? 'inhibitory' : value < 0 ? 'negative' : 'positive'})`}
        />
        <rect
          x={left ? 45 : 695}
          y={y - 17}
          width="180"
          height="34"
          rx="4"
          fill="#fff"
          stroke="#cbd5e1"
        />
        <text
          x={left ? 135 : 785}
          y={y + 4}
          textAnchor="middle"
          fill="#334155"
          fontSize="13"
          fontFamily="monospace"
        >
          <title>{`${fullLabel}: ${value.toPrecision(4)}`}</title>
          {label}
        </text>
      </g>
    )
  }

  return (
    <div className="border rounded-md bg-white">
      <div className="flex flex-wrap gap-x-4 gap-y-1 border-b px-3 py-2 text-xs text-slate-600">
        {result.mode === 'global' ? (
          <>
            <span className="inline-flex items-center gap-1.5">
              <span className="size-2 rounded-full bg-teal-700" /> Positive
              weight
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="size-2 rounded-full bg-red-600" /> Negative
              weight
            </span>
          </>
        ) : (
          <span className="inline-flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-amber-700" /> Inhibitory
            score
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 920 ${height}`}
          width="920"
          height={height}
          role="img"
          aria-label="Directed feature weight graph"
        >
          <defs>
            {[
              ['positive', '#168a80'],
              ['negative', '#dc5454'],
              ['inhibitory', '#b45309'],
            ].map(([name, color]) => (
              <marker
                key={name}
                id={`arrow-${name}`}
                markerWidth="8"
                markerHeight="8"
                refX="7"
                refY="4"
                orient="auto"
              >
                <path d="M 0 0 L 8 4 L 0 8 z" fill={color} />
              </marker>
            ))}
          </defs>
          <text x="45" y="30" fill="#64748b" fontSize="12">
            UPSTREAM
          </text>
          <text x="385" y="30" fill="#64748b" fontSize="12">
            SELECTED FEATURE
          </text>
          {result.mode === 'global' && (
            <text x="695" y="30" fill="#64748b" fontSize="12">
              DOWNSTREAM
            </text>
          )}
          {upstream.map((edge, index) => renderEdge(edge, index, 'upstream'))}
          {downstream.map((edge, index) =>
            renderEdge(edge, index, 'downstream'),
          )}
          <rect
            x="385"
            y={centerY - 22}
            width="150"
            height="44"
            rx="4"
            fill="#253746"
          />
          <text
            x="460"
            y={centerY + 5}
            textAnchor="middle"
            fill="#fff"
            fontSize="13"
            fontFamily="monospace"
          >
            <title>{`${result.target.saeName} #${result.target.featureIndex}`}</title>
            {shortName(
              result.upstream[0]?.target ?? result.downstream[0]?.source ?? '',
              result.target.featureIndex,
            )}
          </text>
        </svg>
      </div>
      {(result.upstream.length > 12 || result.downstream.length > 12) && (
        <p className="border-t px-3 py-2 text-xs text-slate-500">
          Graph shows 12 connections per direction. The table includes all
          results.
        </p>
      )}
    </div>
  )
}
