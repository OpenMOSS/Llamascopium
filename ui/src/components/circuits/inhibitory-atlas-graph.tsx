import { Link } from '@tanstack/react-router'
import * as d3 from 'd3'
import { Maximize2, Minus, Plus } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { GlobalWeightEdge, GlobalWeightResult } from '@/api/circuits'

type AtlasNode = d3.SimulationNodeDatum & {
  id: string
  hook: string
  saeName: string
  featureIndex: number
  interpretation?: string | null
  depth: number
}

type AtlasLink = d3.SimulationLinkDatum<AtlasNode> & {
  source: AtlasNode
  target: AtlasNode
  edge: GlobalWeightEdge
  value: number
}

const keyOf = (hook: string, feature: number) => `${hook}:${feature}`

function labelOf(node: AtlasNode) {
  const layer = node.hook.match(/^blocks\.(\d+)\./)?.[1] ?? '?'
  return `L${layer}${node.hook.includes('attn') ? 'A' : 'M'}#${node.featureIndex}`
}

export function InhibitoryAtlasGraph({
  result,
  normalized,
}: {
  result: GlobalWeightResult
  normalized: boolean
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [minimumPercent, setMinimumPercent] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const edges = result.connections ?? result.upstream
  const rootEdge = edges.find(
    (edge) =>
      edge.targetSaeName === result.target.saeName &&
      edge.targetFeature === result.target.featureIndex,
  )
  const rootId = rootEdge ? keyOf(rootEdge.target, rootEdge.targetFeature) : ''

  const graph = useMemo(() => {
    const nodes = new Map<string, AtlasNode>()
    const add = (
      hook: string,
      saeName: string,
      featureIndex: number,
      interpretation?: string | null,
    ) => {
      const id = keyOf(hook, featureIndex)
      const existing = nodes.get(id)
      if (existing) {
        if (!existing.interpretation) existing.interpretation = interpretation
      } else {
        nodes.set(id, {
          id,
          hook,
          saeName,
          featureIndex,
          interpretation,
          depth: 0,
        })
      }
    }
    for (const edge of edges) {
      add(
        edge.source,
        edge.sourceSaeName,
        edge.sourceFeature,
        edge.sourceInterpretation,
      )
      add(
        edge.target,
        edge.targetSaeName,
        edge.targetFeature,
        edge.targetInterpretation,
      )
    }
    const depths = new Map([[rootId, 0]])
    for (let level = 0; level < 10; level++) {
      let added = false
      for (const edge of edges) {
        if (depths.get(keyOf(edge.target, edge.targetFeature)) !== level)
          continue
        const source = keyOf(edge.source, edge.sourceFeature)
        if (!depths.has(source)) {
          depths.set(source, level + 1)
          added = true
        }
      }
      if (!added) break
    }
    for (const node of nodes.values()) node.depth = depths.get(node.id) ?? 0
    const maxValue = Math.max(
      0,
      ...edges.map((edge) =>
        normalized ? (edge.normalizedInhibitoryScore ?? 0) : (edge.score ?? 0),
      ),
    )
    return { nodes, maxValue }
  }, [edges, normalized, rootId])

  const threshold = (graph.maxValue * minimumPercent) / 100
  const visibleEdges = useMemo(
    () =>
      edges.filter(
        (edge) =>
          (normalized
            ? (edge.normalizedInhibitoryScore ?? 0)
            : (edge.score ?? 0)) >= threshold,
      ),
    [edges, normalized, threshold],
  )
  const visibleIds = new Set([rootId])
  for (const edge of visibleEdges) {
    visibleIds.add(keyOf(edge.source, edge.sourceFeature))
    visibleIds.add(keyOf(edge.target, edge.targetFeature))
  }
  const selectedNode = graph.nodes.get(selectedId ?? rootId)
  const activeId =
    selectedNode && visibleIds.has(selectedNode.id) ? selectedNode.id : rootId
  const activeNode = graph.nodes.get(activeId)
  const incidentEdges = visibleEdges.filter(
    (edge) =>
      keyOf(edge.source, edge.sourceFeature) === activeId ||
      keyOf(edge.target, edge.targetFeature) === activeId,
  )

  useEffect(() => {
    const element = svgRef.current
    if (!element) return
    const svg = d3.select(element)
    svg.selectAll('*').remove()
    const canvas = svg.append('g')
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.4, 3])
      .on('zoom', (event) =>
        canvas.attr('transform', event.transform.toString()),
      )
    zoomRef.current = zoom
    svg.call(zoom)

    const nodes = Array.from(visibleIds)
      .map((id) => graph.nodes.get(id))
      .filter((node): node is AtlasNode => node !== undefined)
      .map((node) => ({ ...node }))
    const byId = new Map(nodes.map((node) => [node.id, node]))
    const links: AtlasLink[] = visibleEdges.flatMap((edge) => {
      const source = byId.get(keyOf(edge.source, edge.sourceFeature))
      const target = byId.get(keyOf(edge.target, edge.targetFeature))
      if (!source || !target) return []
      return [
        {
          source,
          target,
          edge,
          value: normalized
            ? (edge.normalizedInhibitoryScore ?? 0)
            : (edge.score ?? 0),
        },
      ]
    })
    const linkGroup = canvas.append('g').attr('class', 'atlas-links')
    const link = linkGroup
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#e8a34d')
      .attr(
        'stroke-width',
        (item) => 1 + 4 * Math.sqrt(item.value / (graph.maxValue || 1)),
      )
      .attr('stroke-opacity', 0.7)
    link
      .append('title')
      .text(
        (item) =>
          `${labelOf(item.source)} → ${labelOf(item.target)}: ${item.value.toPrecision(3)}`,
      )

    const nodeGroup = canvas.append('g').attr('class', 'atlas-nodes')
    const node = nodeGroup
      .selectAll<SVGGElement, AtlasNode>('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer')
      .on('click', (event, item) => {
        event.stopPropagation()
        setSelectedId(item.id)
      })
    node
      .append('circle')
      .attr('r', (item) => (item.id === rootId ? 11 : 8))
      .attr('fill', (item) =>
        item.id === rootId
          ? '#20b6a4'
          : item.hook.includes('attn')
            ? '#38bdf8'
            : '#fb923c',
      )
      .attr('stroke', '#b8c3d1')
      .attr('stroke-width', 1.5)
    node
      .append('text')
      .attr('x', 14)
      .attr('y', 4)
      .attr('fill', '#f1f5f9')
      .attr('font-size', 12)
      .attr('font-family', 'monospace')
      .text(labelOf)
    node
      .append('title')
      .text(
        (item) =>
          `${item.saeName} #${item.featureIndex}${item.interpretation ? `\n${item.interpretation}` : ''}`,
      )

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink<AtlasNode, AtlasLink>(links)
          .id((item) => item.id)
          .distance(105)
          .strength(0.35),
      )
      .force('charge', d3.forceManyBody().strength(-260))
      .force('collision', d3.forceCollide<AtlasNode>().radius(38))
      .force(
        'x',
        d3.forceX<AtlasNode>((item) => 970 - item.depth * 270).strength(0.7),
      )
      .force('y', d3.forceY<AtlasNode>(340).strength(0.08))
      .on('tick', () => {
        link
          .attr('x1', (item) => item.source.x ?? 0)
          .attr('y1', (item) => item.source.y ?? 0)
          .attr('x2', (item) => item.target.x ?? 0)
          .attr('y2', (item) => item.target.y ?? 0)
        node.attr(
          'transform',
          (item) => `translate(${item.x ?? 0},${item.y ?? 0})`,
        )
      })
    node.call(
      d3
        .drag<SVGGElement, AtlasNode>()
        .on('start', (event, item) => {
          if (!event.active) simulation.alphaTarget(0.2).restart()
          item.fx = item.x
          item.fy = item.y
        })
        .on('drag', (event, item) => {
          item.fx = event.x
          item.fy = event.y
        })
        .on('end', (event, item) => {
          if (!event.active) simulation.alphaTarget(0)
          item.fx = null
          item.fy = null
        }),
    )
    return () => {
      simulation.stop()
      svg.on('.zoom', null)
    }
  }, [graph, visibleEdges, rootId, normalized])

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    const adjacent = new Set([activeId])
    svg
      .selectAll<SVGLineElement, AtlasLink>('.atlas-links line')
      .each((link) => {
        if (link.source.id === activeId) adjacent.add(link.target.id)
        if (link.target.id === activeId) adjacent.add(link.source.id)
      })
    svg
      .selectAll<SVGLineElement, AtlasLink>('.atlas-links line')
      .attr('stroke-opacity', (link) =>
        link.source.id === activeId || link.target.id === activeId ? 0.9 : 0.12,
      )
    svg
      .selectAll<SVGGElement, AtlasNode>('.atlas-nodes > g')
      .attr('opacity', (node) => (adjacent.has(node.id) ? 1 : 0.24))
      .select('circle')
      .attr('stroke', (node) => (node.id === activeId ? '#ffffff' : '#b8c3d1'))
      .attr('stroke-width', (node) => (node.id === activeId ? 3 : 1.5))
  }, [activeId, visibleEdges])

  const zoomBy = (factor: number) => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(200)
        .call(zoomRef.current.scaleBy, factor)
    }
  }

  return (
    <div className="border border-slate-800 bg-[#151820]">
      <div className="flex flex-wrap items-center gap-4 border-b border-white/10 px-4 py-2 text-xs text-slate-200">
        <span>
          {visibleIds.size} features · {visibleEdges.length} connections
        </span>
        <label className="flex items-center gap-2">
          <span>Min score</span>
          <input
            type="range"
            min={0}
            max={100}
            value={minimumPercent}
            onChange={(event) => setMinimumPercent(Number(event.target.value))}
            className="w-32 accent-amber-400"
            aria-label="Minimum connection score"
          />
          <span className="w-16 text-right tabular-nums">
            {threshold.toPrecision(2)}
          </span>
        </label>
        <div className="ml-auto flex gap-1">
          <button
            type="button"
            onClick={() => zoomBy(1.25)}
            title="Zoom in"
            aria-label="Zoom in"
            className="rounded p-1 hover:bg-white/10"
          >
            <Plus className="size-4" />
          </button>
          <button
            type="button"
            onClick={() => zoomBy(0.8)}
            title="Zoom out"
            aria-label="Zoom out"
            className="rounded p-1 hover:bg-white/10"
          >
            <Minus className="size-4" />
          </button>
          <button
            type="button"
            onClick={() =>
              svgRef.current &&
              zoomRef.current &&
              d3
                .select(svgRef.current)
                .transition()
                .duration(200)
                .call(zoomRef.current.transform, d3.zoomIdentity)
            }
            title="Reset view"
            aria-label="Reset view"
            className="rounded p-1 hover:bg-white/10"
          >
            <Maximize2 className="size-4" />
          </button>
        </div>
      </div>
      <div className="grid lg:grid-cols-[minmax(0,1fr)_250px]">
        <div className="min-w-0 overflow-x-auto">
          <svg
            ref={svgRef}
            viewBox="0 0 1100 680"
            className="h-[520px] min-w-[800px] w-full touch-none lg:h-[680px]"
            role="img"
            aria-label="Inhibitory global weight atlas"
          />
        </div>
        <div className="min-w-0 border-t border-white/10 p-4 text-slate-100 lg:border-l lg:border-t-0">
          {activeNode && (
            <>
              <div className="font-mono text-sm font-semibold">
                {labelOf(activeNode)}
              </div>
              <p className="mt-2 break-words text-xs leading-5 text-slate-300">
                {activeNode.interpretation || 'No interpretation available'}
              </p>
              <Link
                to="/dictionaries/$dictionaryName/features/$featureIndex"
                params={{
                  dictionaryName: activeNode.saeName,
                  featureIndex: String(activeNode.featureIndex),
                }}
                className="mt-3 block break-all text-xs text-sky-300 hover:underline"
              >
                {activeNode.saeName} #{activeNode.featureIndex}
              </Link>
              <div className="mt-5 border-t border-white/10 pt-3 text-xs text-slate-400">
                {incidentEdges.length} adjacent connections
              </div>
              <div className="mt-2 max-h-64 space-y-1 overflow-y-auto">
                {incidentEdges.map((edge, index) => {
                  const other =
                    keyOf(edge.source, edge.sourceFeature) === activeId
                      ? graph.nodes.get(keyOf(edge.target, edge.targetFeature))
                      : graph.nodes.get(keyOf(edge.source, edge.sourceFeature))
                  return (
                    other && (
                      <button
                        key={`${other.id}-${index}`}
                        type="button"
                        onClick={() => setSelectedId(other.id)}
                        className="flex w-full items-center justify-between gap-2 rounded-sm px-1 py-1 text-left text-xs hover:bg-white/10"
                      >
                        <span className="truncate font-mono">
                          {labelOf(other)}
                        </span>
                        <span className="shrink-0 tabular-nums text-amber-300">
                          {(normalized
                            ? edge.normalizedInhibitoryScore
                            : edge.score
                          )?.toPrecision(3)}
                        </span>
                      </button>
                    )
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
