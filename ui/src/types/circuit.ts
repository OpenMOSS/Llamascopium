import { z } from 'zod'
import { FeatureSchema } from './feature'

export const QKTracingResultsSchema = z.object({
  pairWiseContributors: z.array(z.tuple([z.string(), z.string(), z.number()])),
  topQMarginalContributors: z.array(z.tuple([z.string(), z.number()])),
  topKMarginalContributors: z.array(z.tuple([z.string(), z.number()])),
})

export type QKTracingResults = z.infer<typeof QKTracingResultsSchema>

export const FeatureNodeSchema = z.object({
  featureType: z.enum(['lorsa', 'cross layer transcoder', 'matryoshka sae']),
  nodeId: z.string(),
  layer: z.number(),
  ctxIdx: z.number(),
  isTargetLogit: z.boolean(),
  saeName: z.string(),
  activation: z.number(),
  feature: FeatureSchema,
  qkTracingResults: QKTracingResultsSchema.nullish(),
  isFromQkTracing: z.boolean().default(false),
  matryoshkaFeatureRange: z.tuple([z.number(), z.number()]).nullish(),
})

export type FeatureNode = z.infer<typeof FeatureNodeSchema>

export const TokenNodeSchema = z.object({
  featureType: z.literal('embedding'),
  nodeId: z.string(),
  layer: z.number(),
  ctxIdx: z.number(),
  token: z.string(),
  isFromQkTracing: z.boolean().default(false),
})

export type TokenNode = z.infer<typeof TokenNodeSchema>

export const ErrorNodeSchema = z.object({
  featureType: z.enum([
    'lorsa error',
    'mlp reconstruction error',
    'residual reconstruction error',
  ]),
  nodeId: z.string(),
  layer: z.number(),
  ctxIdx: z.number(),
  isFromQkTracing: z.boolean().default(false),
})

export type ErrorNode = z.infer<typeof ErrorNodeSchema>

export const LogitNodeSchema = z.object({
  featureType: z.literal('logit'),
  nodeId: z.string(),
  layer: z.number(),
  ctxIdx: z.number(),
  tokenProb: z.number(),
  token: z.string(),
  isFromQkTracing: z.boolean().default(false),
})

export type LogitNode = z.infer<typeof LogitNodeSchema>

export const BiasNodeSchema = z.object({
  featureType: z.literal('bias'),
  nodeId: z.string(),
  layer: z.number(),
  ctxIdx: z.number(),
  isFromQkTracing: z.boolean().default(false),
})

export type BiasNode = z.infer<typeof BiasNodeSchema>

export const NodeSchema = z.discriminatedUnion('featureType', [
  FeatureNodeSchema,
  TokenNodeSchema,
  ErrorNodeSchema,
  LogitNodeSchema,
  BiasNodeSchema,
])

export type Node = z.infer<typeof NodeSchema>

export const EdgeSchema = z.object({
  source: z.string(),
  target: z.string(),
  weight: z.number(),
})

export type Edge = z.infer<typeof EdgeSchema>

export const PositionedNodeSchema = NodeSchema.and(
  z.object({
    pos: z.tuple([z.number(), z.number()]),
  }),
)

export type PositionedNode = z.infer<typeof PositionedNodeSchema>

export const PositionedEdgeSchema = EdgeSchema.and(
  z.object({
    pathStr: z.string(),
  }),
)

export type PositionedEdge = z.infer<typeof PositionedEdgeSchema>

export const CircuitMetadataSchema = z.object({
  promptTokens: z.array(z.string()),
})

export type CircuitMetadata = z.infer<typeof CircuitMetadataSchema>

export const CircuitDataSchema = z.object({
  nodes: z.array(NodeSchema),
  edges: z.array(EdgeSchema),
  metadata: CircuitMetadataSchema,
})

export type CircuitData = z.infer<typeof CircuitDataSchema>

export const QKNodeDataSchema = z.object({
  targetNode: NodeSchema,
  referencedNodes: z.array(NodeSchema),
})

export type QKNodeData = z.infer<typeof QKNodeDataSchema>

export const VisStateSchema = z.object({
  clickedId: z.string().nullable(),
  hoveredId: z.string().nullable(),
  selectedIds: z.array(z.string()).optional(),
})

export type VisState = z.infer<typeof VisStateSchema>
