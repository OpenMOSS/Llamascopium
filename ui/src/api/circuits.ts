import { queryOptions } from '@tanstack/react-query'
import { createServerFn } from '@tanstack/react-start'
import camelcaseKeys from 'camelcase-keys'
import { z } from 'zod'
import type { CircuitData, QKNodeData } from '@/types/circuit'
import { parseWithPrettify } from '@/utils/zod'
import { CircuitDataSchema, QKNodeDataSchema } from '@/types/circuit'

export type CircuitStatus = 'pending' | 'running' | 'completed' | 'failed'

export type MatryoshkaFeatureRange = [number, number]

export interface CircuitFeatureTarget {
  saeName: string
  featureIndex: number
  position: number
}

export type LegacyCircuitFeatureTarget = [number, number, number, boolean]
export type CircuitFeatureTargetInput =
  | LegacyCircuitFeatureTarget
  | CircuitFeatureTarget

export interface CircuitConfig {
  desiredLogitProb: number
  maxFeatureNodes: number
  qkTracingTopk: number
  maxNLogits: number
  listOfFeatures?: CircuitFeatureTargetInput[]
  matryoshkaFeatureRange?: MatryoshkaFeatureRange | null
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface PlainTextInput {
  inputType: 'plain_text'
  text: string
}

export interface ChatTemplateInput {
  inputType: 'chat_template'
  messages: ChatMessage[]
}

export type CircuitInput = PlainTextInput | ChatTemplateInput

export interface CircuitListItem {
  id: string
  name: string | null
  group: string | null
  saeSetName: string
  saeSetSeries: string
  prompt: string
  input: CircuitInput
  config: CircuitConfig
  createdAt: string
  parentId?: string | null
  // Status tracking fields
  status: CircuitStatus
  progress: number
  progressPhase: string | null
  errorMessage: string | null
}

export interface CircuitStatusResponse {
  status: CircuitStatus
  progress: number
  progressPhase: string | null
  errorMessage: string | null
}

export interface GenerateCircuitParams {
  saeSetName: string
  input: CircuitInput
  name?: string
  group?: string
  desiredLogitProb?: number
  maxFeatureNodes?: number
  qkTracingTopk?: number
  maxNLogits?: number
  listOfFeatures?: CircuitFeatureTargetInput[]
  matryoshkaFeatureRange?: MatryoshkaFeatureRange | null
  parentId?: string
}

export interface GenerateCircuitResponse {
  circuitId: string
  status: CircuitStatus
  name: string | null
  group: string | null
  saeSetName: string
  prompt: string
  config: CircuitConfig
  input: CircuitInput
  createdAt: string
  parentId: string | null
}

export interface FetchCircuitParams {
  circuitId: string
  nodeThreshold?: number
  edgeThreshold?: number
}

export interface FetchCircuitResponse {
  circuitId: string
  name: string | null
  group: string | null
  saeSetName: string
  prompt: string
  input: CircuitInput
  config: CircuitConfig
  graphData: CircuitData
  createdAt: string
  parentId: string | null
  status: CircuitStatus
  nodeThreshold: number
  edgeThreshold: number
}

export const fetchSaeSets = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/sae-sets`)
    const data = await response.json()
    return parseWithPrettify(z.array(z.string()), data)
  },
)

export interface MatryoshkaRangeOptions {
  saeNames: string[]
  prefixes: MatryoshkaFeatureRange[]
  segments: MatryoshkaFeatureRange[]
}

export const fetchMatryoshkaRanges = createServerFn({ method: 'GET' })
  .inputValidator((data: { saeSetName: string }) => data)
  .handler(async ({ data: { saeSetName } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/sae-sets/${encodeURIComponent(saeSetName)}/matryoshka-ranges`,
    )
    if (!response.ok) {
      throw new Error(await response.text())
    }
    return camelcaseKeys(await response.json(), {
      deep: true,
    }) as MatryoshkaRangeOptions
  })

export const createSaeSet = createServerFn({ method: 'POST' })
  .inputValidator((data: { name: string; saeNames: string[] }) => data)
  .handler(async ({ data: { name, saeNames } }) => {
    const response = await fetch(`${process.env.BACKEND_URL}/sae-sets`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name,
        sae_names: saeNames,
      }),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || 'Failed to create SAE set')
    }

    return await response.json()
  })

export const previewInput = createServerFn({ method: 'POST' })
  .inputValidator((data: { saeSetName: string; input: CircuitInput }) => data)
  .handler(async ({ data: { saeSetName, input } }) => {
    // Convert input to backend format
    const backendInput =
      input.inputType === 'plain_text'
        ? { input_type: 'plain_text', text: input.text }
        : {
            input_type: 'chat_template',
            messages: input.messages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }

    const response = await fetch(`${process.env.BACKEND_URL}/preview`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sae_set_name: saeSetName,
        input: backendInput,
      }),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || 'Failed to preview input')
    }

    const result = await response.json()
    return camelcaseKeys(result, { deep: true }) as {
      prompt: string
      nextTokens: { token: string; prob: number }[]
    }
  })

export const fetchCircuits = createServerFn({ method: 'GET' }).handler(
  async () => {
    const response = await fetch(`${process.env.BACKEND_URL}/circuits`)
    if (!response.ok) {
      throw new Error(`Failed to fetch circuits: ${await response.text()}`)
    }
    const data = await response.json()
    return camelcaseKeys(data, { deep: true }) as CircuitListItem[]
  },
)

export const fetchCircuitStatus = createServerFn({ method: 'GET' })
  .inputValidator((data: { circuitId: string }) => data)
  .handler(async ({ data: { circuitId } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/circuits/${circuitId}/status`,
    )

    if (!response.ok) {
      throw new Error(
        `Failed to fetch circuit status: ${await response.text()}`,
      )
    }

    const data = await response.json()
    return camelcaseKeys(data, { deep: true }) as CircuitStatusResponse
  })

export const fetchCircuit = createServerFn({ method: 'GET' })
  .inputValidator((data: FetchCircuitParams) => data)
  .handler(
    async ({
      data: { circuitId, nodeThreshold = 0.8, edgeThreshold = 0.98 },
    }) => {
      const params = new URLSearchParams({
        node_threshold: nodeThreshold.toString(),
        edge_threshold: edgeThreshold.toString(),
      })

      const response = await fetch(
        `${process.env.BACKEND_URL}/circuits/${circuitId}?${params}`,
      )

      // Handle 202 status for in-progress circuits
      if (response.status === 202) {
        const status = response.headers.get('X-Circuit-Status') || 'pending'
        throw new Error(`Circuit is not ready (status: ${status})`)
      }

      if (!response.ok) {
        throw new Error(`Failed to fetch circuit: ${await response.text()}`)
      }

      const data = await response.json()

      const transformedData = {
        ...data,
        graphData: {
          ...data.graph_data,
          edges: data.graph_data.links,
          links: undefined,
        },
      }
      delete transformedData.graph_data

      const result = camelcaseKeys(transformedData, {
        deep: true,
      }) as FetchCircuitResponse

      result.graphData = parseWithPrettify(CircuitDataSchema, result.graphData)

      return result
    },
  )

export const generateCircuit = createServerFn({ method: 'POST' })
  .inputValidator((data: GenerateCircuitParams) => data)
  .handler(async ({ data }) => {
    const {
      saeSetName,
      input,
      name,
      group,
      desiredLogitProb,
      maxFeatureNodes,
      qkTracingTopk,
      maxNLogits,
      listOfFeatures,
      matryoshkaFeatureRange,
      parentId,
    } = data

    // Convert input to backend format
    const backendInput =
      input.inputType === 'plain_text'
        ? { input_type: 'plain_text', text: input.text }
        : {
            input_type: 'chat_template',
            messages: input.messages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }

    const response = await fetch(
      `${process.env.BACKEND_URL}/circuits?sae_set_name=${encodeURIComponent(saeSetName)}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: backendInput,
          name: name || null,
          group: group || null,
          desired_logit_prob: desiredLogitProb,
          max_feature_nodes: maxFeatureNodes,
          qk_tracing_topk: qkTracingTopk,
          max_n_logits: maxNLogits,
          list_of_features: listOfFeatures?.map((target) =>
            Array.isArray(target)
              ? target
              : {
                  sae_name: target.saeName,
                  feature_index: target.featureIndex,
                  position: target.position,
                },
          ),
          matryoshka_feature_range: matryoshkaFeatureRange ?? null,
          parent_id: parentId,
        }),
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to generate circuit: ${await response.text()}`)
    }

    const result = await response.json()

    // The response now only includes circuit metadata, not graph data
    const finalResult = camelcaseKeys(result, {
      deep: true,
    }) as GenerateCircuitResponse

    return finalResult
  })

export const deleteCircuit = createServerFn({ method: 'POST' })
  .inputValidator((data: { circuitId: string }) => data)
  .handler(async ({ data: { circuitId } }) => {
    const response = await fetch(
      `${process.env.BACKEND_URL}/circuits/${circuitId}`,
      {
        method: 'DELETE',
      },
    )

    if (!response.ok) {
      throw new Error(`Failed to delete circuit: ${await response.text()}`)
    }

    return await response.json()
  })

export const saeSetsQueryOptions = () =>
  queryOptions({
    queryKey: ['sae-sets'],
    queryFn: () => fetchSaeSets(),
  })

export const circuitsQueryOptions = () =>
  queryOptions({
    queryKey: ['circuits'],
    queryFn: () => fetchCircuits(),
    refetchInterval: (query) => {
      return query.state.data?.some(
        (c) => c.status === 'pending' || c.status === 'running',
      )
        ? 2000
        : false
    },
  })

export const circuitStatusQueryOptions = (circuitId: string) =>
  queryOptions({
    queryKey: ['circuit-status', circuitId],
    queryFn: () => fetchCircuitStatus({ data: { circuitId } }),
    enabled: !!circuitId,
    refetchInterval: (query) => {
      // Poll every 2 seconds if circuit is pending or running
      const status = query.state.data?.status
      if (status === 'pending' || status === 'running') {
        return 2000
      }
      return false
    },
  })

export const circuitQueryOptions = (
  circuitId: string,
  nodeThreshold: number = 0.8,
  edgeThreshold: number = 0.98,
) =>
  queryOptions({
    queryKey: ['circuit', circuitId, nodeThreshold, edgeThreshold],
    queryFn: () =>
      fetchCircuit({ data: { circuitId, nodeThreshold, edgeThreshold } }),
    staleTime: (query) => {
      return query.state.data?.status === 'completed' ? Infinity : 0
    },
    enabled: !!circuitId,
  })

export const fetchCircuitQKNode = createServerFn({ method: 'GET' })
  .inputValidator(
    (data: {
      circuitId: string
      nodeId: string
      nodeThreshold?: number
      edgeThreshold?: number
    }) => data,
  )
  .handler(
    async ({
      data: { circuitId, nodeId, nodeThreshold = 0.6, edgeThreshold = 0.8 },
    }) => {
      const params = new URLSearchParams({
        node_threshold: nodeThreshold.toString(),
        edge_threshold: edgeThreshold.toString(),
      })

      const response = await fetch(
        `${process.env.BACKEND_URL}/circuits/${circuitId}/qk/${encodeURIComponent(nodeId)}?${params}`,
      )

      if (response.status === 202) {
        const status = response.headers.get('X-Circuit-Status') || 'pending'
        throw new Error(`Circuit is not ready (status: ${status})`)
      }

      if (!response.ok) {
        throw new Error(
          `Failed to fetch QK node data: ${await response.text()}`,
        )
      }

      const data = await response.json()
      const result = camelcaseKeys(data, { deep: true }) as QKNodeData
      return parseWithPrettify(QKNodeDataSchema, result)
    },
  )

export const circuitQKNodeQueryOptions = (
  circuitId: string,
  nodeId: string,
  nodeThreshold: number = 0.6,
  edgeThreshold: number = 0.8,
) =>
  queryOptions({
    queryKey: [
      'circuit-qk-node',
      circuitId,
      nodeId,
      nodeThreshold,
      edgeThreshold,
    ],
    queryFn: () =>
      fetchCircuitQKNode({
        data: { circuitId, nodeId, nodeThreshold, edgeThreshold },
      }),
    staleTime: Infinity,
    enabled: !!circuitId && !!nodeId,
  })
