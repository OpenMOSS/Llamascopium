import { z } from "zod";

export const TextTokenOriginSchema = z.object({
  key: z.literal("text"),
  range: z.tuple([z.number(), z.number()]),
});

export type TextTokenOrigin = z.infer<typeof TextTokenOriginSchema>;

export const ImageTokenOriginSchema = z.object({
  key: z.literal("image"),
  imageIndex: z.number(),
  rect: z.tuple([z.number(), z.number(), z.number(), z.number()]),
});

export type ImageTokenOrigin = z.infer<typeof ImageTokenOriginSchema>;

export const TokenOriginSchema = z.union([TextTokenOriginSchema, ImageTokenOriginSchema]);

export type TokenOrigin = z.infer<typeof TokenOriginSchema>;

export const FeatureSampleCompactSchema = z.object({
  text: z.string().nullish(),
  images: z.array(z.string()).nullish(),
  origins: z.array(TokenOriginSchema.nullable()),
  // COO format: indices where feature is active
  featureActsIndices: z.array(z.number()),
  // COO format: corresponding activation values
  featureActsValues: z.array(z.number()),
  // Z pattern data: indices in COO format for z pattern contributions
  zPatternIndices: z.array(z.array(z.number())).nullish(),
  // Z pattern data: corresponding contribution values
  zPatternValues: z.array(z.number()).nullish(),
});

export type FeatureSampleCompact = z.infer<typeof FeatureSampleCompactSchema>;

export const InterpretationSchema = z.object({
  text: z.string(),
  validation: z
    .array(
      z.object({
        method: z.string(),
        status: z.string().optional(),
        passed: z.boolean().optional(),
        rule: z.record(z.string(), z.any()).optional(),
        threshold: z.record(z.string(), z.any()).optional(),
        cases: z.record(z.string(), z.any()).optional(),
        selectionMetrics: z.record(z.string(), z.any()).optional(),
        detail: z
          .object({
            prompt: z.string(),
            response: z.any(),
          })
          .optional(),
      })
    )
    .default([]),
  detail: z
    .object({
      userPrompt: z.string(),
      systemPrompt: z.string(),
      response: z.object({
        steps: z.array(z.string()),
        finalExplanation: z.string(),
        complexity: z.number(),
        activationConsistency: z.number(),
      }),
    })
    .optional(),
  complexity: z.number().optional(),
  consistency: z.number().optional(),
  passed: z.boolean().optional(),
  time: z.any().optional(),
});

export type Interpretation = z.infer<typeof InterpretationSchema>;

export const FeatureSchema = z.object({
  featureIndex: z.number(),
  dictionaryName: z.string(),
  analysisName: z.string(),
  featureActivationHistogram: z.any().nullish(),
  decoderNorms: z.array(z.number()).nullish(),
  decoderSimilarityMatrix: z.array(z.array(z.number())).nullish(),
  decoderInnerProductMatrix: z.array(z.array(z.number())).nullish(),
  actTimes: z.number(),
  maxFeatureAct: z.number(),
  nAnalyzedTokens: z.number().nullish(),
  actTimesModalities: z.record(z.string(), z.number()).nullish(),
  maxFeatureActsModalities: z.record(z.string(), z.number()).nullish(),
  sampleGroups: z.array(
    z.object({
      analysisName: z.string(),
      samples: z.array(FeatureSampleCompactSchema),
    })
  ),
  logits: z
    .object({
      topPositive: z.array(
        z.object({
          logit: z.number(),
          token: z.string(),
        })
      ),
      topNegative: z.array(
        z.object({
          logit: z.number(),
          token: z.string(),
        })
      ),
      histogram: z.any(),
    })
    .nullish(),
  interpretation: InterpretationSchema.nullish(),
  isBookmarked: z.boolean().optional(),
});

export type Feature = z.infer<typeof FeatureSchema>;
