import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Feature } from "@/types/feature";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AppPagination } from "@/components/ui/pagination";
import { ChessBoard } from "@/components/chess/chess-board";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LinkGraphContainer } from "./link-graph-container";
import { transformCircuitData } from "./link-graph/utils";
import {
  annotateCircuitTaxonomyFeatures,
  annotateCircuitTaxonomyFeature,
  buildCircuitTaxonomyEvidenceExportUrl,
  CircuitTaxonomyCircuitDetail,
  CircuitTaxonomyCircuitSummary,
  CircuitTaxonomyDirectoryOption,
  CircuitTaxonomyFeatureRef,
  fetchCircuitTaxonomyCircuit,
  fetchCircuitTaxonomyCircuits,
  fetchCircuitTaxonomyDirectories,
  fetchCircuitTaxonomyResumeTarget,
  fetchCircuitTaxonomyReviewState,
  fetchFeatureByDictionaryName,
  resetCircuitTaxonomyReviewState,
  saveAllCircuitTaxonomyEvidence,
  saveCircuitTaxonomyReviewState,
  snapshotCircuitTaxonomyReviewState,
} from "@/utils/api";
import { normalizeZPattern } from "@/utils/activationUtils";
import { extractFenFromText, validateFen } from "@/utils/fenUtils";

const TAXONOMY_PREFIX_RE = /^\[(Det|Src|Tgt|Val|Reg|Cap|Pro|Mov|Tac|Spa|Uninterpretable)\]\s*/;
const TAXONOMY_TOP_ACTIVATION_PAGE_SIZE = 6;
const TAXONOMY_REVIEW_STORAGE_KEY = "circuit-taxonomy-review-proposals";

const getFeatureCacheKey = (featureRef: CircuitTaxonomyFeatureRef) =>
  `${featureRef.dictionary_name}:${featureRef.feature_index}`;

const extractTaxonomyPrefix = (text?: string | null) => {
  if (!text) {
    return "";
  }
  const match = text.match(TAXONOMY_PREFIX_RE);
  return match?.[1] ?? "";
};

type ChessTopActivationSample = {
  fen: string;
  activationStrength: number;
  activations: number[] | undefined;
  zPatternIndices?: number[][];
  zPatternValues?: number[];
  sampleIndex: number;
};

type CircuitTaxonomyReviewStatus = "pending" | "approved" | "rejected" | "error";

type CircuitTaxonomyReviewProposal = {
  id: string;
  directoryId?: string;
  fileName?: string;
  circuitIndex?: number | null;
  featureIndexInCircuit?: number | null;
  dictionaryName: string;
  featureIndex: number;
  layer?: number | null;
  featureType?: string | null;
  nodeId?: string | null;
  originalTaxonomy?: string;
  taxonomy: string;
  confidence?: number | null;
  rationale?: string;
  evidenceSummary?: string;
  reviewNote?: string;
  status?: CircuitTaxonomyReviewStatus;
  error?: string;
};

const normalizeReviewProposal = (raw: unknown, index: number): CircuitTaxonomyReviewProposal | null => {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const item = raw as Record<string, unknown>;
  const dictionaryName = String(item.dictionaryName ?? item.dictionary_name ?? "").trim();
  const rawFeatureIndex = item.featureIndex ?? item.feature_index;
  const featureIndex = typeof rawFeatureIndex === "number" ? rawFeatureIndex : Number(rawFeatureIndex);
  const taxonomy = String(item.taxonomy ?? item.label ?? "").replace(/^\[|\]$/g, "").trim();
  const originalTaxonomy = String(
    item.originalTaxonomy ?? item.original_taxonomy ?? item.taxonomy ?? item.label ?? "",
  ).replace(/^\[|\]$/g, "").trim();

  if (!dictionaryName || !Number.isFinite(featureIndex) || !taxonomy) {
    return null;
  }

  const id = String(item.id ?? `${dictionaryName}:${featureIndex}:${taxonomy}:${index}`);
  const rawFeatureIndexInCircuit = item.featureIndexInCircuit ?? item.feature_index_in_circuit;
  const featureIndexInCircuit =
    rawFeatureIndexInCircuit === undefined || rawFeatureIndexInCircuit === null
      ? null
      : Number(rawFeatureIndexInCircuit);

  return {
    id,
    directoryId: item.directoryId ? String(item.directoryId) : item.directory_id ? String(item.directory_id) : undefined,
    fileName: item.fileName ? String(item.fileName) : item.file_name ? String(item.file_name) : undefined,
    circuitIndex:
      item.circuitIndex === undefined && item.circuit_index === undefined
        ? null
        : Number(item.circuitIndex ?? item.circuit_index),
    featureIndexInCircuit: Number.isFinite(featureIndexInCircuit) ? featureIndexInCircuit : null,
    dictionaryName,
    featureIndex,
    layer: item.layer === undefined || item.layer === null ? null : Number(item.layer),
    featureType: item.featureType ? String(item.featureType) : item.feature_type ? String(item.feature_type) : null,
    nodeId: item.nodeId ? String(item.nodeId) : item.node_id ? String(item.node_id) : null,
    originalTaxonomy: originalTaxonomy || taxonomy,
    taxonomy,
    confidence:
      item.confidence === undefined || item.confidence === null ? null : Number(item.confidence),
    rationale: item.rationale ? String(item.rationale) : "",
    evidenceSummary: item.evidenceSummary ? String(item.evidenceSummary) : item.evidence_summary ? String(item.evidence_summary) : "",
    reviewNote: item.reviewNote ? String(item.reviewNote) : item.review_note ? String(item.review_note) : "",
    status:
      item.status === "approved" || item.status === "rejected" || item.status === "error"
        ? item.status
        : "pending",
    error: item.error ? String(item.error) : undefined,
  };
};

const parseReviewProposalImport = (
  rawText: string,
): { proposals: CircuitTaxonomyReviewProposal[]; activeReviewIndex?: number } => {
  const trimmed = rawText.trim();
  if (!trimmed) {
    return { proposals: [] };
  }

  let rawItems: unknown[];
  let activeReviewIndex: number | undefined;
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      rawItems = parsed;
    } else if (parsed && typeof parsed === "object" && Array.isArray((parsed as Record<string, unknown>).proposals)) {
      const parsedObject = parsed as Record<string, unknown>;
      rawItems = parsedObject.proposals as unknown[];
      const rawActiveIndex = parsedObject.activeReviewIndex ?? parsedObject.active_review_index;
      if (rawActiveIndex !== undefined && rawActiveIndex !== null) {
        const parsedActiveIndex = Number(rawActiveIndex);
        if (Number.isFinite(parsedActiveIndex)) {
          activeReviewIndex = parsedActiveIndex;
        }
      }
    } else {
      rawItems = [parsed];
    }
  } catch {
    rawItems = trimmed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  }

  const proposals = rawItems
    .map((item, index) => normalizeReviewProposal(item, index))
    .filter((item): item is CircuitTaxonomyReviewProposal => item !== null);

  return { proposals, activeReviewIndex };
};

const extractChessTopActivationSamples = (
  sampleGroup: Feature["sampleGroups"][0] | null | undefined,
): ChessTopActivationSample[] => {
  if (!sampleGroup) {
    return [];
  }

  const chessSamples: ChessTopActivationSample[] = [];

  sampleGroup.samples.forEach((sample, sampleIndex) => {
      const fen = extractFenFromText(sample.text ?? "");
      if (!fen || !validateFen(fen)) {
        return;
      }

      let activations: number[] | undefined;
      let activationStrength = 0;

      if (Array.isArray(sample.featureActsIndices) && Array.isArray(sample.featureActsValues)) {
        activations = new Array(64).fill(0);
        for (
          let idx = 0;
          idx < Math.min(sample.featureActsIndices.length, sample.featureActsValues.length);
          idx += 1
        ) {
          const boardIndex = sample.featureActsIndices[idx];
          const value = sample.featureActsValues[idx];

          if (boardIndex >= 0 && boardIndex < 64) {
            activations[boardIndex] = value;
            if (Math.abs(value) > Math.abs(activationStrength)) {
              activationStrength = value;
            }
          }
        }
      }

      chessSamples.push({
        fen,
        activationStrength,
        activations,
        ...normalizeZPattern(sample.zPatternIndices, sample.zPatternValues),
        sampleIndex,
      });
    });

  return chessSamples.sort((a, b) => Math.abs(b.activationStrength) - Math.abs(a.activationStrength));
};

const CircuitTaxonomyTopActivationBoards = ({
  sampleGroup,
}: {
  sampleGroup: Feature["sampleGroups"][0];
}) => {
  const [page, setPage] = useState(1);

  const chessSamples = useMemo(() => extractChessTopActivationSamples(sampleGroup), [sampleGroup]);
  const maxPage = Math.max(1, Math.ceil(chessSamples.length / TAXONOMY_TOP_ACTIVATION_PAGE_SIZE));
  const currentSamples = useMemo(
    () =>
      chessSamples.slice(
        (page - 1) * TAXONOMY_TOP_ACTIVATION_PAGE_SIZE,
        page * TAXONOMY_TOP_ACTIVATION_PAGE_SIZE,
      ),
    [chessSamples, page],
  );

  useEffect(() => {
    setPage(1);
  }, [sampleGroup]);

  if (chessSamples.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No activation samples containing chessboard were found for this feature.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Showing {chessSamples.length} top activation samples as chess boards.
      </p>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {currentSamples.map((sample, index) => (
          <div
            key={`${sample.sampleIndex}-${sample.fen}`}
            className="rounded-lg border bg-background p-3"
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                Sample #{(page - 1) * TAXONOMY_TOP_ACTIVATION_PAGE_SIZE + index + 1}
              </span>
              <span className="text-xs text-muted-foreground">
                Max act: {sample.activationStrength.toFixed(3)}
              </span>
            </div>

            <ChessBoard
              fen={sample.fen}
              size="small"
              showCoordinates
              activations={sample.activations}
              zPatternIndices={sample.zPatternIndices}
              zPatternValues={sample.zPatternValues}
              sampleIndex={sample.sampleIndex}
              analysisName="Taxonomy Top Activation"
              flip_activation={sample.fen.includes(" b ")}
              autoFlipWhenBlack
            />
          </div>
        ))}
      </div>

      {maxPage > 1 && <AppPagination page={page} setPage={setPage} maxPage={maxPage} />}
    </div>
  );
};

type EnsureFeatureLoadOptions = {
  syncState?: boolean;
  forceRefresh?: boolean;
};

export const CircuitTaxonomyAnnotation = () => {
  const [directories, setDirectories] = useState<CircuitTaxonomyDirectoryOption[]>([]);
  const [taxonomyLabels, setTaxonomyLabels] = useState<string[]>([]);
  const [selectedDirectoryId, setSelectedDirectoryId] = useState<string>("");
  const [circuits, setCircuits] = useState<CircuitTaxonomyCircuitSummary[]>([]);
  const [selectedCircuitFile, setSelectedCircuitFile] = useState<string>("");
  const [circuitDetail, setCircuitDetail] = useState<CircuitTaxonomyCircuitDetail | null>(null);
  const [currentFeatureIndex, setCurrentFeatureIndex] = useState(0);
  const [selectedTaxonomy, setSelectedTaxonomy] = useState("");
  const [featureCache, setFeatureCache] = useState<Record<string, Feature>>({});
  const [clickedId, setClickedId] = useState<string | null>(null);
  const [loadingDirectories, setLoadingDirectories] = useState(false);
  const [loadingCircuits, setLoadingCircuits] = useState(false);
  const [loadingCircuitDetail, setLoadingCircuitDetail] = useState(false);
  const [saving, setSaving] = useState(false);
  const [jumpingToPending, setJumpingToPending] = useState(false);
  const [exportingEvidence, setExportingEvidence] = useState(false);
  const [exportingDirectoryEvidence, setExportingDirectoryEvidence] = useState(false);
  const [savingAllCircuitEvidence, setSavingAllCircuitEvidence] = useState(false);
  const [savedDirectoryEvidencePath, setSavedDirectoryEvidencePath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviewProposals, setReviewProposals] = useState<CircuitTaxonomyReviewProposal[]>([]);
  const [reviewImportText, setReviewImportText] = useState("");
  const [activeReviewIndex, setActiveReviewIndex] = useState(0);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewSaving, setReviewSaving] = useState(false);
  const [reviewBatchSaving, setReviewBatchSaving] = useState(false);
  const [reviewSnapshotSaving, setReviewSnapshotSaving] = useState(false);
  const [reviewResetting, setReviewResetting] = useState(false);
  const [reviewStateMessage, setReviewStateMessage] = useState<string | null>(null);
  const [reviewSnapshotAwaitingChoice, setReviewSnapshotAwaitingChoice] = useState(false);

  const featureCacheRef = useRef<Record<string, Feature>>({});
  const pendingFeatureLoads = useRef<Map<string, Promise<Feature | null>>>(new Map());
  const pendingResumeFeatureIndexRef = useRef<number | null>(null);
  const reviewStateLoadedRef = useRef(false);
  const reviewStateSaveTimerRef = useRef<number | null>(null);

  const persistReviewStateNow = useCallback(
    async (proposals: CircuitTaxonomyReviewProposal[], activeIndex: number) => {
      try {
        const response = await saveCircuitTaxonomyReviewState(proposals, activeIndex);
        setReviewStateMessage(`Saved ${response.proposal_count ?? proposals.length} review items to backend.`);
      } catch (saveError) {
        setReviewStateMessage(saveError instanceof Error ? `Backend review save failed: ${saveError.message}` : "Backend review save failed.");
      }
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;

    const loadReviewState = async () => {
      try {
        const response = await fetchCircuitTaxonomyReviewState();
        if (cancelled) {
          return;
        }

        if (response.status === "loaded" && Array.isArray(response.proposals) && response.proposals.length > 0) {
          const proposals = response.proposals
            .map((item, index) => normalizeReviewProposal(item, index))
            .filter((item): item is CircuitTaxonomyReviewProposal => item !== null);
          setReviewProposals(proposals);
          setActiveReviewIndex(Math.min(Math.max(response.active_review_index ?? 0, 0), Math.max(proposals.length - 1, 0)));
          setReviewStateMessage(`Restored ${proposals.length} review items from backend.`);
          reviewStateLoadedRef.current = true;
          return;
        }
      } catch (loadError) {
        if (!cancelled) {
          setReviewStateMessage(loadError instanceof Error ? `Backend review restore failed: ${loadError.message}` : "Backend review restore failed.");
        }
      }

      const stored = window.localStorage.getItem(TAXONOMY_REVIEW_STORAGE_KEY);
      if (!stored || cancelled) {
        reviewStateLoadedRef.current = true;
        return;
      }

      try {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          const proposals = parsed
            .map((item, index) => normalizeReviewProposal(item, index))
            .filter((item): item is CircuitTaxonomyReviewProposal => item !== null);
          setReviewProposals(proposals);
          setReviewStateMessage(`Restored ${proposals.length} review items from localStorage.`);
        }
      } catch {
        window.localStorage.removeItem(TAXONOMY_REVIEW_STORAGE_KEY);
      } finally {
        reviewStateLoadedRef.current = true;
      }
    };

    void loadReviewState();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(TAXONOMY_REVIEW_STORAGE_KEY, JSON.stringify(reviewProposals));

    if (!reviewStateLoadedRef.current) {
      return;
    }

    if (reviewStateSaveTimerRef.current) {
      window.clearTimeout(reviewStateSaveTimerRef.current);
    }

    reviewStateSaveTimerRef.current = window.setTimeout(() => {
      void persistReviewStateNow(reviewProposals, activeReviewIndex);
    }, 600);

    return () => {
      if (reviewStateSaveTimerRef.current) {
        window.clearTimeout(reviewStateSaveTimerRef.current);
      }
    };
  }, [activeReviewIndex, persistReviewStateNow, reviewProposals]);

  useEffect(() => {
    let cancelled = false;
    const loadDirectories = async () => {
      setLoadingDirectories(true);
      setError(null);
      try {
        const response = await fetchCircuitTaxonomyDirectories();
        if (cancelled) {
          return;
        }
        setDirectories(response.directories);
        setTaxonomyLabels(response.taxonomy_labels);
        if (!selectedDirectoryId && response.directories.length > 0) {
          setSelectedDirectoryId(response.directories[0].id);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load circuit directories");
        }
      } finally {
        if (!cancelled) {
          setLoadingDirectories(false);
        }
      }
    };

    loadDirectories();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedDirectoryId) {
      return;
    }

    let cancelled = false;
    const loadCircuits = async () => {
      setLoadingCircuits(true);
      setError(null);
      setCircuitDetail(null);
      setFeatureCache({});
      featureCacheRef.current = {};
      setSelectedCircuitFile("");
      setCurrentFeatureIndex(0);
      try {
        const response = await fetchCircuitTaxonomyCircuits(selectedDirectoryId);
        if (cancelled) {
          return;
        }
        setCircuits(response.circuits);
        if (response.circuits.length > 0) {
          const resumeTarget = await fetchCircuitTaxonomyResumeTarget(selectedDirectoryId);
          if (cancelled) {
            return;
          }
          const resumedFileName =
            !resumeTarget.completed && resumeTarget.file_name
              ? resumeTarget.file_name
              : response.circuits[0].file_name;
          pendingResumeFeatureIndexRef.current = resumeTarget.feature_index ?? null;
          setSelectedCircuitFile(resumedFileName);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load circuit list");
          setCircuits([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingCircuits(false);
        }
      }
    };

    loadCircuits();
    return () => {
      cancelled = true;
    };
  }, [selectedDirectoryId]);

  const currentFeatureRef = useMemo(() => circuitDetail?.features[currentFeatureIndex] ?? null, [circuitDetail, currentFeatureIndex]);
  const activeReviewProposal = reviewProposals[activeReviewIndex] ?? null;
  const reviewCounts = useMemo(
    () => ({
      pending: reviewProposals.filter((item) => (item.status ?? "pending") === "pending").length,
      approved: reviewProposals.filter((item) => item.status === "approved").length,
      error: reviewProposals.filter((item) => item.status === "error").length,
      edited: reviewProposals.filter(
        (item) => item.originalTaxonomy && item.taxonomy && item.originalTaxonomy !== item.taxonomy,
      ).length,
    }),
    [reviewProposals],
  );

  const storeFeatureInCache = useCallback(
    (
      featureRef: CircuitTaxonomyCircuitDetail["features"][number],
      feature: Feature,
      options: EnsureFeatureLoadOptions = {},
    ) => {
      const cacheKey = getFeatureCacheKey(featureRef);
      featureCacheRef.current = {
        ...featureCacheRef.current,
        [cacheKey]: feature,
      };

      if (options.syncState !== false) {
        setFeatureCache((prev) => ({
          ...prev,
          [cacheKey]: feature,
        }));
      }
    },
    [],
  );

  const ensureFeatureLoaded = useCallback(
    async (
      featureRef: CircuitTaxonomyCircuitDetail["features"][number] | null,
      options: EnsureFeatureLoadOptions = {},
    ) => {
      if (!featureRef) {
        return null;
      }

      const cacheKey = getFeatureCacheKey(featureRef);
      const cachedFeature = featureCacheRef.current[cacheKey];
      if (cachedFeature && !options.forceRefresh) {
        if (options.syncState !== false) {
          setFeatureCache((prev) => ({
            ...prev,
            ...(prev[cacheKey] ? {} : { [cacheKey]: cachedFeature }),
          }));
        }
        return cachedFeature;
      }

      if (!options.forceRefresh && pendingFeatureLoads.current.has(cacheKey)) {
        return pendingFeatureLoads.current.get(cacheKey) ?? null;
      }

      const request = fetchFeatureByDictionaryName(
        featureRef.dictionary_name,
        featureRef.feature_index,
        { forceRefresh: options.forceRefresh },
      )
        .then((feature) => {
          if (feature) {
            storeFeatureInCache(featureRef, feature, options);
          }
          return feature;
        })
        .finally(() => {
          pendingFeatureLoads.current.delete(cacheKey);
        });

      pendingFeatureLoads.current.set(cacheKey, request);
      return request;
    },
    [storeFeatureInCache],
  );

  const refreshFeature = useCallback(async (featureRef: CircuitTaxonomyFeatureRef | null) => {
    if (!featureRef) {
      return null;
    }
    const refreshed = await ensureFeatureLoaded(featureRef, { forceRefresh: true });
    return refreshed;
  }, [ensureFeatureLoaded]);

  useEffect(() => {
    if (!selectedDirectoryId || !selectedCircuitFile) {
      return;
    }

    let cancelled = false;
    const loadCircuitDetail = async () => {
      setLoadingCircuitDetail(true);
      setError(null);
      try {
        const detail = await fetchCircuitTaxonomyCircuit(selectedDirectoryId, selectedCircuitFile);
        if (cancelled) {
          return;
        }

        const targetFeatureIndex = Math.min(
          Math.max(
            pendingResumeFeatureIndexRef.current
              ?? detail.first_unannotated_feature_index
              ?? 0,
            0,
          ),
          Math.max(detail.total_features - 1, 0),
        );
        pendingResumeFeatureIndexRef.current = null;
        const targetFeatureRef = detail.features[targetFeatureIndex] ?? null;

        await ensureFeatureLoaded(targetFeatureRef);

        if (cancelled) {
          return;
        }

        setCircuitDetail(detail);
        setCurrentFeatureIndex(targetFeatureIndex);
        setClickedId(targetFeatureRef?.node_id ?? null);
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load circuit detail");
        }
      } finally {
        if (!cancelled) {
          setLoadingCircuitDetail(false);
        }
      }
    };

    loadCircuitDetail();
    return () => {
      cancelled = true;
    };
  }, [ensureFeatureLoaded, selectedDirectoryId, selectedCircuitFile]);

  useEffect(() => {
    const prefetch = async () => {
      if (!circuitDetail) {
        return;
      }
      for (let offset = 0; offset < 3; offset += 1) {
        const featureRef = circuitDetail.features[currentFeatureIndex + offset];
        if (!featureRef) {
          continue;
        }
        void ensureFeatureLoaded(featureRef);
      }
    };
    void prefetch();
  }, [circuitDetail, currentFeatureIndex, ensureFeatureLoaded]);

  const currentFeature = currentFeatureRef ? featureCache[getFeatureCacheKey(currentFeatureRef)] ?? null : null;
  const currentFeatureMatchesActiveReview =
    !!activeReviewProposal &&
    !!currentFeatureRef &&
    currentFeatureRef.dictionary_name === activeReviewProposal.dictionaryName &&
    currentFeatureRef.feature_index === activeReviewProposal.featureIndex;

  useEffect(() => {
    setSelectedTaxonomy(extractTaxonomyPrefix(currentFeature?.interpretation?.text));
  }, [currentFeature?.dictionaryName, currentFeature?.featureIndex, currentFeature?.interpretation?.text]);

  useEffect(() => {
    setClickedId(currentFeatureRef?.node_id ?? null);
  }, [currentFeatureRef?.node_id]);

  const graphData = useMemo(() => {
    if (!circuitDetail) {
      return null;
    }
    return transformCircuitData(circuitDetail.graph_data as any);
  }, [circuitDetail]);

  const topActivationGroup = useMemo(() => {
    if (!currentFeature) {
      return null;
    }
    return (
      currentFeature.sampleGroups.find((group) => group.analysisName === "top_activations") ??
      currentFeature.sampleGroups[0] ??
      null
    );
  }, [currentFeature]);

  const goToFeatureIndex = useCallback((nextIndex: number) => {
    if (!circuitDetail) {
      return;
    }
    const clamped = Math.min(Math.max(nextIndex, 0), Math.max(circuitDetail.total_features - 1, 0));
    setCurrentFeatureIndex(clamped);
  }, [circuitDetail]);

  const goToNextFeature = useCallback(async () => {
    if (!circuitDetail || !selectedDirectoryId || !selectedCircuitFile) {
      return;
    }
    const resumeTarget = await fetchCircuitTaxonomyResumeTarget(
      selectedDirectoryId,
      selectedCircuitFile,
      currentFeatureIndex + 1,
    );
    if (resumeTarget.completed || resumeTarget.file_name === null || resumeTarget.feature_index === null) {
      return;
    }

    if (resumeTarget.file_name === selectedCircuitFile && circuitDetail.file_name === selectedCircuitFile) {
      setCurrentFeatureIndex(resumeTarget.feature_index);
      setClickedId(circuitDetail.features[resumeTarget.feature_index]?.node_id ?? null);
      return;
    }

    pendingResumeFeatureIndexRef.current = resumeTarget.feature_index;
    setSelectedCircuitFile(resumeTarget.file_name);
  }, [circuitDetail, currentFeatureIndex, selectedCircuitFile, selectedDirectoryId]);

  const jumpToCurrentFilePendingFeature = useCallback(async () => {
    if (!circuitDetail || !selectedDirectoryId || !selectedCircuitFile) {
      return;
    }

    setJumpingToPending(true);
    setError(null);
    try {
      const resumeTarget = await fetchCircuitTaxonomyResumeTarget(
        selectedDirectoryId,
        selectedCircuitFile,
        0,
      );

      if (resumeTarget.completed || resumeTarget.feature_index === null) {
        window.alert("This file has no remaining unlabeled features.");
        return;
      }

      if (resumeTarget.file_name !== selectedCircuitFile) {
        return;
      }

      setCurrentFeatureIndex(resumeTarget.feature_index);
      setClickedId(circuitDetail.features[resumeTarget.feature_index]?.node_id ?? null);
    } catch (jumpError) {
      setError(jumpError instanceof Error ? jumpError.message : "Failed to jump to pending feature");
    } finally {
      setJumpingToPending(false);
    }
  }, [circuitDetail, selectedCircuitFile, selectedDirectoryId]);

  const handleGraphFeatureSelect = useCallback((feature: Feature | null) => {
    if (!feature || !circuitDetail) {
      return;
    }

    const matchedIndex = circuitDetail.features.findIndex(
      (item) =>
        item.dictionary_name === feature.dictionaryName && item.feature_index === feature.featureIndex,
    );
    if (matchedIndex >= 0) {
      setFeatureCache((prev) => ({
        ...prev,
        [`${feature.dictionaryName}:${feature.featureIndex}`]: feature,
      }));
      setCurrentFeatureIndex(matchedIndex);
    }
  }, [circuitDetail]);

  const handleSaveAndNext = useCallback(async () => {
    if (!currentFeatureRef) {
      return;
    }
    if (!selectedTaxonomy) {
      window.alert("Please select a taxonomy label first.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      let response = await annotateCircuitTaxonomyFeature(
        currentFeatureRef.dictionary_name,
        currentFeatureRef.feature_index,
        selectedTaxonomy,
      );

      if (response.status === "conflict") {
        const confirmed = window.confirm(
          `Current interpretation starts with [${response.existing_taxonomy}]. Replace it with [${selectedTaxonomy}]?`,
        );
        if (!confirmed) {
          return;
        }
        response = await annotateCircuitTaxonomyFeature(
          currentFeatureRef.dictionary_name,
          currentFeatureRef.feature_index,
          selectedTaxonomy,
          true,
        );
      }

      if (response.status === "updated") {
        await refreshFeature(currentFeatureRef);
      }

      await goToNextFeature();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save taxonomy label");
    } finally {
      setSaving(false);
    }
  }, [currentFeatureRef, goToNextFeature, refreshFeature, selectedTaxonomy]);

  const updateReviewProposal = useCallback(
    (proposalId: string, update: Partial<CircuitTaxonomyReviewProposal>) => {
      setReviewProposals((prev) =>
        prev.map((proposal) => (proposal.id === proposalId ? { ...proposal, ...update } : proposal)),
      );
    },
    [],
  );

  const moveToNextPendingReview = useCallback(
    (fromIndex: number) => {
      const nextIndex = reviewProposals.findIndex(
        (proposal, index) => index > fromIndex && (proposal.status ?? "pending") === "pending",
      );
      if (nextIndex >= 0) {
        setActiveReviewIndex(nextIndex);
        return { proposal: reviewProposals[nextIndex], index: nextIndex };
      }
      return null;
    },
    [reviewProposals],
  );

  const handleImportReviewProposals = useCallback(() => {
    setReviewError(null);
    try {
      const parsed = parseReviewProposalImport(reviewImportText);
      if (parsed.proposals.length === 0) {
        setReviewError("No valid proposals were found.");
        return;
      }

      setReviewProposals((prev) => {
        const seen = new Set(prev.map((proposal) => proposal.id));
        const next = [...prev];
        for (const proposal of parsed.proposals) {
          if (!seen.has(proposal.id)) {
            next.push(proposal);
            seen.add(proposal.id);
          }
        }
        return next;
      });
      setReviewSnapshotAwaitingChoice(false);
      if (parsed.activeReviewIndex !== undefined) {
        setActiveReviewIndex(Math.min(Math.max(parsed.activeReviewIndex, 0), Math.max(parsed.proposals.length - 1, 0)));
      } else {
        setActiveReviewIndex((prev) => (reviewProposals.length === 0 ? 0 : prev));
      }
      setReviewImportText("");
    } catch (importError) {
      setReviewError(importError instanceof Error ? importError.message : "Failed to parse review proposals.");
    }
  }, [reviewImportText, reviewProposals.length]);

  const handleImportReviewProposalFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) {
        return;
      }

      setReviewError(null);
      try {
        const parsedBatches = await Promise.all(
          Array.from(files).map(async (file) => parseReviewProposalImport(await file.text())),
        );
        const imported = parsedBatches.flatMap((batch) => batch.proposals);
        if (imported.length === 0) {
          setReviewError("No valid proposals were found in the selected files.");
          return;
        }

        setReviewProposals((prev) => {
          const seen = new Set(prev.map((proposal) => proposal.id));
          const next = [...prev];
          for (const proposal of imported) {
            if (!seen.has(proposal.id)) {
              next.push(proposal);
              seen.add(proposal.id);
            }
          }
          return next;
        });
        setActiveReviewIndex((prev) => (reviewProposals.length === 0 ? 0 : prev));
        setReviewSnapshotAwaitingChoice(false);
        setReviewStateMessage(`Imported ${imported.length} review items from ${files.length} file(s).`);
      } catch (importError) {
        setReviewError(importError instanceof Error ? importError.message : "Failed to parse review proposal files.");
      }
    },
    [reviewProposals.length],
  );

  const handleSnapshotReviewState = useCallback(async () => {
    if (reviewStateSaveTimerRef.current) {
      window.clearTimeout(reviewStateSaveTimerRef.current);
      reviewStateSaveTimerRef.current = null;
    }

    setReviewSnapshotSaving(true);
    setReviewError(null);
    try {
      const response = await snapshotCircuitTaxonomyReviewState(reviewProposals, activeReviewIndex);
      setReviewStateMessage(
        `Saved permanent review snapshot (${response.proposal_count ?? reviewProposals.length} items) to ${
          response.snapshot_relative_path ?? response.relative_path ?? "backend"
        }. Choose whether to continue from this review-state or start a blank one.`,
      );
      setReviewSnapshotAwaitingChoice(true);
    } catch (snapshotError) {
      setReviewError(snapshotError instanceof Error ? snapshotError.message : "Failed to save review snapshot.");
    } finally {
      setReviewSnapshotSaving(false);
    }
  }, [activeReviewIndex, reviewProposals]);

  const handleContinueReviewState = useCallback(() => {
    setReviewSnapshotAwaitingChoice(false);
    setReviewStateMessage(`Continuing from the current review-state copy (${reviewProposals.length} items).`);
  }, [reviewProposals.length]);

  const handleStartBlankReviewState = useCallback(async () => {
    if (reviewStateSaveTimerRef.current) {
      window.clearTimeout(reviewStateSaveTimerRef.current);
      reviewStateSaveTimerRef.current = null;
    }

    setReviewResetting(true);
    setReviewError(null);
    try {
      const response = await resetCircuitTaxonomyReviewState();
      setReviewProposals([]);
      setActiveReviewIndex(0);
      window.localStorage.removeItem(TAXONOMY_REVIEW_STORAGE_KEY);
      setReviewSnapshotAwaitingChoice(false);
      setReviewStateMessage(
        `Started a blank review-state at ${response.relative_path ?? "backend"}. Import a new LLM proposal to continue.`,
      );
    } catch (resetError) {
      setReviewError(resetError instanceof Error ? resetError.message : "Failed to reset review state.");
    } finally {
      setReviewResetting(false);
    }
  }, []);

  const handleAddCurrentFeatureToReview = useCallback(() => {
    if (!currentFeatureRef || !circuitDetail) {
      return;
    }

    const proposal: CircuitTaxonomyReviewProposal = {
      id: `${currentFeatureRef.dictionary_name}:${currentFeatureRef.feature_index}:manual`,
      directoryId: selectedDirectoryId,
      fileName: selectedCircuitFile,
      circuitIndex: circuitDetail.circuit_index,
      featureIndexInCircuit: currentFeatureIndex,
      dictionaryName: currentFeatureRef.dictionary_name,
      featureIndex: currentFeatureRef.feature_index,
      layer: currentFeatureRef.layer,
      featureType: currentFeatureRef.feature_type,
      nodeId: currentFeatureRef.node_id,
      originalTaxonomy: selectedTaxonomy || extractTaxonomyPrefix(currentFeature?.interpretation?.text) || "Uninterpretable",
      taxonomy: selectedTaxonomy || extractTaxonomyPrefix(currentFeature?.interpretation?.text) || "Uninterpretable",
      confidence: null,
      rationale: "Manual review item created from the current feature.",
      evidenceSummary: currentFeature?.interpretation?.text || "",
      status: "pending",
    };

    setReviewProposals((prev) => {
      if (prev.some((item) => item.id === proposal.id)) {
        return prev;
      }
      return [...prev, proposal];
    });
    setReviewSnapshotAwaitingChoice(false);
    setActiveReviewIndex(reviewProposals.length);
  }, [
    circuitDetail,
    currentFeature,
    currentFeatureIndex,
    currentFeatureRef,
    reviewProposals.length,
    selectedCircuitFile,
    selectedDirectoryId,
    selectedTaxonomy,
  ]);

  const handleSelectReviewProposal = useCallback(
    async (proposal: CircuitTaxonomyReviewProposal, index: number) => {
      setActiveReviewIndex(index);
      setReviewError(null);

      try {
        if (proposal.directoryId && proposal.directoryId !== selectedDirectoryId) {
          setSelectedDirectoryId(proposal.directoryId);
        }

        if (proposal.directoryId && proposal.fileName) {
          const detail = await fetchCircuitTaxonomyCircuit(proposal.directoryId, proposal.fileName);
          const matchedIndex =
            proposal.featureIndexInCircuit !== null && proposal.featureIndexInCircuit !== undefined
              ? proposal.featureIndexInCircuit
              : detail.features.findIndex(
                  (feature) =>
                    feature.dictionary_name === proposal.dictionaryName &&
                    feature.feature_index === proposal.featureIndex,
                );

          if (matchedIndex < 0 || !detail.features[matchedIndex]) {
            throw new Error("The proposal feature was not found in its circuit file.");
          }

          const featureRef = detail.features[matchedIndex];
          await ensureFeatureLoaded(featureRef);
          setSelectedCircuitFile(proposal.fileName);
          setCircuitDetail(detail);
          setCurrentFeatureIndex(matchedIndex);
          setClickedId(featureRef.node_id ?? null);
          setSelectedTaxonomy(proposal.taxonomy);
          return;
        }

        if (circuitDetail) {
          const matchedIndex = circuitDetail.features.findIndex(
            (feature) =>
              feature.dictionary_name === proposal.dictionaryName &&
              feature.feature_index === proposal.featureIndex,
          );
          if (matchedIndex >= 0) {
            const featureRef = circuitDetail.features[matchedIndex];
            await ensureFeatureLoaded(featureRef);
            setCurrentFeatureIndex(matchedIndex);
            setClickedId(featureRef.node_id ?? null);
            setSelectedTaxonomy(proposal.taxonomy);
            return;
          }
        }

        throw new Error("This proposal needs directoryId and fileName to jump from another circuit.");
      } catch (selectError) {
        const message = selectError instanceof Error ? selectError.message : "Failed to select review proposal.";
        setReviewError(message);
        updateReviewProposal(proposal.id, { status: "error", error: message });
      }
    },
    [circuitDetail, ensureFeatureLoaded, selectedDirectoryId, updateReviewProposal],
  );

  const handleChangeActiveReviewTaxonomy = useCallback(
    (taxonomy: string) => {
      if (!activeReviewProposal) {
        return;
      }
      updateReviewProposal(activeReviewProposal.id, {
        originalTaxonomy: activeReviewProposal.originalTaxonomy ?? activeReviewProposal.taxonomy,
        taxonomy,
        status: activeReviewProposal.status === "approved" ? "pending" : activeReviewProposal.status,
        error: undefined,
      });
      setSelectedTaxonomy(taxonomy);
    },
    [activeReviewProposal, updateReviewProposal],
  );

  const handleDownloadReviewEdits = useCallback(() => {
    const editedProposals = reviewProposals.filter(
      (proposal) =>
        proposal.originalTaxonomy &&
        proposal.taxonomy &&
        proposal.originalTaxonomy !== proposal.taxonomy,
    );

    if (editedProposals.length === 0) {
      window.alert("No edited taxonomy labels to download.");
      return;
    }

    const lines = editedProposals.map((proposal) =>
      JSON.stringify({
        id: proposal.id,
        directory_id: proposal.directoryId,
        file_name: proposal.fileName,
        circuit_index: proposal.circuitIndex,
        feature_index_in_circuit: proposal.featureIndexInCircuit,
        dictionary_name: proposal.dictionaryName,
        feature_index: proposal.featureIndex,
        layer: proposal.layer,
        feature_type: proposal.featureType,
        node_id: proposal.nodeId,
        original_taxonomy: proposal.originalTaxonomy,
        reviewed_taxonomy: proposal.taxonomy,
        status: proposal.status ?? "pending",
        confidence: proposal.confidence ?? null,
        rationale: proposal.rationale ?? "",
        evidence_summary: proposal.evidenceSummary ?? "",
        review_note: proposal.reviewNote ?? "",
      }),
    );

    const blob = new Blob([`${lines.join("\n")}\n`], { type: "application/x-ndjson" });
    const objectUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = "circuit-taxonomy-review-edits.jsonl";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(objectUrl);
  }, [reviewProposals]);

  const handleApproveReviewProposal = useCallback(async () => {
    if (!activeReviewProposal) {
      return;
    }

    setReviewSaving(true);
    setReviewError(null);
    try {
      let response = await annotateCircuitTaxonomyFeature(
        activeReviewProposal.dictionaryName,
        activeReviewProposal.featureIndex,
        activeReviewProposal.taxonomy,
      );

      if (response.status === "conflict") {
        const confirmed = window.confirm(
          `Current interpretation starts with [${response.existing_taxonomy}]. Replace it with [${activeReviewProposal.taxonomy}]?`,
        );
        if (!confirmed) {
          return;
        }
        response = await annotateCircuitTaxonomyFeature(
          activeReviewProposal.dictionaryName,
          activeReviewProposal.featureIndex,
          activeReviewProposal.taxonomy,
          true,
        );
      }

      if (
        currentFeatureRef?.dictionary_name === activeReviewProposal.dictionaryName &&
        currentFeatureRef?.feature_index === activeReviewProposal.featureIndex
      ) {
        await refreshFeature(currentFeatureRef);
      }

      updateReviewProposal(activeReviewProposal.id, {
        status: "approved",
        error: undefined,
      });
      const nextReview = moveToNextPendingReview(activeReviewIndex);
      if (nextReview) {
        await handleSelectReviewProposal(nextReview.proposal, nextReview.index);
      }
    } catch (approveError) {
      const message = approveError instanceof Error ? approveError.message : "Failed to approve proposal.";
      setReviewError(message);
      updateReviewProposal(activeReviewProposal.id, { status: "error", error: message });
    } finally {
      setReviewSaving(false);
    }
  }, [
    activeReviewIndex,
    activeReviewProposal,
    currentFeatureRef,
    handleSelectReviewProposal,
    moveToNextPendingReview,
    refreshFeature,
    updateReviewProposal,
  ]);

  const handleApproveReviewProposals = useCallback(
    async (scope: "current-circuit" | "all") => {
      const proposalsToApprove = reviewProposals.filter((proposal) => {
        if (!proposal.taxonomy || (proposal.status ?? "pending") === "approved") {
          return false;
        }
        if (scope === "all") {
          return true;
        }
        return Boolean(selectedCircuitFile) && proposal.fileName === selectedCircuitFile;
      });

      if (proposalsToApprove.length === 0) {
        setReviewError(scope === "all" ? "No pending review proposals to approve." : "No pending review proposals for the current circuit.");
        return;
      }

      const confirmed = window.confirm(
        scope === "all"
          ? `Approve and save ${proposalsToApprove.length} pending review proposals across all imported circuits?`
          : `Approve and save ${proposalsToApprove.length} pending review proposals for ${selectedCircuitFile}?`,
      );
      if (!confirmed) {
        return;
      }

      setReviewBatchSaving(true);
      setReviewError(null);
      try {
        const response = await annotateCircuitTaxonomyFeatures(
          proposalsToApprove.map((proposal) => ({
            dictionary_name: proposal.dictionaryName,
            feature_index: proposal.featureIndex,
            taxonomy: proposal.taxonomy,
          })),
          true,
        );
        const resultByProposalId = new Map<string, Record<string, unknown>>();
        response.results.forEach((result, index) => {
          resultByProposalId.set(proposalsToApprove[index].id, result);
        });

        const nextProposals = reviewProposals.map((proposal) => {
          const result = resultByProposalId.get(proposal.id);
          if (!result) {
            return proposal;
          }
          if (result.status === "error") {
            return {
              ...proposal,
              status: "error" as const,
              error: String(result.error ?? "Failed to approve proposal."),
            };
          }
          return {
            ...proposal,
            status: "approved" as const,
            error: undefined,
          };
        });

        setReviewProposals(nextProposals);
        const nextPendingIndex = nextProposals.findIndex((proposal) => (proposal.status ?? "pending") === "pending");
        setActiveReviewIndex(nextPendingIndex >= 0 ? nextPendingIndex : 0);
        if (reviewStateSaveTimerRef.current) {
          window.clearTimeout(reviewStateSaveTimerRef.current);
        }
        await persistReviewStateNow(nextProposals, nextPendingIndex >= 0 ? nextPendingIndex : 0);
        setReviewStateMessage(
          `Batch approved ${response.counts.updated ?? 0} updated, ${response.counts.unchanged ?? 0} unchanged, ${response.counts.error ?? 0} errors.`,
        );

        if (currentFeatureRef) {
          await refreshFeature(currentFeatureRef);
        }
      } catch (approveError) {
        setReviewError(approveError instanceof Error ? approveError.message : "Failed to batch approve review proposals.");
      } finally {
        setReviewBatchSaving(false);
      }
    },
    [
      currentFeatureRef,
      persistReviewStateNow,
      refreshFeature,
      reviewProposals,
      selectedCircuitFile,
    ],
  );

const handleExportEvidence = useCallback(async () => {
    if (!selectedDirectoryId || !selectedCircuitFile) {
      return;
    }

    setExportingEvidence(true);
    setError(null);
    try {
      const url = buildCircuitTaxonomyEvidenceExportUrl(selectedDirectoryId, {
        fileName: selectedCircuitFile,
        startFeatureIndex: 0,
        limit: 0,
        maxSamples: 6,
        topSquares: 8,
        topZ: 12,
        includeAnnotated: true,
        singleFileOnly: true,
      });
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `circuit-taxonomy-evidence-${selectedCircuitFile.replace(/[^a-zA-Z0-9_-]+/g, "_")}.jsonl`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(objectUrl);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Failed to export taxonomy evidence");
    } finally {
      setExportingEvidence(false);
    }
  }, [selectedCircuitFile, selectedDirectoryId]);

  const handleExportDirectoryEvidence = useCallback(async () => {
    if (!selectedDirectoryId) {
      return;
    }

    setExportingDirectoryEvidence(true);
    setError(null);
    try {
      const url = buildCircuitTaxonomyEvidenceExportUrl(selectedDirectoryId, {
        startFeatureIndex: 0,
        limit: 0,
        maxSamples: 6,
        topSquares: 8,
        topZ: 12,
        includeAnnotated: true,
      });
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `circuit-taxonomy-evidence-${selectedDirectoryId.replace(/[^a-zA-Z0-9_-]+/g, "_")}.jsonl`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(objectUrl);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Failed to export directory taxonomy evidence");
    } finally {
      setExportingDirectoryEvidence(false);
    }
  }, [selectedDirectoryId]);

  const handleSaveAllCircuitEvidence = useCallback(async () => {
    if (!selectedDirectoryId) {
      return;
    }

    setSavingAllCircuitEvidence(true);
    setSavedDirectoryEvidencePath(null);
    setError(null);
    try {
      const response = await saveAllCircuitTaxonomyEvidence(selectedDirectoryId, {
        maxSamples: 6,
        topSquares: 8,
        topZ: 12,
        includeAnnotated: true,
      });
      setSavedDirectoryEvidencePath(
        `${response.relative_path} (${response.file_count ?? 0} files, ${response.item_count} items)`,
      );
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save per-circuit taxonomy evidence");
    } finally {
      setSavingAllCircuitEvidence(false);
    }
  }, [selectedDirectoryId]);

  const directoryLabel = directories.find((item) => item.id === selectedDirectoryId)?.label ?? selectedDirectoryId;
  const featureProgressValue = circuitDetail ? currentFeatureIndex + 1 : 0;
  const featureProgressMax = circuitDetail?.total_features ?? 1;
  const circuitProgressValue = circuitDetail ? circuitDetail.circuit_index + 1 : 0;
  const circuitProgressMax = circuitDetail?.total_circuits ?? 1;

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Taxonomy Annotation</CardTitle>
          <CardDescription>
            Load a saved circuit, browse its feature list in layer order, and write taxonomy labels back to MongoDB interpretations.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <span className="text-sm font-medium">Circuit Directory</span>
            <Select value={selectedDirectoryId} onValueChange={setSelectedDirectoryId} disabled={loadingDirectories}>
              <SelectTrigger>
                <SelectValue placeholder="Select a circuit directory" />
              </SelectTrigger>
              <SelectContent>
                {directories.map((directory) => (
                  <SelectItem key={directory.id} value={directory.id}>
                    {directory.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => void handleExportDirectoryEvidence()}
                  disabled={!selectedDirectoryId || exportingDirectoryEvidence}
                >
                  {exportingDirectoryEvidence ? "Exporting Directory..." : "Export Directory Evidence"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => void handleSaveAllCircuitEvidence()}
                  disabled={!selectedDirectoryId || savingAllCircuitEvidence}
                >
                  {savingAllCircuitEvidence ? "Saving Circuit Files..." : "Save All Circuit Evidence"}
                </Button>
              </div>
              {savedDirectoryEvidencePath && (
                <div className="mt-2 break-all text-xs text-muted-foreground">
                  Saved to {savedDirectoryEvidencePath}
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <span className="text-sm font-medium">Circuit File</span>
            <Select value={selectedCircuitFile} onValueChange={setSelectedCircuitFile} disabled={loadingCircuits || circuits.length === 0}>
              <SelectTrigger>
                <SelectValue placeholder="Select a circuit file" />
              </SelectTrigger>
              <SelectContent>
                {circuits.map((circuit) => (
                  <SelectItem key={circuit.file_name} value={circuit.file_name}>
                    {`${circuit.index + 1}. ${circuit.file_name}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-300">
          <CardContent className="pt-6 text-sm text-red-600">{error}</CardContent>
        </Card>
      )}

      {circuitDetail && (
        <Card>
          <CardContent className="grid gap-4 pt-6 md:grid-cols-2">
            <div className="flex flex-col gap-2">
              <div className="text-sm font-medium">{directoryLabel}</div>
              <div className="text-sm text-muted-foreground break-all">{circuitDetail.file_name}</div>
              <div className="text-sm text-muted-foreground">
                Prompt: {String(circuitDetail.metadata.prompt ?? "")}
              </div>
              <div className="text-sm text-muted-foreground">
                Target Move: {String(circuitDetail.metadata.target_move ?? "-")}
              </div>
            </div>

            <div className="grid gap-4">
              <div className="grid gap-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Circuit Progress</span>
                  <span>{circuitProgressValue} / {circuitProgressMax}</span>
                </div>
                <Progress value={circuitProgressValue} max={circuitProgressMax} />
              </div>

              <div className="grid gap-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Feature Progress</span>
                  <span>{featureProgressValue} / {featureProgressMax}</span>
                </div>
                <Progress value={featureProgressValue} max={featureProgressMax} />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.3fr_1fr]">
        <Card className="min-h-[720px]">
          <CardHeader>
            <CardTitle>Circuit Graph</CardTitle>
            <CardDescription>
              Click a node to jump to that feature. Embedding and logit nodes stay hidden in this annotation view.
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[720px]">
            {graphData ? (
              <LinkGraphContainer
                data={graphData}
                clickedId={clickedId}
                onNodeClick={(node) => setClickedId(node.nodeId || null)}
                onFeatureSelect={handleGraphFeatureSelect}
                hideEmbLogit
              />
            ) : (
              <div className="text-sm text-muted-foreground">
                {loadingCircuitDetail ? "Loading circuit graph..." : "Select a circuit to start."}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Current Feature</CardTitle>
              <CardDescription>
                {currentFeatureRef
                  ? `${currentFeatureRef.label} | dictionary ${currentFeatureRef.dictionary_name}`
                  : "No feature selected"}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {currentFeatureRef && (
                <>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={() => goToFeatureIndex(currentFeatureIndex - 1)}
                      disabled={currentFeatureIndex <= 0}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => goToNextFeature()}
                      disabled={!circuitDetail}
                    >
                      Skip
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => void jumpToCurrentFilePendingFeature()}
                      disabled={!circuitDetail || jumpingToPending}
                    >
                      {jumpingToPending ? "Finding Pending..." : "Jump To Pending"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => void handleExportEvidence()}
                      disabled={!circuitDetail || exportingEvidence}
                    >
                      {exportingEvidence ? "Exporting..." : "Export Circuit Evidence"}
                    </Button>
                  </div>

                  <div className="grid gap-2">
                    <div className="text-sm font-medium">Taxonomy Label</div>
                    <div className="grid grid-cols-2 gap-2">
                      {taxonomyLabels.map((label) => (
                        <Button
                          key={label}
                          type="button"
                          variant={selectedTaxonomy === label ? "default" : "outline"}
                          onClick={() => setSelectedTaxonomy(label)}
                        >
                          {label}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div className="grid gap-2 text-sm">
                    <div>
                      <span className="font-medium">Current Prefix:</span>{" "}
                      {extractTaxonomyPrefix(currentFeature?.interpretation?.text) || "None"}
                    </div>
                    <div className="whitespace-pre-wrap rounded-md border bg-slate-50 p-3">
                      {currentFeature
                        ? (currentFeature.interpretation?.text || "No interpretation available.")
                        : "Loading feature interpretation and top activation samples..."}
                    </div>
                  </div>

                  <Button onClick={handleSaveAndNext} disabled={saving || !currentFeatureRef}>
                    {saving ? "Saving..." : "Confirm And Next"}
                  </Button>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Top Activation Samples</CardTitle>
              <CardDescription>
                The current feature is loaded first, and the next few features are prefetched in the background for faster annotation.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {currentFeature && topActivationGroup ? (
                <CircuitTaxonomyTopActivationBoards sampleGroup={topActivationGroup} />
              ) : (
                <div className="text-sm text-muted-foreground">
                  {currentFeatureRef ? "Loading top activation samples..." : "Select a feature to inspect samples."}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Tabs defaultValue="review" className="w-full">
        <TabsList>
          <TabsTrigger value="review">
            LLM Review Queue ({reviewCounts.pending} pending)
          </TabsTrigger>
          <TabsTrigger value="import">Import Proposals</TabsTrigger>
        </TabsList>

        <TabsContent value="review" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>LLM Taxonomy Review</CardTitle>
              <CardDescription>
                Review candidate labels before committing them to MongoDB interpretations.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
              <div className="flex flex-col gap-3">
                <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
                  <span>Pending: {reviewCounts.pending}</span>
                  <span>Approved: {reviewCounts.approved}</span>
                  <span>Errors: {reviewCounts.error}</span>
                  <span>Edited: {reviewCounts.edited}</span>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    onClick={handleAddCurrentFeatureToReview}
                    disabled={!currentFeatureRef}
                  >
                    Add Current Feature
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => activeReviewProposal && void handleSelectReviewProposal(activeReviewProposal, activeReviewIndex)}
                    disabled={!activeReviewProposal}
                  >
                    Jump To Feature
                  </Button>
                  <Button onClick={handleApproveReviewProposal} disabled={!activeReviewProposal || reviewSaving}>
                    {reviewSaving ? "Approving..." : "Approve And Save"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => void handleApproveReviewProposals("current-circuit")}
                    disabled={!selectedCircuitFile || reviewBatchSaving}
                  >
                    {reviewBatchSaving ? "Approving..." : "Approve Current Circuit"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => void handleApproveReviewProposals("all")}
                    disabled={reviewProposals.length === 0 || reviewBatchSaving}
                  >
                    {reviewBatchSaving ? "Approving..." : "Approve All Imported"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleDownloadReviewEdits}
                    disabled={reviewCounts.edited === 0}
                  >
                    Download Review Edits
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => void handleSnapshotReviewState()}
                    disabled={reviewProposals.length === 0 || reviewSnapshotSaving}
                  >
                    {reviewSnapshotSaving ? "Saving Snapshot..." : "Save Review State"}
                  </Button>
                </div>

                {reviewError && (
                  <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700">
                    {reviewError}
                  </div>
                )}
                {reviewStateMessage && (
                  <div className="space-y-3 rounded-md border bg-slate-50 p-3 text-sm text-muted-foreground">
                    <div>{reviewStateMessage}</div>
                    {reviewSnapshotAwaitingChoice && (
                      <div className="flex flex-wrap gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={handleContinueReviewState}
                          disabled={reviewResetting}
                        >
                          Continue Current Review State
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => void handleStartBlankReviewState()}
                          disabled={reviewResetting}
                        >
                          {reviewResetting ? "Starting Blank..." : "Start Blank Review State"}
                        </Button>
                      </div>
                    )}
                  </div>
                )}

                <div className="max-h-[520px] overflow-y-auto rounded-md border">
                  {reviewProposals.length === 0 ? (
                    <div className="p-4 text-sm text-muted-foreground">
                      No LLM proposals loaded yet. Import JSON or JSONL proposals to start reviewing.
                    </div>
                  ) : (
                    reviewProposals.map((proposal, index) => (
                      <button
                        key={proposal.id}
                        type="button"
                        className={`block w-full border-b p-3 text-left text-sm last:border-b-0 ${
                          index === activeReviewIndex ? "bg-slate-100" : "bg-background"
                        }`}
                        onClick={() => void handleSelectReviewProposal(proposal, index)}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-medium">
                            [{proposal.taxonomy}] {proposal.dictionaryName} #{proposal.featureIndex}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {proposal.status ?? "pending"}
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {proposal.fileName ?? "current circuit"}{" "}
                          {proposal.confidence !== null && proposal.confidence !== undefined
                            ? `| confidence ${proposal.confidence.toFixed(2)}`
                            : ""}
                          {proposal.originalTaxonomy && proposal.originalTaxonomy !== proposal.taxonomy
                            ? ` | ${proposal.originalTaxonomy} -> ${proposal.taxonomy}`
                            : ""}
                        </div>
                      </button>
                    ))
                  )}
                </div>
              </div>

              <div className="flex flex-col gap-4">
                {activeReviewProposal ? (
                  <>
                    <div className="grid gap-2 text-sm">
                      <div className="text-lg font-semibold">
                        [{activeReviewProposal.taxonomy}] {activeReviewProposal.dictionaryName} #
                        {activeReviewProposal.featureIndex}
                      </div>
                      <div className="text-muted-foreground">
                        {activeReviewProposal.fileName ?? "No circuit file recorded"}
                      </div>
                      <div>
                        <span className="font-medium">Status:</span>{" "}
                        {activeReviewProposal.status ?? "pending"}
                      </div>
                      <div>
                        <span className="font-medium">Confidence:</span>{" "}
                        {activeReviewProposal.confidence !== null &&
                        activeReviewProposal.confidence !== undefined
                          ? activeReviewProposal.confidence.toFixed(2)
                          : "-"}
                      </div>
                      <div>
                        <span className="font-medium">Layer / Type:</span>{" "}
                        {activeReviewProposal.layer ?? "-"} / {activeReviewProposal.featureType ?? "-"}
                      </div>
                    </div>

                    <div className="grid gap-2">
                      <div className="grid gap-2 rounded-md border bg-slate-50 p-3 text-sm sm:grid-cols-2">
                        <div>
                          <div className="text-xs font-medium text-muted-foreground">Original Review Label</div>
                          <div className="mt-1 font-semibold">
                            {activeReviewProposal.originalTaxonomy ?? activeReviewProposal.taxonomy}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs font-medium text-muted-foreground">Reviewed Label</div>
                          <div className="mt-1 font-semibold">{activeReviewProposal.taxonomy}</div>
                        </div>
                      </div>
                      <div className="text-sm font-medium">Set Reviewed Label</div>
                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                        {taxonomyLabels.map((label) => (
                          <Button
                            key={label}
                            type="button"
                            variant={activeReviewProposal.taxonomy === label ? "default" : "outline"}
                            onClick={() => handleChangeActiveReviewTaxonomy(label)}
                          >
                            {label}
                          </Button>
                        ))}
                      </div>
                    </div>

                    <div className="grid gap-2">
                      <div className="text-sm font-medium">Rationale</div>
                      <div className="min-h-[96px] whitespace-pre-wrap rounded-md border bg-slate-50 p-3 text-sm">
                        {activeReviewProposal.rationale || "No rationale provided."}
                      </div>
                    </div>

                    <div className="grid gap-2">
                      <div className="text-sm font-medium">Evidence Summary</div>
                      <div className="min-h-[132px] whitespace-pre-wrap rounded-md border bg-slate-50 p-3 text-sm">
                        {activeReviewProposal.evidenceSummary || "No evidence summary provided."}
                      </div>
                    </div>

                    <div className="grid gap-2">
                      <div className="text-sm font-medium">Review Note</div>
                      <Textarea
                        value={activeReviewProposal.reviewNote ?? ""}
                        onChange={(event) =>
                          updateReviewProposal(activeReviewProposal.id, {
                            reviewNote: event.target.value,
                          })
                        }
                        placeholder="Optional note about why you changed or accepted this label."
                        className="min-h-[88px] text-sm"
                      />
                    </div>

                    <div className="grid gap-2">
                      <div className="text-sm font-medium">Top Activation Samples</div>
                      <div className="rounded-md border p-3">
                        {currentFeatureMatchesActiveReview && currentFeature && topActivationGroup ? (
                          <CircuitTaxonomyTopActivationBoards sampleGroup={topActivationGroup} />
                        ) : (
                          <div className="text-sm text-muted-foreground">
                            Click Jump To Feature if samples are not loaded for this proposal yet.
                          </div>
                        )}
                      </div>
                    </div>

                    {activeReviewProposal.error && (
                      <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700">
                        {activeReviewProposal.error}
                      </div>
                    )}

                  </>
                ) : (
                  <div className="text-sm text-muted-foreground">Select a proposal to review.</div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="import" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Import LLM Proposals</CardTitle>
              <CardDescription>
                Paste JSONL, a JSON array, or review-state.json with proposals. Required fields are dictionary_name, feature_index, and taxonomy.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <Textarea
                value={reviewImportText}
                onChange={(event) => setReviewImportText(event.target.value)}
                placeholder='{"dictionary_name":"BT4_lorsa_L0A_k30_e16","feature_index":123,"taxonomy":"Src","confidence":0.82,"rationale":"...","evidence_summary":"..."}'
                className="min-h-[220px] font-mono text-sm"
              />
              <div className="flex flex-wrap gap-2">
                <Button onClick={handleImportReviewProposals}>Import</Button>
                <label className="inline-flex h-10 cursor-pointer items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground">
                  Import Files
                  <input
                    type="file"
                    accept=".json,.jsonl,application/json,application/x-ndjson"
                    multiple
                    className="hidden"
                    onChange={(event) => {
                      void handleImportReviewProposalFiles(event.target.files);
                      event.target.value = "";
                    }}
                  />
                </label>
                <Button variant="outline" onClick={() => setReviewImportText("")}>
                  Clear
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
