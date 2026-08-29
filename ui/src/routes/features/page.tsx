import { AppNavbar } from "@/components/app/navbar";
import { SectionNavigator } from "@/components/app/section-navigator";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Combobox } from "@/components/ui/combobox";
import { FeatureSchema } from "@/types/feature";
import { decode } from "@msgpack/msgpack";
import camelcaseKeys from "camelcase-keys";
import { useEffect, useState, useMemo, useCallback, Suspense, lazy } from "react";
import { useSearchParams } from "react-router-dom";
import { useAsyncFn, useMount, useDebounce } from "react-use";
import { z } from "zod";

const FeatureCard = lazy(() =>
  import("@/components/feature/feature-card").then((module) => ({
    default: module.FeatureCard,
  })),
);
const FeatureInterpretationCard = lazy(() =>
  import("@/components/feature/interpret").then((module) => ({
    default: module.FeatureInterpretationCard,
  })),
);
import { ChessBoard } from "@/components/chess/chess-board";
import { CustomFenInput } from "@/components/feature/custom-fen-input";

let boardCounter = 0;
const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/+$/, "");

const backendUrl = (path: string) => `${BACKEND_URL}${path}`;

const fetchFromBackend = async (path: string, init?: RequestInit) => {
  const url = backendUrl(path);
  try {
    return await fetch(url, init);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Backend request failed (${url}): ${message}`);
  }
};


export const FeaturesPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  
  // Use local state for features
  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);
  const [featureIndex, setFeatureIndex] = useState<number>(0);
  const [currentFeature, setCurrentFeature] = useState<any>(null);
  const [featureLoading, setFeatureLoading] = useState<boolean>(false);
  const [featureError, setFeatureError] = useState<string | null>(null);

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetchFromBackend("/dictionaries")
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [analysesState, fetchAnalyses] = useAsyncFn(async (dictionary: string) => {
    if (!dictionary) return [];

    return await fetchFromBackend(`/dictionaries/${encodeURIComponent(dictionary)}/analyses`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [inputValue, setInputValue] = useState<string>("0");
  const [loadingRandomFeature, setLoadingRandomFeature] = useState<boolean>(false);
  const [selectedSubsample, setSelectedSubsample] = useState<string | "all">("all");

  // Debounce the input value to avoid excessive updates
  useDebounce(
    () => {
      const parsed = parseInt(inputValue);
      if (!isNaN(parsed) && parsed !== featureIndex) {
        setFeatureIndex(parsed);
      }
    },
    300,
    [inputValue]
  );

  const handleFeatureIndexChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  }, []);

  const [featureState, fetchFeature] = useAsyncFn(
    async (
      dictionary: string | null,
      featureIndex: number | string = "random",
      analysisName: string | null = null
    ) => {
      if (!dictionary) {
        alert("Please select a dictionary first");
        return;
      }

       setLoadingRandomFeature(featureIndex === "random");
      setFeatureLoading(true);
      setFeatureError(null);

      try {
        const query = analysisName
          ? `?feature_analysis_name=${encodeURIComponent(analysisName)}`
          : "";
        const feature = await fetchFromBackend(
          `/dictionaries/${encodeURIComponent(dictionary)}/features/${featureIndex}${query}`,
          {
        method: "GET",
        headers: {
          Accept: "application/x-msgpack",
        },
          }
        )
        .then(async (res) => {
          if (!res.ok) {
            throw new Error(await res.text());
          }
          return res;
        })
        .then(async (res) => await res.arrayBuffer())
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .then((res) => decode(new Uint8Array(res)) as any)
        .then((res) =>
          camelcaseKeys(res, {
            deep: true,
              stopPaths: ["sample_groups.samples.context"],
          })
        )
        .then((res) => FeatureSchema.parse(res));
      
      setFeatureIndex(feature.featureIndex);
      setSelectedAnalysis(feature.analysisName);
      setCurrentFeature(feature);
      setSearchParams({
        dictionary,
        featureIndex: feature.featureIndex.toString(),
        analysis: feature.analysisName,
      });
      return feature;
      } catch (error) {
        setFeatureError(error instanceof Error ? error.message : 'Failed to load feature');
        throw error;
      } finally {
        setFeatureLoading(false);
      }
    }
  );

  useMount(async () => {
    await fetchDictionaries();
    if (searchParams.get("dictionary")) {
      const dict = searchParams.get("dictionary")!;
      const analysisParam = searchParams.get("analysis");
      setSelectedDictionary(dict);

      fetchAnalyses(dict).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analysisParam || analyses[0]);
        }
      });

      if (searchParams.get("featureIndex")) {
        setFeatureIndex(parseInt(searchParams.get("featureIndex")!));
        fetchFeature(dict, searchParams.get("featureIndex")!, analysisParam || null);
      }
    }
  });

  useEffect(() => {
    if (dictionariesState.value && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchAnalyses(dictionariesState.value[0]).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analyses[0]);
        }
      });

      fetchFeature(dictionariesState.value[0], "random", selectedAnalysis);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  useEffect(() => {
    if (selectedDictionary) {
      fetchAnalyses(selectedDictionary);
      setSelectedAnalysis(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDictionary]);

  useEffect(() => {
    setSelectedSubsample("all");
  }, [currentFeature?.featureIndex, currentFeature?.dictionaryName]);


  // Memoize dictionary options for Combobox
  const dictionaryOptions = useMemo(() => {
    if (!dictionariesState.value) return [];
    return dictionariesState.value.map((dict) => ({
      value: dict,
      label: dict,
    }));
  }, [dictionariesState.value]);

  // Memoize sections calculation
  const sections = useMemo(() => [
    {
      title: "Histogram",
      id: "Histogram",
    },
    {
      title: "Decoder Norms",
      id: "DecoderNorms",
    },
    {
      title: "Similarity Matrix",
      id: "DecoderSimilarityMatrix",
    },
    {
      title: "Inner Product Matrix",
      id: "DecoderInnerProductMatrix",
    },
    {
      title: "Logits",
      id: "Logits",
    },
    {
      title: "Top Activation",
      id: "Activation",
    },
  ].filter((section) => (currentFeature && currentFeature.logits != null) || section.id !== "Logits"), [currentFeature]);

  return (
    <div id="Top">
      <AppNavbar />
      <div className="pt-4 pb-20 px-20 flex flex-col items-center gap-12">
        <div className="container grid grid-cols-[auto_600px_auto_auto] justify-center items-center gap-4">
          <span className="font-bold justify-self-end">Select dictionary:</span>
          <Combobox
            disabled={dictionariesState.loading || featureLoading}
            value={selectedDictionary || null}
            onChange={(value) => {
              setSelectedDictionary(value);
            }}
            options={dictionaryOptions}
            placeholder="Select dictionary..."
            commandPlaceholder="Search dictionary..."
            emptyIndicator="No matching dictionary found"
            className="w-full"
          />
          <Button
            disabled={dictionariesState.loading || featureLoading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis);
            }}
          >
            Go
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Select analysis:</span>
          <Select
            disabled={analysesState.loading || !selectedDictionary || featureLoading}
            value={selectedAnalysis || undefined}
            onValueChange={setSelectedAnalysis}
          >
            <SelectTrigger className="bg-white">
              <SelectValue placeholder="Select an analysis" />
            </SelectTrigger>
            <SelectContent>
              {analysesState.value?.map((analysis, i) => (
                <SelectItem key={i} value={analysis}>
                  {analysis}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            disabled={analysesState.loading || !selectedDictionary || featureLoading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis);
            }}
          >
            Apply
          </Button>
          <span className="font-bold"></span>

          <span className="font-bold justify-self-end">Choose a specific feature:</span>
          <Input
            disabled={dictionariesState.loading || selectedDictionary === null || featureLoading}
            id="feature-input"
            className="bg-white"
            type="number"
            value={inputValue}
            onChange={handleFeatureIndexChange}
          />
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureLoading}
            onClick={async () => await fetchFeature(selectedDictionary, featureIndex, selectedAnalysis)}
          >
            Go
          </Button>
          <Button
            disabled={dictionariesState.loading || selectedDictionary === null || featureLoading}
            onClick={async () => {
              await fetchFeature(selectedDictionary, "random", selectedAnalysis);
            }}
          >
            Show Random Feature
          </Button>
                    </div>

        {/* Custom FEN analysis area */}
        <div className="container w-full max-w-6xl mx-auto mb-8">
          <CustomFenInput
            dictionary={selectedDictionary}
            featureIndex={featureIndex}
            disabled={
              dictionariesState.loading ||
              selectedDictionary === null ||
              featureLoading
            }
          />
        </div>

        {/* Interpretation editor for current feature (with backend submit) */}
        {currentFeature && (
          <div className="container w-full max-w-6xl mx-auto mb-8">
            <Suspense fallback={<div>Loading Interpretation...</div>}>
              <FeatureInterpretationCard feature={currentFeature} />
            </Suspense>
          </div>
        )}

        {featureLoading && !loadingRandomFeature && (
          <div>
            Loading Feature <span className="font-bold">#{featureIndex}</span>...
          </div>
        )}
        {featureLoading && loadingRandomFeature && <div>Loading Random Living Feature...</div>}
        {featureError && <div className="text-red-500 font-bold">Error: {featureError}</div>}
        
        {!featureLoading && currentFeature && (() => {
          const feature = currentFeature;
          
          console.log('Feature #' + feature.featureIndex + ' (Activation times: ' + feature.actTimes + ')');
          
          // Check possible sample data fields
          const possibleSampleFields = ['sampleGroups', 'samples', 'sample_groups', 'sample_groups_', 'samples_'];
          let actualSampleGroups: any[] = [];
          
          for (const field of possibleSampleFields) {
            if (feature[field]) {
              if (Array.isArray(feature[field]) && feature[field].length > 0) {
                actualSampleGroups = feature[field];
                break;
              }
            }
          }
          
          // If no sample groups are found, try to directly find the samples field
          if (!actualSampleGroups.length && feature.samples) {
            actualSampleGroups = [{ samples: feature.samples, analysisName: "default" }];
          }

          const subsampleLabel = (name: string) => {
            if (name === "top_activations") return "Top Activations";
            if (/^sampling_[\d.]+$/.test(name)) {
              const pct = name.replace("sampling_", "");
              return `Sampling ${(parseFloat(pct) * 100).toFixed(0)}%`;
            }
            return name;
          };

          const groupsToShow =
            selectedSubsample === "all"
              ? actualSampleGroups
              : actualSampleGroups.filter((g) => g.analysisName === selectedSubsample);

          // Collect all samples containing FEN and their information
          interface ChessSample {
            fen: string;
            sample: any;
            groupIndex: number;
            sampleIndex: number;
            lineIndex: number;
            activationsArray?: number[];
            zPatternIndices?: number[][];
            zPatternValues?: number[];
            maxActivation: number;
            analysisName: string;
            contextId?: number;
            activeColor: string;
          }
          
          const chessSamples: ChessSample[] = [];
          let totalSamples = 0;
          let samplesWithText = 0;
          let samplesWithActivations = 0;
          let debugInfo: string[] = [];
          
          // Iterate through selected sample groups, collect samples containing FEN
          for (const [groupIndex, group] of groupsToShow.entries()) {
            console.log(`Check sample group ${groupIndex}:`, {
              analysisName: group.analysisName,
              samplesCount: group.samples?.length || 0
            });
            
            totalSamples += group.samples.length;
                
            for (const [sampleIndex, sample] of (group.samples || []).entries()) {
              if (sample.text) samplesWithText++;
              
              // Check if there is activation value data
              const hasActivations = (sample as any).featureActsIndices && (sample as any).featureActsValues && 
                                    (sample as any).featureActsIndices.length > 0 && (sample as any).featureActsValues.length > 0;
              if (hasActivations) samplesWithActivations++;
              
              // Detailed debug information
              if (sampleIndex < 3) { // Only show detailed information for the first 3 samples
                console.log(`Sample ${sampleIndex} detailed information:`, {
                  hasText: !!sample.text,
                  textLength: sample.text?.length || 0,
                  textPreview: sample.text?.substring(0, 100) || 'No text',
                  hasFeatureActsIndices: !!(sample as any).featureActsIndices,
                  featureActsIndicesLength: (sample as any).featureActsIndices?.length || 0,
                  hasFeatureActsValues: !!(sample as any).featureActsValues,
                  featureActsValuesLength: (sample as any).featureActsValues?.length || 0,
                  hasZPatternIndices: !!(sample as any).zPatternIndices,
                  zPatternIndicesLength: (sample as any).zPatternIndices?.length || 0,
                  hasZPatternValues: !!(sample as any).zPatternValues,
                  zPatternValuesLength: (sample as any).zPatternValues?.length || 0
                });
              }
                  
              if (sample.text) {
                const lines = sample.text.split('\n');
                
                for (const [lineIndex, line] of lines.entries()) {
                  const trimmed = line.trim();
                  
                  // Just contains 8 parts separated by slashes, and each part is a valid combination of chess pieces or numbers
                  if (trimmed.includes('/')) {
                    const parts = trimmed.split(/\s+/);
                    
                    if (parts.length >= 6) {
                      const [boardPart, activeColor, castling, enPassant, halfmove, fullmove] = parts;
                      const boardRows = boardPart.split('/');
                      
                      if (boardRows.length === 8) {
                        let isValidBoard = true;
                        let totalSquares = 0;
                        
                        // Check if each row matches FEN format
                        for (const row of boardRows) {
                          if (!/^[rnbqkpRNBQKP1-8]+$/.test(row)) {
                            isValidBoard = false;
                            break;
                          }
                          
                          // Calculate the number of squares in each row
                          let rowSquares = 0;
                          for (const char of row) {
                            if (/\d/.test(char)) {
                              rowSquares += parseInt(char);
                            } else {
                              rowSquares += 1;
                            }
                          }
                          totalSquares += rowSquares;
                        }
                        
                        // Validate that the total number of squares is 64, and the active color is valid
                        if (isValidBoard && totalSquares === 64 && /^[wb]$/.test(activeColor)) {
                          console.log(`Found valid FEN string: "${trimmed}"`);
                          console.log(`Board rows:`, boardRows);
                          console.log(`Total squares:`, totalSquares);
                          
                          // Build activation value array and calculate maximum activation value
                          let activationsArray: number[] | undefined = undefined;
                          let maxActivation = 0;
                          
                          if ((sample as any).featureActsIndices && (sample as any).featureActsValues) {
                            activationsArray = new Array(64).fill(0);
                            const indices = (sample as any).featureActsIndices;
                            const values = (sample as any).featureActsValues;
                            
                            for (let i = 0; i < Math.min(indices.length, values.length); i++) {
                              const index = indices[i];
                              const value = values[i];
                              if (index >= 0 && index < 64) {
                                activationsArray[index] = value;
                                // Calculate maximum activation value (absolute value)
                                if (Math.abs(value) > Math.abs(maxActivation)) {
                                  maxActivation = value;
                                }
                              }
                            }
                            
                            console.log(`Build activation value array: ${activationsArray.filter(v => v !== 0).length} non-zero values, maximum activation value: ${maxActivation}`);
                          }

                          // Build Z Pattern array and print debug information
                          let zPatternIndices: number[][] | undefined = undefined;
                          let zPatternValues: number[] | undefined = undefined;
                          if ((sample as any).zPatternIndices && (sample as any).zPatternValues) {
                            const zpIdxRaw = (sample as any).zPatternIndices;
                            zPatternIndices = Array.isArray(zpIdxRaw) && Array.isArray(zpIdxRaw[0]) ? zpIdxRaw : [zpIdxRaw];
                            zPatternValues = (sample as any).zPatternValues;

                            try {
                              const len = zPatternValues?.length || 0;
                              console.log(`z_pattern_values shape: (${len},)`);
                              // Print all indices and values
                              console.log('z all idx:', zPatternIndices);
                              console.log('z all val:', zPatternValues);
                              // Identify format and print all pairs
                              const looksLikePairList = Array.isArray(zPatternIndices?.[0]) && (zPatternIndices?.[0] as number[]).length === 2;
                              if (looksLikePairList) {
                                const pairsAll = (zPatternIndices || []).map((p, i) => ([p, (zPatternValues || [])[i]]));
                                console.log('z pairs (all):', pairsAll);
                              } else {
                                const pairs: Array<[number[], number]> = [];
                                for (let t = 0; t < (zPatternIndices || []).length; t++) {
                                  const srcs = zPatternIndices?.[t] || [];
                                  const v = (zPatternValues || [])[t];
                                  for (const s of srcs) {
                                    pairs.push([[s, t], v]);
                                  }
                                }
                                console.log('z pairs (all expanded to [source,target]):', pairs);
                              }
                            } catch (e) {
                              console.warn('Error printing z_pattern debug information:', e);
                            }
                          }

                          // Collect sample information
                          chessSamples.push({
                            fen: trimmed,
                            sample,
                            groupIndex,
                            sampleIndex,
                            lineIndex,
                            activationsArray,
                            zPatternIndices,
                            zPatternValues,
                            maxActivation,
                            analysisName: group.analysisName || '',
                            contextId: (sample as any).context_idx,
                            activeColor
                          });
                          
                          break; // Found a valid FEN, jump out of the loop
                        } else {
                          debugInfo.push(`FEN validation failed: ${trimmed.substring(0, 30)}...`);
                        }
                      } else {
                        debugInfo.push(`Number of rows error: ${boardRows.length} rows`);
                      }
                    } else {
                      debugInfo.push(`Part missing: ${parts.length} parts`);
                    }
                  }
                }
              }
            }
          }
          
          // Sort by maximum activation value from largest to smallest
          chessSamples.sort((a, b) => Math.abs(b.maxActivation) - Math.abs(a.maxActivation));
          
          console.log('Analysis completed: found ' + chessSamples.length + ' FEN samples');
          console.log('Sorted maximum activation values:', chessSamples.slice(0, 10).map(s => s.maxActivation.toFixed(3)));
          
          // Generate chessboard components
          const chessBoards: JSX.Element[] = chessSamples.map((chessSample, index) => (
            <ChessBoard
              key={`chess-${chessSample.groupIndex}-${chessSample.sampleIndex}-${chessSample.lineIndex}-${index}`}
              fen={chessSample.fen}
              activations={chessSample.activationsArray}
              zPatternIndices={chessSample.zPatternIndices}
              zPatternValues={chessSample.zPatternValues}
              sampleIndex={chessSample.sampleIndex}
              analysisName={chessSample.analysisName}
              contextId={chessSample.contextId}
              autoFlipWhenBlack={true}
              flip_activation={Boolean(chessSample.activeColor === 'b')}
              showSelfPlay={true}
              disableAutoAnalyze={true}
            />
          ));
          
          const validFENFound = chessSamples.length;
          const uniqueSubsampleNames = Array.from(
            new Set(actualSampleGroups.map((g) => g.analysisName).filter(Boolean))
          );

          if (chessBoards.length > 0) {
            console.log(`Generated ${chessBoards.length} chessboards`);
          }

          return (
            <div className="w-full max-w-6xl mx-auto mb-8">
              <div className="text-center mb-6">
                <h3 className="text-xl font-bold mb-2">Feature #{feature.featureIndex} Chessboard Visualization</h3>
                <div className="text-sm text-gray-600 mb-4">
                  {validFENFound > 0
                    ? `Found ${validFENFound} samples containing FEN (sorted by maximum activation value)`
                    : selectedSubsample !== "all"
                      ? `No FEN samples in ${subsampleLabel(selectedSubsample)}. Try another subsample or All.`
                      : "No samples containing FEN in this feature."}
                </div>
                {uniqueSubsampleNames.length > 1 && (
                  <div className="flex flex-wrap justify-center gap-2">
                    <Button
                      variant={selectedSubsample === "all" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedSubsample("all")}
                    >
                      All
                    </Button>
                    {uniqueSubsampleNames.map((name) => (
                      <Button
                        key={name}
                        variant={selectedSubsample === name ? "default" : "outline"}
                        size="sm"
                        onClick={() => setSelectedSubsample(name)}
                      >
                        {subsampleLabel(name)}
                      </Button>
                    ))}
                  </div>
                )}
              </div>

              {chessBoards.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {chessBoards}
                </div>
              ) : (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-yellow-800 mb-4">⚠️ No chessboard visualization found</h3>
                  <div className="text-sm text-yellow-700 mb-4">
                    <strong>Feature #{feature.featureIndex}</strong> (Activation times: {feature.actTimes})
                  </div>
                  <div className="text-xs text-yellow-600 mb-4">
                    <strong>Data statistics:</strong><br/>
                    • Total samples: {totalSamples}<br/>
                    • Found FEN: {validFENFound}
                  </div>
                  <div className="text-xs text-yellow-500 mt-4">
                    💡 Possible reasons:<br/>
                    1. Sample does not contain FEN formatted text<br/>
                    2. Sample data structure does not match expectations
                  </div>
                </div>
              )}
            </div>
          );
        })()}
            
        {!featureLoading && currentFeature && (
            <div className="flex gap-12 w-full">
              <Suspense fallback={<div>Loading Feature Card...</div>}>
              <FeatureCard feature={currentFeature} />
              </Suspense>
              <SectionNavigator sections={sections} />
          </div>
        )}
      </div>
    </div>
  );
};
