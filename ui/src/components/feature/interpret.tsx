import { Feature, Interpretation, InterpretationSchema } from "@/types/feature";
import { useState, useCallback, Suspense, useEffect } from "react";
import { Button } from "../ui/button";
import { Ban, Check, Info, ChevronDown, ChevronRight, Copy, CheckCircle2, CircleDashed } from "lucide-react";
import { useAsyncFn } from "react-use";
import { Textarea } from "../ui/textarea";
import camelcaseKeys from "camelcase-keys";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../ui/hover-card";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

const FeatureCustomInterpretionArea = ({
  feature,
  defaultInterpretation,
  onInterpretation,
  onInterpretationSaved,
}: {
  feature: Feature;
  defaultInterpretation: Interpretation | null;
  onInterpretation: (interpretation: Interpretation) => void;
  onInterpretationSaved?: (interpretation: Interpretation) => Promise<void> | void;
}) => {
  const [customInput, setCustomInput] = useState<string>(defaultInterpretation?.text || "");
  const [state, submit] = useAsyncFn(async () => {
    if (!customInput) {
      alert("Please enter your interpretation.");
      return;
    }
    const interpretation = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${feature.dictionaryName}/features/${
        feature.featureIndex
      }/interpret?type=custom&custom_interpretation=${encodeURIComponent(customInput)}`,
      {
        method: "POST",
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => InterpretationSchema.parse(camelcaseKeys(res)));
    onInterpretation(interpretation);
    await onInterpretationSaved?.(interpretation);
    return interpretation;
  }, [customInput, feature.dictionaryName, feature.featureIndex, onInterpretation, onInterpretationSaved]);

  useEffect(() => {
    setCustomInput(defaultInterpretation?.text || "");
  }, [defaultInterpretation?.text, feature.dictionaryName, feature.featureIndex]);

  return (
    <div className="flex flex-col gap-4">
      <Textarea
        placeholder="Type your custom interpretation here."
        value={customInput}
        onChange={(e) => setCustomInput(e.target.value)}
      />
      <Button onClick={submit} disabled={state.loading}>
        Submit
      </Button>
      {state.error && <p className="text-red-500">{state.error.message}</p>}
    </div>
  );
};

export const FeatureInterpretation = ({
  feature,
  onInterpretationSaved,
}: {
  feature: Feature;
  onInterpretationSaved?: (interpretation: Interpretation) => Promise<void> | void;
}) => {
  const [showCustomInput, setShowCustomInput] = useState<boolean>(false);
  const [isUserPromptExpanded, setIsUserPromptExpanded] = useState<boolean>(false);
  const [isSystemPromptExpanded, setIsSystemPromptExpanded] = useState<boolean>(false);
  const [expandedValidationSections, setExpandedValidationSections] = useState<Record<number, Record<string, boolean>>>({});

  const [interpretation, setInterpretation] = useState<Interpretation | null>(feature.interpretation || null);

  const [validating, setValidating] = useState<boolean>(false);

  useEffect(() => {
    setInterpretation(feature.interpretation || null);
  }, [feature.dictionaryName, feature.featureIndex, feature.interpretation]);

  useEffect(() => {
    setShowCustomInput(false);
    setIsUserPromptExpanded(false);
    setIsSystemPromptExpanded(false);
    setExpandedValidationSections({});
    setValidating(false);
  }, [feature.dictionaryName, feature.featureIndex]);

  // Function to highlight tokens within << and >> with colored text
  const highlightTokens = useCallback((text: string): React.ReactNode => {
    if (!text) return "";
    
    const regex = /<<([^>]*)>>/g;
    const parts = [];
    let lastIndex = 0;
    let match;
    
    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      
      // Add the highlighted token with delimiters
      parts.push(
        <span key={match.index} className="text-blue-600 font-medium">
          {`<<${match[1]}>>`}
        </span>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add any remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return <>{parts}</>;
  }, []);

  const testNameMap = useCallback((method: string, passed?: boolean, status?: string) => {
    if (status === "pending") return "Chess Rule Validation Pending";
    if (status === "diagnostic") return "Chess Rule Diagnostic Pending";
    switch (method) {
      case "detection":
        return `Detection Test ${passed ? "Passed" : "Failed"}`;
      case "fuzzing":
        return `Fuzzing Test ${passed ? "Passed" : "Failed"}`;
      case "manual":
        return "Human Validated";
      default:
        return `Unknown Test ${method} ${passed ? "Passed" : "Failed"}`;
    }
  }, []);

  const [state, autoInterpretation] = useAsyncFn(async () => {
    const interpretation = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${
        feature.dictionaryName
      }/features/${feature.featureIndex}/interpret?type=auto`,
      {
        method: "POST",
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => InterpretationSchema.parse(camelcaseKeys(res)));
    setInterpretation(interpretation);
    setValidating(true);
    const interpretationValidated = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${
        feature.dictionaryName
      }/features/${feature.featureIndex}/interpret?type=validate`,
      {
        method: "POST",
      }
    )
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => InterpretationSchema.parse(camelcaseKeys(res)))
      .finally(() => setValidating(false));
    setInterpretation(interpretationValidated);
    await onInterpretationSaved?.(interpretationValidated);
    return interpretationValidated;
  }, [feature.dictionaryName, feature.featureIndex, onInterpretationSaved]);

  // Toggle validation section expand/collapse state
  const toggleValidationSection = (validationIndex: number, section: string) => {
    setExpandedValidationSections((prev) => {
      const newState = { ...prev };
      if (!newState[validationIndex]) {
        newState[validationIndex] = {};
      }
      newState[validationIndex][section] = !newState[validationIndex]?.[section];
      return newState;
    });
  };

  // Check if a validation section is expanded
  const isValidationSectionExpanded = (validationIndex: number, section: string) => {
    return !!expandedValidationSections[validationIndex]?.[section];
  };

  return (
    <div className="flex flex-col w-full gap-4">
      <p className="font-bold">Interpretation</p>
      <div className="flex gap-12">
        <div className="flex flex-col justify-between gap-4 basis-2/3 min-w-2/3">
          {!interpretation && <p className="text-neutral-500">No interpretation available.</p>}
          {interpretation && (
            <div className="flex flex-col gap-4">
              <p>
                {highlightTokens(interpretation.text)}
                {interpretation.detail && (
                  <HoverCard>
                    <HoverCardTrigger>
                      <Info size={20} className="hover:text-blue-500 inline-block mx-2" />
                    </HoverCardTrigger>
                    <HoverCardContent className="w-[800px] max-h-[400px] overflow-y-auto">
                      <div className="flex flex-col gap-4 whitespace-pre-wrap">
                        {interpretation.detail.userPrompt && (
                          <div className="flex flex-col gap-2">
                            <div 
                              className="text-sm font-bold flex items-center cursor-pointer" 
                              onClick={() => setIsUserPromptExpanded(!isUserPromptExpanded)}
                            >
                              {isUserPromptExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                              <span className="ml-1">User Prompt</span>
                            </div>
                            {isUserPromptExpanded && (
                              <div className="text-sm bg-blue-50 p-3 rounded-md border border-blue-200 relative">
                                <CopyButton text={interpretation.detail.userPrompt} />
                                {highlightTokens(interpretation.detail.userPrompt)}
                              </div>
                            )}
                          </div>
                        )}
                        
                        {interpretation.detail.systemPrompt && (
                          <div className="flex flex-col gap-2">
                            <div 
                              className="text-sm font-bold flex items-center cursor-pointer" 
                              onClick={() => setIsSystemPromptExpanded(!isSystemPromptExpanded)}
                            >
                              {isSystemPromptExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                              <span className="ml-1">System Prompt</span>
                            </div>
                            {isSystemPromptExpanded && (
                              <div className="text-sm bg-gray-50 p-3 rounded-md border border-gray-200 relative">
                                <CopyButton text={interpretation.detail.systemPrompt} />
                                {highlightTokens(interpretation.detail.systemPrompt)}
                              </div>
                            )}
                          </div>
                        )}
                        
                        {interpretation.detail.response?.finalExplanation && (
                          <div className="flex flex-col gap-2">
                            <div className="text-sm font-bold">Final Explanation:</div>
                            <div className="text-sm bg-slate-50 p-3 rounded-md border border-slate-200">
                              {highlightTokens(interpretation.detail.response.finalExplanation)}
                            </div>
                          </div>
                        )}
                        
                        {interpretation.detail.response?.steps && interpretation.detail.response.steps.length > 0 && (
                          <div className="flex flex-col gap-2">
                            <div className="text-sm font-bold">Reasoning Steps:</div>
                            <div className="flex flex-col gap-2">
                              {interpretation.detail.response.steps.map((step: string, index: number) => (
                                <div key={index} className="text-sm bg-slate-50 p-2 rounded-md border border-slate-200">
                                  <span className="font-semibold text-slate-500">Step {index + 1}:</span> {highlightTokens(step)}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {(interpretation.detail.response?.activationConsistency !== undefined || 
                          interpretation.detail.response?.complexity !== undefined) && (
                          <div className="grid grid-cols-2 gap-4 mt-2">
                            {interpretation.detail.response?.activationConsistency !== undefined && (
                              <div className="text-sm">
                                <span className="font-bold">Activation Consistency:</span>{" "}
                                {interpretation.detail.response.activationConsistency}
                              </div>
                            )}
                            {interpretation.detail.response?.complexity !== undefined && (
                              <div className="text-sm">
                                <span className="font-bold">Explanation Complexity:</span>{" "}
                                {interpretation.detail.response.complexity}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </HoverCardContent>
                  </HoverCard>
                )}
              </p>
              
              {(interpretation.complexity !== undefined || 
                interpretation.consistency !== undefined || 
                interpretation.time !== undefined) && (
                <div className="flex flex-wrap gap-4 text-sm text-neutral-600">
                  {interpretation.complexity !== undefined && (
                    <div>
                      <span className="font-medium">Complexity:</span> {interpretation.complexity.toFixed(2)}
                    </div>
                  )}
                  {interpretation.consistency !== undefined && (
                    <div>
                      <span className="font-medium">Consistency:</span> {interpretation.consistency.toFixed(2)}
                    </div>
                  )}
                  {interpretation.passed !== undefined && (
                    <div className="flex items-center gap-1">
                      <span className="font-medium">Status:</span>
                      {interpretation.passed ? (
                        <span className="text-green-500 flex items-center gap-1">
                          <Check size={16} /> Passed
                        </span>
                      ) : (
                        <span className="text-red-500 flex items-center gap-1">
                          <Ban size={16} /> Failed
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          <div className="flex gap-4">
            <Button size="sm" onClick={autoInterpretation} disabled={state.loading}>
              Automatic Interpretation
            </Button>
            <Button variant="secondary" size="sm" onClick={() => setShowCustomInput((prev) => !prev)}>
              {showCustomInput
                ? "Hide Custom Interpretation"
                : interpretation
                  ? "Edit Interpretation"
                  : "Provide Custom Interpretation"}
            </Button>
          </div>
        </div>
        <div className="flex flex-col gap-4 basis-1/3 min-w-1/3">
          {interpretation?.validation.map((validation, i) => (
            <div key={i} className="flex items-center gap-2">
              {validation.status === "pending" || validation.status === "diagnostic" ? (
                <CircleDashed size={20} className="text-amber-500" />
              ) : validation.passed ? (
                <Check size={20} className="text-green-500" />
              ) : (
                <Ban size={20} className="text-red-500" />
              )}
              <p>{testNameMap(validation.method, validation.passed, validation.status)}</p>
              {validation.detail && (
                <HoverCard>
                  <HoverCardTrigger>
                    <Info size={20} className="hover:text-blue-500" />
                  </HoverCardTrigger>
                  <HoverCardContent className="w-[800px] flex flex-col gap-4">
                    {validation.method === "activation" && (
                      <>
                        <p className="text-sm">
                          Activation prediction test checks if the LLM can simulate a Sparse Auto Encoder to predict the
                          activation of the feature, given the description of the feature.
                        </p>
                        <p className="text-sm">
                          For simplicity, the LLM is only asked to predict the token with the highest activation. The
                          test is passed if the predicted token is the same as the actual token. White spaces are
                          ignored.
                        </p>
                      </>
                    )}
                    {validation.method === "generative" && (
                      <>
                        <p className="text-sm">
                          Generative test checks if the LLM can generate a sentence that any of the tokens in the
                          sentence can activate the feature.
                        </p>
                        <p className="text-sm">
                          The test is passed if the generated sentence contains at least one token that activates the
                          feature to a certain threshold (currently set to 1.0).
                        </p>
                      </>
                    )}
                    
                    {validation.detail?.prompt && (
                      <div className="flex flex-col gap-2">
                        <div 
                          className="text-sm font-bold flex items-center cursor-pointer" 
                          onClick={() => toggleValidationSection(i, 'prompt')}
                        >
                          {isValidationSectionExpanded(i, 'prompt') ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                          <span className="ml-1">Prompt</span>
                        </div>
                        {isValidationSectionExpanded(i, 'prompt') && (
                          <div className="text-sm bg-blue-50 p-3 rounded-md border border-blue-200 relative">
                            <CopyButton text={validation.detail.prompt} />
                            {highlightTokens(validation.detail.prompt)}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {validation.detail?.response !== undefined && (
                      <div className="flex flex-col gap-2">
                        <div 
                          className="text-sm font-bold flex items-center cursor-pointer" 
                          onClick={() => toggleValidationSection(i, 'response')}
                        >
                          {isValidationSectionExpanded(i, 'response') ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                          <span className="ml-1">Response</span>
                        </div>
                        {isValidationSectionExpanded(i, 'response') && (
                          <div className="text-sm bg-gray-50 p-3 rounded-md border border-gray-200 relative">
                            <CopyButton text={JSON.stringify(validation.detail.response, null, 2)} />
                            <pre className="bg-slate-50 p-2 rounded-md border border-slate-200 overflow-x-auto">
                              {JSON.stringify(validation.detail.response, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    )}
                  </HoverCardContent>
                </HoverCard>
              )}
            </div>
          ))}
          {validating && <p className="text-neutral-500">Validating...</p>}
        </div>
      </div>
      {showCustomInput && (
        <FeatureCustomInterpretionArea
          feature={feature}
          defaultInterpretation={interpretation}
          onInterpretationSaved={onInterpretationSaved}
          onInterpretation={(interpretation) => {
            setInterpretation(interpretation);
            setShowCustomInput(false);
          }}
        />
      )}
    </div>
  );
};

const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };
  
  return (
    <button 
      onClick={handleCopy} 
      className="absolute top-2 right-2 p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-colors"
      aria-label="Copy to clipboard"
    >
      {copied ? <CheckCircle2 size={16} className="text-green-500" /> : <Copy size={16} />}
    </button>
  );
};

/**
 * Reusable card that displays and allows editing a feature's interpretation.
 * Can be used on the features page or inside modals (e.g. get-feature-from-fen).
 */
export const FeatureInterpretationCard = ({
  feature,
  title = "Interpretation",
}: {
  feature: Feature | null;
  title?: string;
}) => {
  if (!feature) {
    return null;
  }

  const featureKey = `${feature.dictionaryName}:${feature.featureIndex}`;

  return (
    <Card>
      {title !== "" && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent className={title === "" ? "pt-6" : undefined}>
        <Suspense fallback={<div>Loading Interpretation...</div>}>
          <FeatureInterpretation key={featureKey} feature={feature} />
        </Suspense>
      </CardContent>
    </Card>
  );
};
