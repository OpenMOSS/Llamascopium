export type SemanticSupernodePosition = {
  token_position?: number | null;
  board_square?: string | null;
  board_squares?: string[];
  rank_file?: unknown;
  label?: string | null;
};

export type SemanticChildFeature = {
  feature_id: string;
  node_id?: string;
  layer?: number | null;
  feature_index?: number | null;
  feature_type?: string | null;
  dictionary_name?: string | null;
  token_position?: number | null;
  board_square?: string | null;
  board_squares?: string[];
  interpretation?: string;
  taxonomy?: string | null;
  activation?: number | null;
  metadata?: Record<string, unknown>;
  raw?: Record<string, unknown>;
};

export type SemanticSupernode = {
  supernode_id: string;
  label: string;
  semantic_type?: string | null;
  role_id?: string | null;
  interpretation?: string;
  position: SemanticSupernodePosition;
  member_feature_ids: string[];
  evidence?: Record<string, unknown>;
  visual?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  raw?: Record<string, unknown>;
};

export type SemanticSupernodeEdge = {
  edge_id: string;
  source: string;
  target: string;
  weight?: number | null;
  label?: string;
  relation_type?: string | null;
  member_edge_ids?: string[];
  metadata?: Record<string, unknown>;
  raw?: Record<string, unknown>;
};

export type SemanticChildFeatureEdge = SemanticSupernodeEdge;

export type SemanticSupernodeGraph = {
  schema_version: string;
  metadata?: Record<string, unknown>;
  board?: {
    fen?: string;
    highlights?: string[];
    [key: string]: unknown;
  };
  action?: {
    san?: string;
    uci?: string;
    prob?: number;
    [key: string]: unknown;
  };
  child_features: SemanticChildFeature[];
  child_feature_edges?: SemanticChildFeatureEdge[];
  semantic_supernodes: SemanticSupernode[];
  semantic_edges: SemanticSupernodeEdge[];
  warnings?: string[];
  raw?: Record<string, unknown>;
};
