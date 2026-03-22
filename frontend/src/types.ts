export type PredictionClass = 'human' | 'ai' | 'spam'

export type BackendStatus = 'unknown' | 'ok' | 'offline'

export interface Probabilities {
  human: number
  ai: number
  spam: number
}

export interface Activations {
  layer1_sample: number[]
  layer2_sample: number[]
  output: number[]
}

export interface Features {
  char_count: number
  word_count: number
  sentence_count: number
  avg_word_len: number
  avg_sentence_len: number
  spam_keyword_ratio: number
  exclamation_ratio: number
  caps_ratio: number
  url_count: number
  dollar_signs: number
  digit_ratio: number
  repeated_punct: number
  ai_phrase_ratio: number
  type_token_ratio: number
  long_word_ratio: number
  sent_variance: number
  punct_diversity: number
  spam_keywords_found: string[]
  ai_phrases_found: string[]
}

export interface AnalyzeResponse {
  prediction: PredictionClass
  confidence: number
  probabilities: Probabilities
  features: Features
  signals: string[]
  activations: Activations
  architecture: {
    input: number
    hidden1: number
    hidden2: number
    output: number
  }
}

export interface AnalyzeErrorResponse {
  error: string
}

export interface NeuralVizProps {
  activations: Activations | undefined
  probabilities: Probabilities | undefined
  isAnalyzing: boolean
}

export interface ResultBadgeProps {
  prediction: PredictionClass
  confidence: number
  signals: string[]
}

export interface FeaturePanelProps {
  features: Features
}

export interface PredictionConfig {
  label: string
  icon: string
  desc: string
}

export interface FeatureKey {
  key: keyof Features
  label: string
  format: (v: number) => string
  bar?: boolean
  max?: number
}

export interface FeatureGroup {
  label: string
  keys: FeatureKey[]
}